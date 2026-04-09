import torch

from compute_energy import field_grads


def nodal_to_element(field, T_conn):
    if T_conn is None:
        return field
    return (field[T_conn[:, 0]] + field[T_conn[:, 1]] + field[T_conn[:, 2]]) / 3.0


def element_to_nodal(field_elem, T_conn, n_nodes, area_elem=None):
    if T_conn is None:
        return field_elem

    nodal_field = torch.zeros((n_nodes,), device=field_elem.device)
    nodal_weight = torch.zeros((n_nodes,), device=field_elem.device)

    if area_elem is None:
        area_elem = torch.ones_like(field_elem)

    for i in range(3):
        idx = T_conn[:, i]
        nodal_field.index_add_(0, idx, field_elem * area_elem)
        nodal_weight.index_add_(0, idx, area_elem)

    # Use dtype-aware tiny only for truly zero weights.
    # IMPORTANT: do not clamp to float32 eps (~1e-7), because element areas in
    # micron-scale meshes can be ~1e-16 and would be over-clamped, causing
    # severe magnitude compression when mapping element fields to nodes.
    tiny = torch.finfo(nodal_weight.dtype).tiny
    nodal_weight_safe = torch.where(
        nodal_weight > 0.0,
        nodal_weight,
        torch.full_like(nodal_weight, tiny),
    )
    return nodal_field / nodal_weight_safe


class ThermoMechModel:
    """
    Thermo-mechanical constitutive model with mixed-mode driving force.
    """

    def __init__(self, E0, v0, GcI, GcII, kappa, l0, etaPF, eps_r):
        self.E0 = E0
        self.v0 = v0
        self.GcI = GcI
        self.GcII = GcII
        self.kappa = kappa
        self.l0 = l0
        self.etaPF = etaPF
        self.eps_r = eps_r

        self.mu = self.E0 / (2.0 * (1.0 + self.v0))
        self.lam = self.E0 * self.v0 / ((1.0 + self.v0) * (1.0 - 2.0 * self.v0))
        self.gc_ratio = self.GcI / self.GcII

    def degradation(self, d):
        return (1.0 - d) ** 2 + self.kappa

    def thermal_conductivity(self, d, k0):
        return self.degradation(d) * k0

    def kinematics(self, inp, ux, uy, T, alpha, Tref, area_elem, T_conn=None):
        dux_dx, dux_dy = field_grads(inp, ux, area_elem, T_conn)
        duy_dx, duy_dy = field_grads(inp, uy, area_elem, T_conn)

        T_elem = nodal_to_element(T, T_conn)
        thermal_eps = alpha * (T_elem - Tref)

        exx_e = dux_dx - thermal_eps
        eyy_e = duy_dy - thermal_eps
        exy_e = 0.5 * (dux_dy + duy_dx)
        ezz_e = -self.v0 / (1.0 - self.v0) * (exx_e + eyy_e)

        return exx_e, eyy_e, exy_e, ezz_e

    def mixed_mode_terms(self, exx_e, eyy_e, exy_e, ezz_e):
        ed = 0.5 * (exx_e - eyy_e)
        em = 0.5 * (exx_e + eyy_e)
        r = torch.sqrt(ed ** 2 + exy_e ** 2 + self.eps_r ** 2)
        r_safe = torch.clamp(r, min=self.eps_r)

        e1 = em + r
        e2 = em - r
        e1p = 0.5 * (e1 + torch.abs(e1))
        e2p = 0.5 * (e2 + torch.abs(e2))
        e3p = 0.5 * (ezz_e + torch.abs(ezz_e))

        # Numerically robust mode decomposition ratios.
        chi = torch.clamp(ed / r_safe, min=-1.0, max=1.0)
        eta = exy_e / r_safe

        sp = e1p + e2p
        dp = e1p - e2p
        epxx = 0.5 * sp + 0.5 * dp * chi
        epyy = 0.5 * sp - 0.5 * dp * chi
        epxy = 0.5 * dp * eta
        epzz = e3p

        tr_e = exx_e + eyy_e + ezz_e
        tr_p = 0.5 * (tr_e + torch.abs(tr_e))
        ep2 = epxx ** 2 + epyy ** 2 + epzz ** 2 + 2.0 * epxy ** 2

        psi_I = 0.5 * self.lam * tr_p ** 2
        psi_II = self.mu * ep2

        return {
            "ed": ed,
            "em": em,
            "r": r,
            "e1": e1,
            "e2": e2,
            "e1p": e1p,
            "e2p": e2p,
            "e3p": e3p,
            "chi": chi,
            "eta": eta,
            "sp": sp,
            "dp": dp,
            "epxx": epxx,
            "epyy": epyy,
            "epxy": epxy,
            "epzz": epzz,
            "tr_e": tr_e,
            "tr_p": tr_p,
            "ep2": ep2,
            "psi_I": psi_I,
            "psi_II": psi_II,
        }

    def stress_split(self, exx_e, eyy_e, exy_e, ezz_e, mixed_terms, d_elem):
        tr_e = mixed_terms["tr_e"]
        tr_p = mixed_terms["tr_p"]
        epxx = mixed_terms["epxx"]
        epyy = mixed_terms["epyy"]
        epxy = mixed_terms["epxy"]
        epzz = mixed_terms["epzz"]

        sig_xx_lin = self.lam * tr_e + 2.0 * self.mu * exx_e
        sig_yy_lin = self.lam * tr_e + 2.0 * self.mu * eyy_e
        sig_xy_lin = 2.0 * self.mu * exy_e
        sig_zz_lin = self.lam * tr_e + 2.0 * self.mu * ezz_e

        sig_xx_plus = self.lam * tr_p + 2.0 * self.mu * epxx
        sig_yy_plus = self.lam * tr_p + 2.0 * self.mu * epyy
        sig_xy_plus = 2.0 * self.mu * epxy
        sig_zz_plus = self.lam * tr_p + 2.0 * self.mu * epzz

        sig_xx_minus = sig_xx_lin - sig_xx_plus
        sig_yy_minus = sig_yy_lin - sig_yy_plus
        sig_xy_minus = sig_xy_lin - sig_xy_plus
        sig_zz_minus = sig_zz_lin - sig_zz_plus

        g_d = self.degradation(d_elem)
        sig_xx = sig_xx_minus + g_d * sig_xx_plus
        sig_yy = sig_yy_minus + g_d * sig_yy_plus
        sig_xy = sig_xy_minus + g_d * sig_xy_plus
        sig_zz = sig_zz_minus + g_d * sig_zz_plus

        return {
            "sig_xx": sig_xx,
            "sig_yy": sig_yy,
            "sig_xy": sig_xy,
            "sig_zz": sig_zz,
            "sig_xx_plus": sig_xx_plus,
            "sig_yy_plus": sig_yy_plus,
            "sig_xy_plus": sig_xy_plus,
            "sig_zz_plus": sig_zz_plus,
            "sig_xx_minus": sig_xx_minus,
            "sig_yy_minus": sig_yy_minus,
            "sig_xy_minus": sig_xy_minus,
            "sig_zz_minus": sig_zz_minus,
        }
