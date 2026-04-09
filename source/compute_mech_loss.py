import torch

from thermo_mech_model import nodal_to_element


def compute_mech_loss(inp, ux, uy, T_phys, d_prev, area_elem, T_conn, thermo_model, thermal_prop, scale_dict=None):
    """
    Mechanical substep loss for staggered thermo-mechanical/phase coupling.
    This substep uses d_prev (i.e., d_n) while solving u_{n+1}; then phase is updated.
    IMPORTANT: T_phys must be mapped physical temperature (not raw network output).
    """
    d_elem = nodal_to_element(d_prev, T_conn)
    exx_e, eyy_e, exy_e, ezz_e = thermo_model.kinematics(
        inp=inp,
        ux=ux,
        uy=uy,
        T=T_phys,
        alpha=thermal_prop.alpha,
        Tref=thermal_prop.Tref,
        area_elem=area_elem,
        T_conn=T_conn,
    )

    mixed = thermo_model.mixed_mode_terms(exx_e, eyy_e, exy_e, ezz_e)
    psi_I = mixed["psi_I"]
    psi_II = mixed["psi_II"]
    psi_plus = psi_I + psi_II

    tr_e = mixed["tr_e"]
    psi_lin = 0.5 * thermo_model.lam * tr_e ** 2 + thermo_model.mu * (
        exx_e ** 2 + eyy_e ** 2 + ezz_e ** 2 + 2.0 * exy_e ** 2
    )
    psi_minus = psi_lin - psi_plus

    g_d = thermo_model.degradation(d_elem)
    psi_mech = psi_minus + g_d * psi_plus
    psi_mech = torch.nan_to_num(psi_mech, nan=0.0, posinf=1e20, neginf=-1e20)
    loss_mech = torch.sum(psi_mech * area_elem)
    loss_mech = torch.nan_to_num(loss_mech, nan=1e20, posinf=1e20, neginf=1e20)
    if scale_dict is None:
        mech_loss_ref = torch.tensor(1.0, device=inp.device, dtype=loss_mech.dtype)
    else:
        mech_loss_ref = torch.tensor(
            float(scale_dict.get("mech_loss_ref", 1.0)),
            device=inp.device,
            dtype=loss_mech.dtype,
        )
    mech_loss_ref = torch.clamp(
        mech_loss_ref,
        min=torch.tensor(1e-18, device=inp.device, dtype=loss_mech.dtype),
    )
    loss_mech_nd = loss_mech / mech_loss_ref

    stress = thermo_model.stress_split(exx_e, eyy_e, exy_e, ezz_e, mixed, d_elem)

    state = {
        "exx_e": exx_e,
        "eyy_e": eyy_e,
        "exy_e": exy_e,
        "ezz_e": ezz_e,
        "psi_I": psi_I,
        "psi_II": psi_II,
        "psi_plus": psi_plus,
        "psi_minus": psi_minus,
        "psi_mech": psi_mech,
        "stress": stress,
        "mech_raw": loss_mech,
        "mech_nd": loss_mech_nd,
    }

    return loss_mech, state
