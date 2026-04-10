import torch


class MonolithicTMPhaseFieldComputation:
    """
    Physical mapping for monolithic TM-phase network.

    Network raw outputs:
      - T_raw(x, y)
      - ux_raw(x, y)
      - uy_raw(x, y)
      - d_raw(x, y)

    Physical maps:
      - T_phys = map_temperature(T_raw)
      - u_phys = map_displacement(ux_raw, uy_raw)
      - d_phys = d_prev + (1-d_prev)*sigmoid(d_raw)
    """

    def __init__(
        self,
        net,
        domain_extrema,
        time,
        uy_rate,
        T_shift,
        T_scale=1.0,
        U_scale=1.0,
        bottom_fix_mode="uxuy",
        enforce_nodal_clamp=False,
    ):
        self.net = net
        self.domain_extrema = domain_extrema
        self.time = time
        self.uy_rate = uy_rate
        self.T_shift = T_shift
        self.T_scale = T_scale
        self.U_scale = U_scale
        self.bottom_fix_mode = bottom_fix_mode
        self.enforce_nodal_clamp = bool(enforce_nodal_clamp)
        self.d_prev = None
        self._boundary_desc_cache = {}
        span = self.domain_extrema[:, 1] - self.domain_extrema[:, 0]
        self.length_char = torch.clamp(
            torch.max(span),
            min=torch.tensor(1e-12, device=span.device, dtype=span.dtype),
        )

    def set_time(self, time_value):
        self.time = time_value

    def set_prev_damage(self, d_prev):
        self.d_prev = d_prev.detach()

    def _distance_to_segment(self, pts, p0, p1):
        # pts: (N,2), p0/p1: (2,)
        v = p1 - p0
        vv = torch.sum(v * v) + torch.tensor(1e-18, device=pts.device, dtype=pts.dtype)
        t = torch.sum((pts - p0) * v, dim=1) / vv
        t = torch.clamp(t, min=0.0, max=1.0)
        proj = p0.unsqueeze(0) + t.unsqueeze(1) * v.unsqueeze(0)
        return torch.norm(pts - proj, dim=1)

    def _get_boundary_desc(self, inp, node_ids):
        if node_ids is None or node_ids.numel() == 0:
            return None
        key = tuple(torch.sort(node_ids.detach().cpu().long())[0].tolist())
        if key in self._boundary_desc_cache:
            return self._boundary_desc_cache[key]

        bpts = inp[node_ids, -2:]
        if bpts.shape[0] == 1:
            desc = {"type": "point", "p": bpts[0].detach()}
            self._boundary_desc_cache[key] = desc
            return desc

        center = torch.mean(bpts, dim=0)
        centered = bpts - center.unsqueeze(0)
        # expensive SVD is cached by node-id signature
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        tangent = vh[0]
        tangent = tangent / (torch.norm(tangent) + torch.tensor(1e-18, device=tangent.device, dtype=tangent.dtype))
        proj = torch.sum(centered * tangent.unsqueeze(0), dim=1)
        p0 = (center + torch.min(proj) * tangent).detach()
        p1 = (center + torch.max(proj) * tangent).detach()
        desc = {"type": "segment", "p0": p0, "p1": p1}
        self._boundary_desc_cache[key] = desc
        return desc

    def _boundary_phi_from_nodes(self, inp, node_ids):
        if node_ids is None or node_ids.numel() == 0:
            return None
        pts = inp[:, -2:]
        desc = self._get_boundary_desc(inp, node_ids)
        if desc is None:
            return None
        if desc["type"] == "point":
            d = torch.norm(pts - desc["p"].unsqueeze(0), dim=1)
            return d / self.length_char
        d = self._distance_to_segment(pts, desc["p0"], desc["p1"])
        return d / self.length_char

    def _r_equivalence(self, phi_list, m=2.0):
        if len(phi_list) == 0:
            return None
        if len(phi_list) == 1:
            return phi_list[0]
        eps = torch.tensor(1e-12, device=phi_list[0].device, dtype=phi_list[0].dtype)
        inv_sum = torch.zeros_like(phi_list[0])
        for phi in phi_list:
            inv_sum = inv_sum + (1.0 / (phi + eps)) ** m
        return inv_sum ** (-1.0 / m)

    def _build_g_transfinite(self, phi_list, g_list):
        if len(phi_list) == 0:
            return None
        if len(phi_list) == 1:
            g0 = g_list[0]
            if torch.is_tensor(g0):
                return g0
            return torch.full_like(phi_list[0], float(g0))

        eps = torch.tensor(1e-12, device=phi_list[0].device, dtype=phi_list[0].dtype)
        w_sum = torch.zeros_like(phi_list[0])
        gw_sum = torch.zeros_like(phi_list[0])
        for phi, g in zip(phi_list, g_list):
            w = 1.0 / (phi + eps)
            if torch.is_tensor(g):
                g_use = g
            else:
                g_use = torch.full_like(phi, float(g))
            w_sum = w_sum + w
            gw_sum = gw_sum + w * g_use
        return gw_sum / (w_sum + eps)

    def _extend_boundary_values_nearest(self, inp, boundary_nodes, boundary_values):
        pts = inp[:, -2:]
        bpts = inp[boundary_nodes, -2:]
        d2 = torch.sum((pts.unsqueeze(1) - bpts.unsqueeze(0)) ** 2, dim=2)
        nearest = torch.argmin(d2, dim=1)
        return boundary_values[nearest]

    def _build_anchor_patch_nodes(self, inp, mechanical_bottom_nodes, fixed_point_nodes):
        if fixed_point_nodes is None or fixed_point_nodes.numel() == 0:
            return None
        if mechanical_bottom_nodes is None or mechanical_bottom_nodes.numel() == 0:
            return fixed_point_nodes

        x_all = inp[:, -2]
        x_bottom = x_all[mechanical_bottom_nodes]
        x_anchor = torch.mean(x_all[fixed_point_nodes])
        x_sorted, _ = torch.sort(x_bottom)
        if x_sorted.numel() >= 3:
            dx = x_sorted[1:] - x_sorted[:-1]
            dx = dx[dx > 0]
            if dx.numel() > 0:
                dx_med = torch.median(dx)
            else:
                dx_med = torch.tensor(0.0, device=inp.device, dtype=inp.dtype)
        else:
            dx_med = torch.tensor(0.0, device=inp.device, dtype=inp.dtype)

        Lx = torch.clamp(
            self.domain_extrema[0, 1] - self.domain_extrema[0, 0],
            min=torch.tensor(1e-12, device=inp.device, dtype=inp.dtype),
        )
        half_width = torch.maximum(
            2.0 * dx_med,
            0.01 * Lx,
        )
        mask = torch.abs(x_bottom - x_anchor) <= half_width
        if torch.any(mask):
            patch = mechanical_bottom_nodes[mask]
            return torch.unique(torch.cat((patch, fixed_point_nodes)))
        return fixed_point_nodes

    def build_anchor_patch_nodes(self, inp, mechanical_bottom_nodes, fixed_point_nodes):
        """
        Public helper used by training orchestrators to build a tiny gauge patch
        around the fixed point for roller-mode rigid-body removal.
        """
        return self._build_anchor_patch_nodes(inp, mechanical_bottom_nodes, fixed_point_nodes)

    def map_temperature(self, T_raw, thermal_dirichlet_nodes=None, T_bc_value=None):
        # Sukumar-style hard Dirichlet ansatz:
        # T = G_T + Phi_T * T_tilde, with Phi_T|Gamma_D = 0 and G_T|Gamma_D = T_bc.
        T_ref = self.T_shift
        DT_ref = self.T_scale
        if not torch.is_tensor(T_ref):
            T_ref = torch.tensor(T_ref, dtype=T_raw.dtype, device=T_raw.device)
        T_ref = T_ref.to(device=T_raw.device, dtype=T_raw.dtype)
        if not torch.is_tensor(DT_ref):
            DT_ref = torch.tensor(DT_ref, dtype=T_raw.dtype, device=T_raw.device)
        DT_ref = torch.clamp(
            DT_ref.to(device=T_raw.device, dtype=T_raw.dtype),
            min=torch.tensor(1e-12, device=T_raw.device, dtype=T_raw.dtype),
        )

        T_bc = T_ref if T_bc_value is None else T_bc_value
        if not torch.is_tensor(T_bc):
            T_bc = torch.tensor(T_bc, dtype=T_raw.dtype, device=T_raw.device)
        T_bc = T_bc.to(device=T_raw.device, dtype=T_raw.dtype)

        if thermal_dirichlet_nodes is None or thermal_dirichlet_nodes.numel() == 0:
            return T_ref + DT_ref * T_raw

        phi_T = self._boundary_phi_from_nodes(inp=self._last_inp, node_ids=thermal_dirichlet_nodes)
        if phi_T is None:
            return T_ref + DT_ref * T_raw

        T_bc_bar = (T_bc - T_ref) / DT_ref

        if T_bc_bar.numel() == thermal_dirichlet_nodes.numel():
            G_T = self._extend_boundary_values_nearest(
                inp=self._last_inp,
                boundary_nodes=thermal_dirichlet_nodes,
                boundary_values=T_bc_bar.reshape(-1),
            )
        else:
            G_T = self._build_g_transfinite([phi_T], [T_bc_bar])
        T_bar = G_T + phi_T * T_raw
        T = T_ref + DT_ref * T_bar

        if self.enforce_nodal_clamp:
            # Optional exact nodal clamp for audit/legacy comparison only.
            T = T.clone()
            if T_bc.numel() == 1:
                T[thermal_dirichlet_nodes] = T_bc
            elif T_bc.numel() == thermal_dirichlet_nodes.numel():
                T[thermal_dirichlet_nodes] = T_bc.reshape(-1)
            else:
                raise ValueError("T_bc_value must be scalar or have same length as thermal_dirichlet_nodes.")
        return T

    def map_displacement(
        self,
        inp,
        ux_raw,
        uy_raw,
        mechanical_top_nodes=None,
        mechanical_bottom_nodes=None,
        fixed_point_nodes=None,
        uy_top_value=None,
    ):
        # Cache input for geometry-aware ADF in map_temperature.
        self._last_inp = inp

        uy_top_curr = self.uy_rate * self.time if uy_top_value is None else uy_top_value
        U_ref = self.U_scale
        if not torch.is_tensor(U_ref):
            U_ref = torch.tensor(U_ref, device=inp.device, dtype=inp.dtype)
        U_ref = torch.clamp(
            torch.abs(U_ref.to(device=inp.device, dtype=inp.dtype)),
            min=torch.tensor(1e-12, device=inp.device, dtype=inp.dtype),
        )

        bottom_fix_mode = str(self.bottom_fix_mode).lower()
        is_roller = bottom_fix_mode in ("uy_only", "y_only", "roller")

        # ===== u_x ansatz (component-wise hard BC) =====
        ux_dir_nodes = []
        ux_g_values = []
        if not is_roller and mechanical_bottom_nodes is not None and mechanical_bottom_nodes.numel() > 0:
            # uxuy mode: bottom ux = 0
            ux_dir_nodes.append(mechanical_bottom_nodes)
            ux_g_values.append(torch.tensor(0.0, device=inp.device, dtype=inp.dtype))
        if fixed_point_nodes is not None and fixed_point_nodes.numel() > 0:
            # Roller mode: use a tiny anchor patch (strategy A) rather than ad-hoc shape factors.
            anchor_nodes = self._build_anchor_patch_nodes(inp, mechanical_bottom_nodes, fixed_point_nodes)
            if anchor_nodes is not None and anchor_nodes.numel() > 0:
                ux_dir_nodes.append(anchor_nodes)
                ux_g_values.append(torch.tensor(0.0, device=inp.device, dtype=inp.dtype))

        ux_phi_list = []
        for nodes in ux_dir_nodes:
            phi_i = self._boundary_phi_from_nodes(inp, nodes)
            if phi_i is not None:
                ux_phi_list.append(phi_i)

        if len(ux_phi_list) == 0:
            Phi_x = torch.ones_like(ux_raw)
            G_x = torch.zeros_like(ux_raw)
        else:
            Phi_x = self._r_equivalence(ux_phi_list, m=2.0)
            G_x = self._build_g_transfinite(ux_phi_list, ux_g_values[: len(ux_phi_list)])
        ux_bar = G_x + Phi_x * ux_raw
        ux = U_ref * ux_bar

        # ===== u_y ansatz (component-wise hard BC) =====
        uy_phi_list = []
        uy_g_list = []
        if mechanical_bottom_nodes is not None and mechanical_bottom_nodes.numel() > 0:
            phi_b = self._boundary_phi_from_nodes(inp, mechanical_bottom_nodes)
            if phi_b is not None:
                uy_phi_list.append(phi_b)
                uy_g_list.append(torch.tensor(0.0, device=inp.device, dtype=inp.dtype))
        if mechanical_top_nodes is not None and mechanical_top_nodes.numel() > 0:
            phi_t = self._boundary_phi_from_nodes(inp, mechanical_top_nodes)
            if phi_t is not None:
                uy_phi_list.append(phi_t)
                if not torch.is_tensor(uy_top_curr):
                    uy_top_curr = torch.tensor(uy_top_curr, device=inp.device, dtype=inp.dtype)
                uy_g_list.append(uy_top_curr.to(device=inp.device, dtype=inp.dtype) / U_ref)

        if len(uy_phi_list) == 0:
            G_y = torch.zeros_like(uy_raw)
            Phi_y = torch.ones_like(uy_raw)
        else:
            G_y = self._build_g_transfinite(uy_phi_list, uy_g_list)
            Phi_y = self._r_equivalence(uy_phi_list, m=2.0)
        uy_bar = G_y + Phi_y * uy_raw
        uy = U_ref * uy_bar

        if self.enforce_nodal_clamp:
            if mechanical_bottom_nodes is not None and mechanical_bottom_nodes.numel() > 0:
                uy = uy.clone()
                uy[mechanical_bottom_nodes] = 0.0

            if fixed_point_nodes is not None and fixed_point_nodes.numel() > 0:
                ux = ux.clone()
                uy = uy.clone()
                # In roller mode, fixed-point anchoring is represented by anchor patch in ansatz.
                # Keep explicit point clamp only for non-roller to preserve legacy semantics.
                if not is_roller:
                    ux[fixed_point_nodes] = 0.0
                    uy[fixed_point_nodes] = 0.0

            if mechanical_top_nodes is not None and mechanical_top_nodes.numel() > 0:
                uy_val = self.uy_rate * self.time if uy_top_value is None else uy_top_value
                if not torch.is_tensor(uy_val):
                    uy_val = torch.tensor(uy_val, dtype=uy.dtype, device=uy.device)
                else:
                    uy_val = uy_val.to(device=uy.device, dtype=uy.dtype)
                uy = uy.clone()
                uy[mechanical_top_nodes] = uy_val

        return ux, uy

    def map_phase_field(self, d_raw, d_prev=None):
        d_bar = torch.sigmoid(d_raw)
        d_prev_use = self.d_prev if d_prev is None else d_prev
        if d_prev_use is None:
            d_prev_use = torch.zeros_like(d_bar)
        d = d_prev_use + (1.0 - d_prev_use) * d_bar
        return torch.clamp(d, min=0.0, max=1.0)

    def forward_raw(self, inp):
        return self.net.forward_raw(inp)

    def fieldCalculation_tm(
        self,
        inp,
        thermal_dirichlet_nodes=None,
        T_bc_value=None,
        mechanical_top_nodes=None,
        mechanical_bottom_nodes=None,
        fixed_point_nodes=None,
    ):
        self._last_inp = inp
        T_raw, ux_raw, uy_raw, d_raw = self.forward_raw(inp=inp)
        T_phys = self.map_temperature(
            T_raw=T_raw,
            thermal_dirichlet_nodes=thermal_dirichlet_nodes,
            T_bc_value=T_bc_value,
        )
        ux, uy = self.map_displacement(
            inp=inp,
            ux_raw=ux_raw,
            uy_raw=uy_raw,
            mechanical_top_nodes=mechanical_top_nodes,
            mechanical_bottom_nodes=mechanical_bottom_nodes,
            fixed_point_nodes=fixed_point_nodes,
        )
        return T_phys, ux, uy, d_raw

    def fieldCalculation_phase(self, inp, d_prev=None, return_raw=False):
        _, _, _, d_raw = self.forward_raw(inp=inp)
        d = self.map_phase_field(d_raw=d_raw, d_prev=d_prev)
        if return_raw:
            return d, d_raw
        return d
