from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from compute_mech_loss import compute_mech_loss
from compute_phase_loss import compute_phase_loss
from compute_thermal_loss import compute_thermal_loss, get_thermal_dirichlet_target
from fit import fit_tm
from state_manager import AcceptedState, CandidateState, InstantaneousState, PathDependentStateManager
from thermo_mech_model import element_to_nodal


def _to_float(val, default=0.0):
    if torch.is_tensor(val):
        return float(val.detach().cpu().item())
    if val is None:
        return float(default)
    return float(val)


def _l2_param_sum(params):
    total = None
    for p in params:
        if not p.requires_grad:
            continue
        term = torch.sum(p ** 2)
        total = term if total is None else total + term
    if total is None:
        return torch.tensor(0.0)
    return total


def _make_auto_weights(term_map, cfg, prev_weights=None, magnitude_map=None):
    eps = float(cfg.get("eps", 1e-12))
    clip_min = float(cfg.get("clip_min", 0.2))
    clip_max = float(cfg.get("clip_max", 5.0))
    beta = float(cfg.get("ema_beta", 0.9))
    normalize = bool(cfg.get("normalize_mean_to_one", True))

    names = list(term_map.keys())
    init_vals = []
    for k in names:
        if magnitude_map is not None and k in magnitude_map:
            v = _to_float(magnitude_map[k], default=0.0)
        else:
            v = abs(_to_float(term_map[k], default=0.0))
        init_vals.append(v)

    w_init = {}
    for k, v in zip(names, init_vals):
        w = 1.0 / (v + eps)
        w = float(np.clip(w, clip_min, clip_max))
        w_init[k] = w

    if normalize and len(w_init) > 0:
        m = max(float(np.mean(list(w_init.values()))), eps)
        for k in w_init:
            w_init[k] /= m

    if prev_weights is None:
        return w_init

    w_new = {}
    for k in names:
        prev = float(prev_weights.get(k, w_init[k]))
        w_new[k] = beta * prev + (1.0 - beta) * w_init[k]

    if normalize and len(w_new) > 0:
        m = max(float(np.mean(list(w_new.values()))), eps)
        for k in w_new:
            w_new[k] /= m
    return w_new


def _weighted_sum_nd(term_map, weight_map):
    total = None
    for k, v in term_map.items():
        c = float(weight_map.get(k, 1.0))
        term = c * v
        total = term if total is None else total + term
    if total is None:
        raise ValueError("No terms provided for weighted sum")
    return total


def _relative_change(curr, prev, eps=1e-12, floor=1e-8):
    num = torch.norm(curr - prev)
    den = torch.maximum(
        torch.maximum(torch.norm(prev), torch.norm(curr)),
        torch.tensor(floor, device=curr.device, dtype=curr.dtype),
    )
    return (num / (den + eps)).item()


@dataclass
class StepAdvanceResult:
    accepted_state: AcceptedState
    instant_state: InstantaneousState
    candidate_state: CandidateState
    step_loss_series: Dict[str, list]
    loss_row: Dict[str, float]
    convergence: Dict[str, float]
    branch_weight_state: Dict[str, Optional[Dict[str, float]]]


class SolverOrchestrator:
    """State-advancer / solver-orchestrator for staggered TM-phase coupling."""

    def __init__(
        self,
        field_comp,
        thermo_model,
        thermal_prop,
        inp,
        T_conn,
        area_T,
        bc_dict,
        optimizer_dict,
        training_dict,
        runtime_scale,
        writer=None,
    ):
        self.field_comp = field_comp
        self.thermo_model = thermo_model
        self.thermal_prop = thermal_prop
        self.inp = inp
        self.T_conn = T_conn
        self.area_T = area_T
        self.bc_dict = bc_dict
        self.optimizer_dict = optimizer_dict
        self.training_dict = training_dict
        self.runtime_scale = runtime_scale
        self.writer = writer

        self.device = inp.device
        self.dtype = inp.dtype

        self.top_nodes = bc_dict.get("mechanical_top_nodes", bc_dict.get("top_nodes"))
        self.bottom_nodes = bc_dict.get("mechanical_bottom_nodes", bc_dict.get("bottom_nodes"))
        self.fixed_nodes = bc_dict.get("fixed_point_nodes", bc_dict.get("point1_node"))
        self.thermal_nodes = bc_dict.get("thermal_dirichlet_nodes")

        self.bottom_fix_mode = str(training_dict.get("bottom_fix_mode", "uxuy")).lower()
        self.is_roller = self.bottom_fix_mode in ("uy_only", "y_only", "roller")

        self.use_ux_gauge = bool(training_dict.get("use_ux_gauge", False)) and self.is_roller
        if self.use_ux_gauge and self.fixed_nodes is not None and self.fixed_nodes.numel() > 0:
            gauge_nodes = self.field_comp.build_anchor_patch_nodes(
                inp=inp,
                mechanical_bottom_nodes=self.bottom_nodes,
                fixed_point_nodes=self.fixed_nodes,
            )
            self.ux_gauge_nodes = (
                gauge_nodes if (gauge_nodes is not None and gauge_nodes.numel() > 0) else self.fixed_nodes
            )
        else:
            self.ux_gauge_nodes = torch.empty((0,), dtype=torch.long, device=self.device)

        self.auto_weight_cfg = dict(training_dict.get("auto_weight_dict", {}))
        self.auto_weight_enabled = bool(self.auto_weight_cfg.get("enabled", False))
        self.branch_auto_cfg = dict(training_dict.get("branch_auto_weight_dict", {}))
        self.thermal_auto = self.auto_weight_enabled and bool(self.branch_auto_cfg.get("thermal", True))
        self.mech_auto = self.auto_weight_enabled and bool(self.branch_auto_cfg.get("mechanical", True))
        self.phase_auto = self.auto_weight_enabled and bool(self.branch_auto_cfg.get("phase", True))

        self.phase_term_cfg = dict(training_dict.get("phase_term_weight_dict", {}))

        self.tol_loss = float(training_dict.get("tol_loss", 1e-4))
        self.tol_field_T = float(training_dict.get("tol_field_T", 1e-3))
        self.tol_field_u = float(training_dict.get("tol_field_u", 1e-3))
        self.tol_field_d = float(training_dict.get("tol_field_d", 1e-3))
        self.tol_hist = float(training_dict.get("tol_hist", 1e-4))
        self.max_inner_iters = int(training_dict.get("max_inner_iters", 3))
        self.conv_patience = int(training_dict.get("conv_patience", 2))
        self.loss_transform_mode = str(training_dict.get("loss_transform_mode", "raw")).lower()
        self.l2_reg_coeff = float(training_dict.get("w_l2_reg", 0.0))

        self.phase_solver_cfg = dict(training_dict.get("phase_solver_dict", {}))
        self.phase_solver_enabled = bool(self.phase_solver_cfg.get("enabled", True))
        self.phase_solver_rounds = int(self.phase_solver_cfg.get("rounds", 2))
        self.phase_solver_patience = int(self.phase_solver_cfg.get("patience", 1))
        self.phase_solver_tol_d = float(self.phase_solver_cfg.get("tol_d", self.tol_field_d))
        self.phase_weight_global = float(training_dict.get("w_phase_global", 1.0))
        self.w_irrev = float(training_dict.get("w_irrev", 1.0))

    def _transform_loss(self, value):
        mode = self.loss_transform_mode
        eps = 1e-12
        if mode == "raw":
            return value
        if mode == "log1p":
            return torch.log1p(torch.clamp(value, min=0.0))
        if mode == "log10":
            return torch.log10(torch.clamp(value, min=0.0) + eps)
        if mode == "signed_log1p":
            return torch.sign(value) * torch.log1p(torch.abs(value))
        if mode == "signed_log10":
            return torch.sign(value) * torch.log10(torch.abs(value) + eps)
        raise ValueError(f"Unknown loss_transform_mode: {mode}")

    def _current_T_bc_target(self, t_now):
        return get_thermal_dirichlet_target(
            bc_dict=self.bc_dict,
            thermal_prop=self.thermal_prop,
            time_value=t_now,
            idx_dirichlet=self.thermal_nodes,
            device=self.device,
            dtype=self.dtype,
        )

    def _compute_bc_u(self, ux, uy, uy_top_target):
        terms = []
        if self.top_nodes is not None and self.top_nodes.numel() > 0:
            terms.append(torch.mean((uy[self.top_nodes] - uy_top_target) ** 2))
        if self.bottom_nodes is not None and self.bottom_nodes.numel() > 0:
            terms.append(torch.mean(uy[self.bottom_nodes] ** 2))
            if not self.is_roller:
                terms.append(torch.mean(ux[self.bottom_nodes] ** 2))
        if (self.fixed_nodes is not None) and (self.fixed_nodes.numel() > 0) and (not self.is_roller):
            terms.append(torch.mean(ux[self.fixed_nodes] ** 2 + uy[self.fixed_nodes] ** 2))
        if len(terms) == 0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        return sum(terms)

    def solve_thermal_step(
        self,
        step_idx,
        inner_iter,
        t_now,
        dt_now,
        T_prev,
        ux_prev,
        uy_prev,
        d_prev,
        prev_weights=None,
    ):
        thermal_weights = None
        loss_info_cache = {}
        disp_reg_ref = torch.tensor(
            float(self.runtime_scale.get("disp_reg_ref", 1.0)),
            device=self.device,
            dtype=self.dtype,
        ).clamp_min(1e-18)

        def thermal_loss_fn():
            nonlocal thermal_weights, loss_info_cache
            T_bc_target = self._current_T_bc_target(t_now)
            T_pred, ux_pred, uy_pred, _ = self.field_comp.fieldCalculation_tm(
                inp=self.inp,
                thermal_dirichlet_nodes=self.thermal_nodes,
                T_bc_value=T_bc_target,
                mechanical_top_nodes=self.top_nodes,
                mechanical_bottom_nodes=self.bottom_nodes,
                fixed_point_nodes=self.fixed_nodes,
            )
            _, _, terms = compute_thermal_loss(
                inp=self.inp,
                T=T_pred,
                T_prev=T_prev,
                d_prev=d_prev,
                area_elem=self.area_T,
                T_conn=self.T_conn,
                thermo_model=self.thermo_model,
                thermal_prop=self.thermal_prop,
                dt=dt_now,
                bc_dict=self.bc_dict,
                time_value=t_now,
            )

            disp_reg_raw = torch.mean((ux_pred - ux_prev) ** 2 + (uy_pred - uy_prev) ** 2)
            disp_reg_nd = disp_reg_raw / disp_reg_ref
            nd_terms = {
                "pde_nd": terms["pde_nd"],
                "bc_nd": terms["bc_nd"],
                "disp_reg_nd": disp_reg_nd,
            }

            if self.thermal_auto:
                if thermal_weights is None:
                    thermal_weights = _make_auto_weights(
                        term_map=nd_terms,
                        cfg=self.auto_weight_cfg,
                        prev_weights=prev_weights,
                    )
            else:
                thermal_weights = {
                    "pde_nd": 1.0,
                    "bc_nd": float(self.training_dict.get("w_bc_T", 1.0)),
                    "disp_reg_nd": float(self.training_dict.get("w_disp_reg", 1.0)),
                }

            loss_train_nd = _weighted_sum_nd(nd_terms, thermal_weights)
            l2_reg = self.l2_reg_coeff * _l2_param_sum(self.field_comp.net.parameters()).to(
                device=self.device, dtype=self.dtype
            )
            loss_obj = self._transform_loss(loss_train_nd + l2_reg)

            loss_info_cache = {
                "thermal_pde_raw": terms["pde_raw"],
                "thermal_bc_raw": terms["bc_raw"],
                "thermal_pde_nd": terms["pde_nd"],
                "thermal_bc_nd": terms["bc_nd"],
                "thermal_disp_reg_raw": disp_reg_raw,
                "thermal_disp_reg_nd": disp_reg_nd,
                "w_th_pde": float(thermal_weights.get("pde_nd", 1.0)),
                "w_th_bc": float(thermal_weights.get("bc_nd", 1.0)),
                "w_th_disp": float(thermal_weights.get("disp_reg_nd", 1.0)),
                "l2_reg": l2_reg,
                "w_l2_reg": self.l2_reg_coeff,
                "loss_train_nd": loss_train_nd.detach(),
            }
            return loss_obj, loss_info_cache

        loss_series = fit_tm(
            loss_fn=thermal_loss_fn,
            params=self.field_comp.net.parameters(),
            optimizer_dict=self.optimizer_dict,
            n_epochs_lbfgs=int(self.optimizer_dict.get("n_epochs_LBFGS_thermal", 0)),
            n_epochs_rprop=int(self.optimizer_dict.get("n_epochs_RPROP_thermal", 0)),
            min_delta=float(self.optimizer_dict.get("optim_rel_tol", 1e-6)),
            writer=self.writer,
            writer_tag=f"thermal/step_{step_idx}/iter_{inner_iter}",
        )

        with torch.no_grad():
            T_bc_target = self._current_T_bc_target(t_now)
            T_curr, ux_curr, uy_curr, _ = self.field_comp.fieldCalculation_tm(
                inp=self.inp,
                thermal_dirichlet_nodes=self.thermal_nodes,
                T_bc_value=T_bc_target,
                mechanical_top_nodes=self.top_nodes,
                mechanical_bottom_nodes=self.bottom_nodes,
                fixed_point_nodes=self.fixed_nodes,
            )

        return {
            "T": T_curr.detach(),
            "ux": ux_curr.detach(),
            "uy": uy_curr.detach(),
            "loss_series": [float(v) for v in loss_series],
            "terms": {k: (_to_float(v) if torch.is_tensor(v) else v) for k, v in loss_info_cache.items()},
            "weights": dict(thermal_weights) if thermal_weights is not None else prev_weights,
            "loss_raw": _to_float(loss_info_cache.get("thermal_pde_raw", 0.0))
            + _to_float(loss_info_cache.get("thermal_bc_raw", 0.0)),
        }

    def solve_mechanical_step(
        self,
        step_idx,
        inner_iter,
        t_now,
        T_lock_target,
        d_prev,
        prev_weights=None,
    ):
        mech_weights = None
        mech_info_cache = {}
        T_lock_ref = torch.tensor(
            float(self.runtime_scale.get("T_lock_ref", 1.0)),
            device=self.device,
            dtype=self.dtype,
        ).clamp_min(1e-18)
        bc_u_ref = torch.tensor(
            float(self.runtime_scale.get("bc_u_ref", 1.0)),
            device=self.device,
            dtype=self.dtype,
        ).clamp_min(1e-18)

        gauge_ux_weight = float(self.training_dict.get("w_gauge_ux", 0.0))

        def mech_loss_fn():
            nonlocal mech_weights, mech_info_cache
            T_bc_target = self._current_T_bc_target(t_now)
            T_pred, ux_pred, uy_pred, _ = self.field_comp.fieldCalculation_tm(
                inp=self.inp,
                thermal_dirichlet_nodes=self.thermal_nodes,
                T_bc_value=T_bc_target,
                mechanical_top_nodes=self.top_nodes,
                mechanical_bottom_nodes=self.bottom_nodes,
                fixed_point_nodes=self.fixed_nodes,
            )
            loss_mech_raw, mech_state_local = compute_mech_loss(
                inp=self.inp,
                ux=ux_pred,
                uy=uy_pred,
                T_phys=T_lock_target,
                d_prev=d_prev,
                area_elem=self.area_T,
                T_conn=self.T_conn,
                thermo_model=self.thermo_model,
                thermal_prop=self.thermal_prop,
                scale_dict=self.runtime_scale,
            )

            T_lock_raw = torch.mean((T_pred - T_lock_target) ** 2)
            T_lock_nd = T_lock_raw / T_lock_ref
            bc_u_raw = self._compute_bc_u(ux_pred, uy_pred, uy_top_target=self.field_comp.uy_rate * self.field_comp.time)
            bc_u_nd = bc_u_raw / bc_u_ref

            if self.use_ux_gauge and self.ux_gauge_nodes.numel() > 0:
                gauge_ux_raw = torch.mean(ux_pred[self.ux_gauge_nodes]) ** 2
            else:
                gauge_ux_raw = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            gauge_ux_nd = gauge_ux_raw / bc_u_ref

            nd_terms = {
                "mech_nd": mech_state_local["mech_nd"],
                "T_lock_nd": T_lock_nd,
                "bc_u_nd": bc_u_nd,
                "gauge_ux_nd": gauge_ux_nd,
            }

            if self.mech_auto:
                if mech_weights is None:
                    mech_weights = _make_auto_weights(
                        term_map=nd_terms,
                        cfg=self.auto_weight_cfg,
                        prev_weights=prev_weights,
                    )
            else:
                mech_weights = {
                    "mech_nd": 1.0,
                    "T_lock_nd": float(self.training_dict.get("w_T_lock", 1.0)),
                    "bc_u_nd": float(self.training_dict.get("w_bc_u", 1.0)),
                    "gauge_ux_nd": gauge_ux_weight,
                }

            loss_train_nd = _weighted_sum_nd(nd_terms, mech_weights)
            l2_reg = self.l2_reg_coeff * _l2_param_sum(self.field_comp.net.parameters()).to(
                device=self.device, dtype=self.dtype
            )
            loss_obj = self._transform_loss(loss_train_nd + l2_reg)

            mech_info_cache = {
                "mech_state": mech_state_local,
                "mech_raw": loss_mech_raw,
                "mech_nd": mech_state_local["mech_nd"],
                "T_lock_raw": T_lock_raw,
                "T_lock_nd": T_lock_nd,
                "bc_u_raw": bc_u_raw,
                "bc_u_nd": bc_u_nd,
                "gauge_ux_raw": gauge_ux_raw,
                "gauge_ux_nd": gauge_ux_nd,
                "w_mech": float(mech_weights.get("mech_nd", 1.0)),
                "w_Tlock": float(mech_weights.get("T_lock_nd", 1.0)),
                "w_bcu": float(mech_weights.get("bc_u_nd", 1.0)),
                "w_gauge_ux": float(mech_weights.get("gauge_ux_nd", 0.0)),
                "l2_reg": l2_reg,
                "w_l2_reg": self.l2_reg_coeff,
                "loss_train_nd": loss_train_nd.detach(),
            }
            return loss_obj, mech_info_cache

        loss_series = fit_tm(
            loss_fn=mech_loss_fn,
            params=self.field_comp.net.parameters(),
            optimizer_dict=self.optimizer_dict,
            n_epochs_lbfgs=int(self.optimizer_dict.get("n_epochs_LBFGS_mech", 0)),
            n_epochs_rprop=int(self.optimizer_dict.get("n_epochs_RPROP_mech", 0)),
            min_delta=float(self.optimizer_dict.get("optim_rel_tol", 1e-6)),
            writer=self.writer,
            writer_tag=f"mech/step_{step_idx}/iter_{inner_iter}",
        )

        with torch.no_grad():
            T_bc_target = self._current_T_bc_target(t_now)
            T_curr, ux_curr, uy_curr, _ = self.field_comp.fieldCalculation_tm(
                inp=self.inp,
                thermal_dirichlet_nodes=self.thermal_nodes,
                T_bc_value=T_bc_target,
                mechanical_top_nodes=self.top_nodes,
                mechanical_bottom_nodes=self.bottom_nodes,
                fixed_point_nodes=self.fixed_nodes,
            )
            _, mech_state_final = compute_mech_loss(
                inp=self.inp,
                ux=ux_curr,
                uy=uy_curr,
                T_phys=T_lock_target,
                d_prev=d_prev,
                area_elem=self.area_T,
                T_conn=self.T_conn,
                thermo_model=self.thermo_model,
                thermal_prop=self.thermal_prop,
                scale_dict=self.runtime_scale,
            )

        mech_terms = {k: (_to_float(v) if torch.is_tensor(v) else v) for k, v in mech_info_cache.items() if k != "mech_state"}
        return {
            "T": T_curr.detach(),
            "ux": ux_curr.detach(),
            "uy": uy_curr.detach(),
            "mech_state": mech_state_final,
            "loss_series": [float(v) for v in loss_series],
            "terms": mech_terms,
            "weights": dict(mech_weights) if mech_weights is not None else prev_weights,
            "loss_raw": _to_float(mech_terms.get("mech_raw", 0.0)),
        }

    def solve_phase_step(
        self,
        step_idx,
        inner_iter,
        d_prev,
        He_nodes_cand,
        He_elem_cand,
        prev_weights=None,
    ):
        phase_weights = prev_weights
        phase_terms_last = {}
        loss_series_all = []
        rounds = self.phase_solver_rounds if self.phase_solver_enabled else 1
        good_count = 0

        d_curr, _ = self.field_comp.fieldCalculation_phase(self.inp, d_prev=d_prev, return_raw=True)
        d_curr = d_curr.detach()

        for phase_round in range(1, rounds + 1):
            phase_weights_round = None
            d_before = d_curr.detach().clone()

            def phase_loss_fn():
                nonlocal phase_weights_round, phase_weights, phase_terms_last
                d_pred, _ = self.field_comp.fieldCalculation_phase(self.inp, d_prev=d_prev, return_raw=True)
                _, _, components = compute_phase_loss(
                    inp=self.inp,
                    d=d_pred,
                    d_prev=d_prev,
                    He=He_nodes_cand,
                    He_elem=He_elem_cand,
                    area_elem=self.area_T,
                    T_conn=self.T_conn,
                    thermo_model=self.thermo_model,
                    dt=torch.tensor(1.0, device=self.device, dtype=self.dtype),
                    irreversibility_weight=self.w_irrev,
                    return_components=True,
                    scale_dict=self.runtime_scale,
                )

                w_cd_m = float(self.phase_term_cfg.get("w_cd", 1.0))
                w_reac_m = float(self.phase_term_cfg.get("w_reac", 1.0))
                w_visc_m = float(self.phase_term_cfg.get("w_visc", 1.0))
                w_ir_m = float(self.phase_term_cfg.get("w_ir", 1.0))

                nd_terms = {
                    "cd_nd": w_cd_m * components["crack_density_nd"],
                    "reac_nd": w_reac_m * components["reaction_nd"],
                    "visc_nd": w_visc_m * components["viscosity_nd"],
                    "ir_nd": w_ir_m * components["ir_nd"],
                }
                mag_terms = {
                    "cd_nd": abs(w_cd_m) * components["crack_density_mag_nd"],
                    "reac_nd": abs(w_reac_m) * components["reaction_mag_nd"],
                    "visc_nd": abs(w_visc_m) * components["viscosity_mag_nd"],
                    "ir_nd": abs(w_ir_m) * components["ir_mag_nd"],
                }

                if self.phase_auto:
                    if phase_weights_round is None:
                        phase_weights_round = _make_auto_weights(
                            term_map=nd_terms,
                            cfg=self.auto_weight_cfg,
                            prev_weights=phase_weights,
                            magnitude_map=mag_terms,
                        )
                else:
                    phase_weights_round = {"cd_nd": 1.0, "reac_nd": 1.0, "visc_nd": 1.0, "ir_nd": 1.0}

                loss_phase_nd = _weighted_sum_nd(nd_terms, phase_weights_round)
                loss_phase_nd = self.phase_weight_global * loss_phase_nd
                l2_reg = self.l2_reg_coeff * _l2_param_sum(self.field_comp.net.parameters()).to(
                    device=self.device, dtype=self.dtype
                )
                loss_obj = self._transform_loss(loss_phase_nd + l2_reg)

                phase_terms_last = {
                    "crack_density_raw": components["crack_density_raw"],
                    "reaction_raw": components["reaction_raw"],
                    "viscosity_raw": components["viscosity_raw"],
                    "phase_domain_raw": components["phase_domain_raw"],
                    "ir_raw": components["ir_raw"],
                    "crack_density_nd": components["crack_density_nd"],
                    "reaction_nd": components["reaction_nd"],
                    "viscosity_nd": components["viscosity_nd"],
                    "phase_domain_nd": components["phase_domain_nd"],
                    "ir_nd": components["ir_nd"],
                    "w_cd": float(phase_weights_round.get("cd_nd", 1.0)),
                    "w_reac": float(phase_weights_round.get("reac_nd", 1.0)),
                    "w_visc": float(phase_weights_round.get("visc_nd", 1.0)),
                    "w_ir": float(phase_weights_round.get("ir_nd", 1.0)),
                    "w_phase_global": self.phase_weight_global,
                    "l2_reg": l2_reg,
                    "loss_train_nd": loss_phase_nd.detach(),
                }
                return loss_obj, phase_terms_last

            loss_round = fit_tm(
                loss_fn=phase_loss_fn,
                params=self.field_comp.net.parameters(),
                optimizer_dict=self.optimizer_dict,
                n_epochs_lbfgs=int(self.optimizer_dict.get("n_epochs_LBFGS_phase", 0)),
                n_epochs_rprop=int(self.optimizer_dict.get("n_epochs_RPROP_phase", 0)),
                min_delta=float(self.optimizer_dict.get("optim_rel_tol", 1e-6)),
                writer=self.writer,
                writer_tag=f"phase/step_{step_idx}/iter_{inner_iter}/round_{phase_round}",
            )
            loss_series_all.extend([float(v) for v in loss_round])

            if self.phase_auto and phase_weights_round is not None:
                phase_weights = dict(phase_weights_round)

            with torch.no_grad():
                d_after, _ = self.field_comp.fieldCalculation_phase(self.inp, d_prev=d_prev, return_raw=True)
            rel_d = _relative_change(d_after, d_before)
            d_curr = d_after.detach()
            if rel_d < self.phase_solver_tol_d:
                good_count += 1
            else:
                good_count = 0
            if good_count >= self.phase_solver_patience:
                break

        return {
            "d": d_curr,
            "loss_series": loss_series_all,
            "terms": {k: (_to_float(v) if torch.is_tensor(v) else v) for k, v in phase_terms_last.items()},
            "weights": phase_weights,
            "loss_raw": _to_float(phase_terms_last.get("phase_domain_raw", 0.0)),
            "ir_raw": _to_float(phase_terms_last.get("ir_raw", 0.0)),
        }

    def advance_step(
        self,
        step_idx: int,
        t_now,
        dt_now,
        accepted_state: AcceptedState,
        state_mgr: PathDependentStateManager,
        branch_weight_state: Optional[Dict[str, Optional[Dict[str, float]]]] = None,
    ):
        if branch_weight_state is None:
            branch_weight_state = {"thermal": None, "mechanical": None, "phase": None}

        self.field_comp.set_time(t_now)
        state_mgr.step_begin()

        T_iter_prev = accepted_state.T.detach().clone()
        ux_iter_prev = accepted_state.ux.detach().clone()
        uy_iter_prev = accepted_state.uy.detach().clone()
        d_iter_prev = accepted_state.d.detach().clone()
        HI_elem_prev, HII_elem_prev, _ = state_mgr.element_state(use_candidate=False)
        inner_prev_total = None
        conv_count = 0

        step_loss_series = {"thermal": [], "mech": [], "phase": []}
        thermal_terms_final = {}
        mech_terms_final = {}
        phase_terms_final = {}
        mech_state_final = None
        cand_state_final = None
        instant_state_final = None
        rel_loss = rel_T = rel_u = rel_d = rel_HI = rel_HII = 1.0

        for inner_iter in range(1, self.max_inner_iters + 1):
            self.field_comp.set_prev_damage(d_iter_prev)

            thermal_res = self.solve_thermal_step(
                step_idx=step_idx,
                inner_iter=inner_iter,
                t_now=t_now,
                dt_now=dt_now,
                T_prev=accepted_state.T,
                ux_prev=ux_iter_prev,
                uy_prev=uy_iter_prev,
                d_prev=d_iter_prev,
                prev_weights=branch_weight_state.get("thermal"),
            )
            branch_weight_state["thermal"] = thermal_res["weights"]
            step_loss_series["thermal"].extend(thermal_res["loss_series"])

            mech_res = self.solve_mechanical_step(
                step_idx=step_idx,
                inner_iter=inner_iter,
                t_now=t_now,
                T_lock_target=thermal_res["T"],
                d_prev=d_iter_prev,
                prev_weights=branch_weight_state.get("mechanical"),
            )
            branch_weight_state["mechanical"] = mech_res["weights"]
            step_loss_series["mech"].extend(mech_res["loss_series"])

            psi_I_elem = mech_res["mech_state"]["psi_I"].detach()
            psi_II_elem = mech_res["mech_state"]["psi_II"].detach()
            HI_elem_cand, HII_elem_cand, He_elem_cand = state_mgr.build_candidate(psi_I_elem, psi_II_elem)
            He_nodes_cand = element_to_nodal(He_elem_cand, self.T_conn, self.inp.shape[0], area_elem=self.area_T)

            phase_res = self.solve_phase_step(
                step_idx=step_idx,
                inner_iter=inner_iter,
                d_prev=d_iter_prev,
                He_nodes_cand=He_nodes_cand,
                He_elem_cand=He_elem_cand,
                prev_weights=branch_weight_state.get("phase"),
            )
            branch_weight_state["phase"] = phase_res["weights"]
            step_loss_series["phase"].extend(phase_res["loss_series"])

            T_curr = mech_res["T"].detach()
            ux_curr = mech_res["ux"].detach()
            uy_curr = mech_res["uy"].detach()
            d_curr = phase_res["d"].detach()

            thermal_terms_final = thermal_res["terms"]
            mech_terms_final = mech_res["terms"]
            phase_terms_final = phase_res["terms"]
            mech_state_final = mech_res["mech_state"]

            cand_state_final = CandidateState(
                HI_elem=HI_elem_cand.detach().clone(),
                HII_elem=HII_elem_cand.detach().clone(),
                He_elem=He_elem_cand.detach().clone(),
                d_cand=d_curr.detach().clone(),
            )
            instant_state_final = InstantaneousState(
                T=T_curr.detach().clone(),
                ux=ux_curr.detach().clone(),
                uy=uy_curr.detach().clone(),
                d_iter=d_curr.detach().clone(),
                psi_I_elem=psi_I_elem.detach().clone(),
                psi_II_elem=psi_II_elem.detach().clone(),
                mech_state=mech_state_final,
            )

            inner_total = (
                float(thermal_terms_final.get("thermal_pde_raw", 0.0))
                + float(thermal_terms_final.get("thermal_bc_raw", 0.0))
                + float(mech_terms_final.get("mech_raw", 0.0))
                + float(phase_terms_final.get("phase_domain_raw", 0.0))
                + self.w_irrev * float(phase_terms_final.get("ir_raw", 0.0))
            )
            if inner_prev_total is None:
                rel_loss = 1.0
            else:
                rel_loss = abs(inner_total - inner_prev_total) / (abs(inner_prev_total) + 1e-12)
            inner_prev_total = inner_total

            rel_T = _relative_change(T_curr, T_iter_prev)
            rel_u = _relative_change(torch.stack((ux_curr, uy_curr), dim=1), torch.stack((ux_iter_prev, uy_iter_prev), dim=1))
            rel_d = _relative_change(d_curr, d_iter_prev)
            rel_HI = _relative_change(HI_elem_cand, HI_elem_prev)
            rel_HII = _relative_change(HII_elem_cand, HII_elem_prev)

            converged_now = (
                (rel_loss < self.tol_loss)
                and (rel_T < self.tol_field_T)
                and (rel_u < self.tol_field_u)
                and (rel_d < self.tol_field_d)
                and (rel_HI < self.tol_hist)
                and (rel_HII < self.tol_hist)
            )
            if converged_now:
                conv_count += 1
            else:
                conv_count = 0

            T_iter_prev = T_curr
            ux_iter_prev = ux_curr
            uy_iter_prev = uy_curr
            d_iter_prev = d_curr
            HI_elem_prev = HI_elem_cand.detach().clone()
            HII_elem_prev = HII_elem_cand.detach().clone()
            if conv_count >= self.conv_patience:
                break

        state_mgr.accept_step(instant_state_final.psi_I_elem, instant_state_final.psi_II_elem)
        HI_elem_acc, HII_elem_acc, He_elem_acc = state_mgr.element_state(use_candidate=False)

        accepted_state_new = AcceptedState(
            step=int(step_idx),
            time=float(_to_float(t_now)),
            T=instant_state_final.T.detach().clone(),
            ux=instant_state_final.ux.detach().clone(),
            uy=instant_state_final.uy.detach().clone(),
            d=instant_state_final.d_iter.detach().clone(),
            HI_elem=HI_elem_acc.detach().clone(),
            HII_elem=HII_elem_acc.detach().clone(),
            He_elem=He_elem_acc.detach().clone(),
        )

        total_loss = (
            float(thermal_terms_final.get("thermal_pde_raw", 0.0))
            + float(thermal_terms_final.get("thermal_bc_raw", 0.0))
            + float(mech_terms_final.get("mech_raw", 0.0))
            + float(phase_terms_final.get("phase_domain_raw", 0.0))
            + self.w_irrev * float(phase_terms_final.get("ir_raw", 0.0))
        )

        loss_row = {
            "step": int(step_idx),
            "time": float(_to_float(t_now)),
            "inner_iters": int(inner_iter),
            "converged": int(conv_count >= self.conv_patience),
            "thermal_loss": float(thermal_terms_final.get("thermal_pde_raw", 0.0) + thermal_terms_final.get("thermal_bc_raw", 0.0)),
            "mech_loss": float(mech_terms_final.get("mech_raw", 0.0)),
            "phase_loss": float(phase_terms_final.get("phase_domain_raw", 0.0) + self.w_irrev * phase_terms_final.get("ir_raw", 0.0)),
            "irreversibility_loss": float(phase_terms_final.get("ir_raw", 0.0)),
            "boundary_loss": float(mech_terms_final.get("bc_u_raw", 0.0)),
            "E_el": float(mech_terms_final.get("mech_raw", 0.0)),
            "E_phase_domain": float(phase_terms_final.get("phase_domain_raw", 0.0)),
            "E_crack_density": float(phase_terms_final.get("crack_density_raw", 0.0)),
            "E_reaction": float(phase_terms_final.get("reaction_raw", 0.0)),
            "E_viscosity": float(phase_terms_final.get("viscosity_raw", 0.0)),
            "total_loss": float(total_loss),
            "loss_total": float(total_loss),
            "loss_T": float(thermal_terms_final.get("thermal_pde_raw", 0.0) + thermal_terms_final.get("thermal_bc_raw", 0.0)),
            "loss_u": float(mech_terms_final.get("mech_raw", 0.0)),
            "loss_d": float(phase_terms_final.get("phase_domain_raw", 0.0)),
            "loss_irrev": float(self.w_irrev * phase_terms_final.get("ir_raw", 0.0)),
            "loss_bc": float(mech_terms_final.get("bc_u_raw", 0.0)),
            "max_d": float(torch.max(accepted_state_new.d).item()),
            "max_HI": float(torch.max(HI_elem_acc).item()),
            "max_HII": float(torch.max(HII_elem_acc).item()),
            "rel_loss": float(rel_loss),
            "rel_T": float(rel_T),
            "rel_u": float(rel_u),
            "rel_d": float(rel_d),
            "rel_HI": float(rel_HI),
            "rel_HII": float(rel_HII),
            "thermal_pde_raw": float(thermal_terms_final.get("thermal_pde_raw", 0.0)),
            "thermal_bc_raw": float(thermal_terms_final.get("thermal_bc_raw", 0.0)),
            "thermal_pde_nd": float(thermal_terms_final.get("thermal_pde_nd", 0.0)),
            "thermal_bc_nd": float(thermal_terms_final.get("thermal_bc_nd", 0.0)),
            "thermal_disp_reg_nd": float(thermal_terms_final.get("thermal_disp_reg_nd", 0.0)),
            "w_th_pde": float(thermal_terms_final.get("w_th_pde", 1.0)),
            "w_th_bc": float(thermal_terms_final.get("w_th_bc", 1.0)),
            "w_th_disp": float(thermal_terms_final.get("w_th_disp", 1.0)),
            "l2_reg_raw": float(thermal_terms_final.get("l2_reg", 0.0)),
            "w_l2_reg": float(thermal_terms_final.get("w_l2_reg", self.l2_reg_coeff)),
            "mech_raw": float(mech_terms_final.get("mech_raw", 0.0)),
            "mech_nd": float(mech_terms_final.get("mech_nd", 0.0)),
            "T_lock_raw": float(mech_terms_final.get("T_lock_raw", 0.0)),
            "T_lock_nd": float(mech_terms_final.get("T_lock_nd", 0.0)),
            "bc_u_raw": float(mech_terms_final.get("bc_u_raw", 0.0)),
            "bc_u_nd": float(mech_terms_final.get("bc_u_nd", 0.0)),
            "gauge_ux_raw": float(mech_terms_final.get("gauge_ux_raw", 0.0)),
            "gauge_ux_nd": float(mech_terms_final.get("gauge_ux_nd", 0.0)),
            "w_mech": float(mech_terms_final.get("w_mech", 1.0)),
            "w_Tlock": float(mech_terms_final.get("w_Tlock", 1.0)),
            "w_bcu": float(mech_terms_final.get("w_bcu", 1.0)),
            "w_gauge_ux": float(mech_terms_final.get("w_gauge_ux", 0.0)),
            "crack_density_raw": float(phase_terms_final.get("crack_density_raw", 0.0)),
            "reaction_raw": float(phase_terms_final.get("reaction_raw", 0.0)),
            "viscosity_raw": float(phase_terms_final.get("viscosity_raw", 0.0)),
            "phase_domain_raw": float(phase_terms_final.get("phase_domain_raw", 0.0)),
            "crack_density_nd": float(phase_terms_final.get("crack_density_nd", 0.0)),
            "reaction_nd": float(phase_terms_final.get("reaction_nd", 0.0)),
            "viscosity_nd": float(phase_terms_final.get("viscosity_nd", 0.0)),
            "phase_domain_nd": float(phase_terms_final.get("phase_domain_nd", 0.0)),
            "ir_nd": float(phase_terms_final.get("ir_nd", 0.0)),
            "w_cd": float(phase_terms_final.get("w_cd", 1.0)),
            "w_reac": float(phase_terms_final.get("w_reac", 1.0)),
            "w_visc": float(phase_terms_final.get("w_visc", 1.0)),
            "w_ir": float(phase_terms_final.get("w_ir", 1.0)),
            "phase_weight": float(self.phase_weight_global),
        }

        convergence = {
            "inner_iters": int(inner_iter),
            "converged": int(conv_count >= self.conv_patience),
            "rel_loss": float(rel_loss),
            "rel_T": float(rel_T),
            "rel_u": float(rel_u),
            "rel_d": float(rel_d),
            "rel_HI": float(rel_HI),
            "rel_HII": float(rel_HII),
        }

        return StepAdvanceResult(
            accepted_state=accepted_state_new,
            instant_state=instant_state_final,
            candidate_state=cand_state_final,
            step_loss_series=step_loss_series,
            loss_row=loss_row,
            convergence=convergence,
            branch_weight_state=branch_weight_state,
        )
