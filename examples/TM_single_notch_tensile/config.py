import numpy as np
import torch
from pathlib import Path
import sys
from torch.utils.tensorboard import SummaryWriter


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


network_dict = {
    "model_type": "MLP",
    "hidden_layers": int(sys.argv[1]) if len(sys.argv) > 1 else 6,
    "neurons": int(sys.argv[2]) if len(sys.argv) > 2 else 120,
    "seed": int(sys.argv[3]) if len(sys.argv) > 3 else 1,
    "activation": str(sys.argv[4]) if len(sys.argv) > 4 else "AdaptiveReLU",
    "init_coeff": float(sys.argv[5]) if len(sys.argv) > 5 else 1.0,
    # Unified-network phase-head bias default: moderate negative value to avoid
    # both d-locking (-8) and globally high-d uplift (0).
    "phase_output_bias": float(sys.argv[6]) if len(sys.argv) > 6 else -4.0,
    "use_input_normalization": True,
}


# Mixed-mode thermo-mechanical phase-field defaults (from the provided report)
tm_model_dict = {
    "E0": 8.15e10,
    "v0": 0.38,
    "Gf0": 2.4,
    "GcI": 2.4,
    "GcII": 2.3846,
    "kappa": 1e-5,
    "l0": 5e-8,
    "etaPF": 1.0,
    "eps_r": 1e-5,
}

thermal_prop_dict = {
    "alpha": 1.89e-5,
    "rho": 1040.0,
    "k0": 418.0,
    "c": 170.0,
    "T0": 293.15,
    "TFinal": 273.15,
    "Tref": 273.15,
    "thk": 1e-3,
}

numr_dict = {
    "gradient_type": "numerical",
    # External FE mesh is already in SI units.
    "length_scale": 1.0,
    "hsize": 1e-5,
    "length": 1e-5,
    "height": 1e-5,
}

optimizer_dict = {
    "weight_decay": 0.0,
    "optim_rel_tol": 1e-6,
    "lr_lbfgs": 0.5,
    "max_iter_lbfgs": 100,
    "history_size_lbfgs": 100,
    "lr_rprop": 1e-5,
    "rprop_step_lo": 1e-10,
    "rprop_step_hi": 1e-2,
    "n_epochs_LBFGS_thermal": 0,
    "n_epochs_RPROP_thermal": 30,
    "n_epochs_LBFGS_mech": 0,
    "n_epochs_RPROP_mech": 60,
    "n_epochs_LBFGS_phase": 0,
    "n_epochs_RPROP_phase": 80,
}

training_dict = {
    "w_bc_T": 1e6,
    "w_disp_reg": 1e2,
    "w_T_lock": 1e6,
    "w_bc_u": 1e8,
    "w_irrev": 1.0,
    "w_phase_global": 1.0,
    "phase_balance_mode": "none",  # kept for legacy diagnostics; branch-internal auto-weight is primary
    # objective transform to improve early-step optimizer stability
    # choices: raw | log1p | log10 | signed_log1p | signed_log10
    "loss_transform_mode": "signed_log1p",
    "phase_balance_target_ratio": 0.5,  # target phase/mech scale per inner step
    "phase_balance_min": 1.0,
    "phase_balance_max": 5e3,
    "auto_rebalance_enabled": False,
    "auto_rebalance_fail_streak": 2,
    "auto_rebalance_growth": 1.6,
    "auto_rebalance_max_target_ratio": 5.0,
    "auto_rebalance_good_streak": 3,
    "auto_rebalance_decay": 0.85,
    "auto_rebalance_min_target_ratio": 0.1,
    # Hysteresis thresholds for rebalance decisions.
    "rebalance_fail_hi": 1.05,
    "rebalance_success_lo": 0.95,
    "adaptive_enabled": True,
    "adaptive_phase_loss_threshold": 1e-4,
    "adaptive_damage_increment_threshold": 5e-4,
    "adaptive_max_passes": 2,
    "adaptive_lbfgs_mult": 1,
    "adaptive_rprop_mult": 1,
    "resume_if_available": True,
    # Resume only from steps with complete loss/field/checkpoint triplets.
    "resume_require_checkpoint": True,
    "max_inner_iters": 12,
    "conv_patience": 1,
    "tol_loss": 8e-1,
    "tol_field_T": 5e-3,
    "tol_field_u": 2e-3,
    "tol_field_d": 5e-4,
    "tol_hist": 2e-1,
    "require_converged_step": False,
    # Bottom support mode:
    # - "uxuy": clamp both ux, uy on mechanical_bottom_nodes
    # - "uy_only": roller-like support (uy=0 on bottom), with point1 anchoring ux
    "bottom_fix_mode": "uy_only",
    # Pure geometry-aware hard mapping; keep False for physical consistency.
    "enforce_nodal_clamp": False,
    # Strongly recommended for thermo-mech-phase: freeze history at step end.
    "history_update_mode": "step_end",
    # Pre-training BC/load check.
    "precheck_before_training": True,
    # If True, scripts generate precheck figures and exit before training.
    "stop_after_precheck": False,
    # Automatic physical-correctness loop controller.
    "auto_physics_loop": True,
    "auto_physics_max_rounds": 5,
    "auto_phase_ratio_growth": 1.6,
    "auto_phase_ratio_cap": 8.0,
    "auto_phase_rprop_growth": 1.35,
    "auto_phase_rprop_cap": 360,
    "auto_irrev_growth": 1.3,
    "auto_irrev_cap": 8.0,
}

time_dict = {"t_start": 0.0, "t_end": 5.0, "dt": 0.01}

# ---------------------------------------------------------------------------
# Commit-1 training scales (raw-recording + nondimensionalized optimization)
# ---------------------------------------------------------------------------
_L_ref_default = max(float(numr_dict["length"]), float(numr_dict["height"]))
_uy_rate_default = 4e-8
_U_ref_default = max(_uy_rate_default * float(time_dict["t_end"]), 1e-12)
_T_ref_default = float(thermal_prop_dict["Tref"])
_DT_ref_default = max(abs(float(thermal_prop_dict["T0"]) - _T_ref_default), 1.0)

scale_dict = {
    "L_ref": _L_ref_default,
    "U_ref": _U_ref_default,
    "T_ref": _T_ref_default,
    "DT_ref": _DT_ref_default,
    "d_ref": 1.0,
}

_eps_ref = scale_dict["U_ref"] / max(scale_dict["L_ref"], 1e-18)
_sigma_ref = tm_model_dict["E0"] * _eps_ref
_psi_ref = tm_model_dict["E0"] * (_eps_ref ** 2)
_H_ref = _psi_ref
_area_ref = scale_dict["L_ref"] ** 2
_domain_area_ref = numr_dict["length"] * numr_dict["height"]
_dt_ref = max(float(time_dict["dt"]), 1e-12)
_thermal_density_ref = (
    thermal_prop_dict["rho"] * thermal_prop_dict["c"] * (scale_dict["DT_ref"] ** 2) / _dt_ref
    + thermal_prop_dict["k0"] * (scale_dict["DT_ref"] / max(scale_dict["L_ref"], 1e-18)) ** 2
)

derived_scale_dict = {
    "eps_ref": _eps_ref,
    "sigma_ref": _sigma_ref,
    "psi_ref": _psi_ref,
    "H_ref": _H_ref,
    "area_ref": _area_ref,
    "thermal_loss_ref": _thermal_density_ref * _domain_area_ref,
    "mech_loss_ref": _psi_ref * _domain_area_ref,
    "phase_loss_ref": max(tm_model_dict["GcI"] / max(tm_model_dict["l0"], 1e-18), _psi_ref) * _domain_area_ref,
    "T_lock_ref": scale_dict["DT_ref"] ** 2,
    "bc_u_ref": scale_dict["U_ref"] ** 2,
    "disp_reg_ref": scale_dict["U_ref"] ** 2,
    "irrev_ref": 1.0,
}

training_dict["scale_dict"] = {**scale_dict, **derived_scale_dict}
training_dict["auto_weight_dict"] = {
    "enabled": True,
    "ema_beta": 0.9,
    "eps": 1e-12,
    "clip_min": 0.2,
    "clip_max": 5.0,
    "normalize_mean_to_one": True,
}
training_dict["branch_auto_weight_dict"] = {
    "thermal": True,
    "mechanical": True,
    "phase": True,
}

# Domain in SI (m): 1e-5 x 1e-5 plate.
domain_extrema = torch.tensor([[0.0, 1.0e-5], [0.0, 1.0e-5]])
# crack_dict is kept only for compatibility with legacy interfaces.
# Real pre-crack geometry is represented by mesh boundary, not by d-seed initialization.
crack_dict = {
    "x_init": [0.0],
    "y_init": [0.0],
    "L_crack": [0.0],
    "angle_crack": [0.0],
}

# uy_top(t) = 4e-8 * t (m)
uy_rate = torch.tensor(4e-8)

# Preferred Physical Group tags (if present in .msh).
# Fallback to geometric inference is automatic when tags are absent.
boundary_tag_dict = {
    "mechanical_top": [10],
    "mechanical_bottom": [2],
    "fixed_point": [1],
    "thermal_dirichlet": [12, 13, 14],
    "notch_faces": [],
    "thermal_insulated": [],
}

PATH_ROOT = Path(__file__).parents[0]
mesh_file = str(PATH_ROOT.parents[2] / Path("geo1_.msh"))

model_path = PATH_ROOT / Path(
    "hl_"
    + str(network_dict["hidden_layers"])
    + "_Neurons_"
    + str(network_dict["neurons"])
    + "_activation_"
    + network_dict["activation"]
    + "_coeff_"
    + str(network_dict["init_coeff"])
    + "_Seed_"
    + str(network_dict["seed"])
    + "_TM_mixed_mode_single_notch"
)
model_path.mkdir(parents=True, exist_ok=True)
trainedModel_path = model_path / Path("best_models")
trainedModel_path.mkdir(parents=True, exist_ok=True)
results_path = model_path / Path("results")
results_path.mkdir(parents=True, exist_ok=True)

with open(model_path / Path("model_settings.txt"), "w") as file:
    file.write(f'hidden_layers: {network_dict["hidden_layers"]}')
    file.write(f'\nneurons: {network_dict["neurons"]}')
    file.write(f'\nseed: {network_dict["seed"]}')
    file.write(f'\nactivation: {network_dict["activation"]}')
    file.write(f'\ncoeff: {network_dict["init_coeff"]}')
    file.write(f'\nphase_output_bias: {network_dict["phase_output_bias"]}')
    file.write(f'\nE0: {tm_model_dict["E0"]}')
    file.write(f'\nv0: {tm_model_dict["v0"]}')
    file.write(f'\nGcI: {tm_model_dict["GcI"]}')
    file.write(f'\nGcII: {tm_model_dict["GcII"]}')
    file.write(f'\nkappa: {tm_model_dict["kappa"]}')
    file.write(f'\nl0: {tm_model_dict["l0"]}')
    file.write(f'\netaPF: {tm_model_dict["etaPF"]}')
    file.write(f'\nalpha: {thermal_prop_dict["alpha"]}')
    file.write(f'\nrho: {thermal_prop_dict["rho"]}')
    file.write(f'\nk0: {thermal_prop_dict["k0"]}')
    file.write(f'\nc: {thermal_prop_dict["c"]}')
    file.write(f'\nT0: {thermal_prop_dict["T0"]}')
    file.write(f'\nTref: {thermal_prop_dict["Tref"]}')
    file.write(f'\nthk: {thermal_prop_dict["thk"]}')
    file.write(f"\nscale_dict: {training_dict['scale_dict']}")
    file.write(f"\nauto_weight_dict: {training_dict['auto_weight_dict']}")
    file.write(f"\nbranch_auto_weight_dict: {training_dict['branch_auto_weight_dict']}")
    file.write(f'\ntime: range({time_dict["t_start"]}, {time_dict["dt"]}, {time_dict["t_end"]})')
    file.write(f"\nmesh_file: {mesh_file}")
    file.write(f"\ndevice: {device}")

writer = SummaryWriter(model_path / Path("TBruns"))
