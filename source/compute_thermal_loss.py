import torch

from compute_energy import field_grads
from thermo_mech_model import nodal_to_element


def get_thermal_dirichlet_target(bc_dict, thermal_prop, time_value, idx_dirichlet, device, dtype):
    """
    Resolve thermal Dirichlet targets from bc_dict with optional time dependence.
    Supported keys in bc_dict:
    - thermal_dirichlet_fn: callable(t) -> scalar or per-node tensor/array
    - thermal_dirichlet_values: array-like per-node values
    - thermal_dirichlet_value: scalar
    Fallback: thermal_prop.T0
    """
    if idx_dirichlet is None or idx_dirichlet.numel() == 0:
        return torch.tensor([], device=device, dtype=dtype)

    if "thermal_dirichlet_fn" in bc_dict and callable(bc_dict["thermal_dirichlet_fn"]):
        target = bc_dict["thermal_dirichlet_fn"](float(time_value))
    elif "thermal_dirichlet_values" in bc_dict:
        target = bc_dict["thermal_dirichlet_values"]
    elif "thermal_dirichlet_value" in bc_dict:
        target = bc_dict["thermal_dirichlet_value"]
    else:
        target = thermal_prop.T0

    if not torch.is_tensor(target):
        target = torch.tensor(target, device=device, dtype=dtype)
    else:
        target = target.to(device=device, dtype=dtype)

    if target.numel() == 1:
        return target
    if target.numel() == idx_dirichlet.numel():
        return target.reshape(-1)
    raise ValueError(
        "Thermal Dirichlet target must be scalar or match thermal_dirichlet_nodes length."
    )


def compute_thermal_loss(inp, T, T_prev, d_prev, area_elem, T_conn, thermo_model, thermal_prop, dt, bc_dict, time_value):
    # Note for the default report setup:
    # initial T = T0 and Dirichlet boundaries also at T0, so the thermal field evolves weakly.
    # This is a physical consequence of the prescribed BCs, not a coupling bug.
    grad_Tx, grad_Ty = field_grads(inp, T, area_elem, T_conn)
    T_elem = nodal_to_element(T, T_conn)
    T_prev_elem = nodal_to_element(T_prev, T_conn)
    d_elem = nodal_to_element(d_prev, T_conn)

    k_d = thermo_model.thermal_conductivity(d_elem, thermal_prop.k0)
    temporal = 0.5 * thermal_prop.rho * thermal_prop.c / dt * (T_elem - T_prev_elem) ** 2
    conductive = 0.5 * k_d * (grad_Tx ** 2 + grad_Ty ** 2)

    loss_domain = torch.sum((temporal + conductive) * area_elem)

    idx_dirichlet = bc_dict["thermal_dirichlet_nodes"]
    if idx_dirichlet.numel() > 0:
        T_target = get_thermal_dirichlet_target(
            bc_dict=bc_dict,
            thermal_prop=thermal_prop,
            time_value=time_value,
            idx_dirichlet=idx_dirichlet,
            device=inp.device,
            dtype=T.dtype,
        )
        loss_bc = torch.mean((T[idx_dirichlet] - T_target) ** 2)
    else:
        loss_bc = torch.tensor(0.0, device=inp.device)

    scale_dict = bc_dict.get("scale_dict", None)
    if scale_dict is None:
        thermal_loss_ref = torch.tensor(1.0, device=inp.device, dtype=T.dtype)
        T_lock_ref = torch.tensor(1.0, device=inp.device, dtype=T.dtype)
    else:
        thermal_loss_ref = torch.tensor(
            float(scale_dict.get("thermal_loss_ref", 1.0)),
            device=inp.device,
            dtype=T.dtype,
        )
        T_lock_ref = torch.tensor(
            float(scale_dict.get("T_lock_ref", 1.0)),
            device=inp.device,
            dtype=T.dtype,
        )
    thermal_loss_ref = torch.clamp(thermal_loss_ref, min=torch.tensor(1e-18, device=inp.device, dtype=T.dtype))
    T_lock_ref = torch.clamp(T_lock_ref, min=torch.tensor(1e-18, device=inp.device, dtype=T.dtype))

    terms = {
        "pde_raw": loss_domain,
        "bc_raw": loss_bc,
        "pde_nd": loss_domain / thermal_loss_ref,
        "bc_nd": loss_bc / T_lock_ref,
        # backward-compatible aliases
        "domain": loss_domain,
        "bc": loss_bc,
    }
    return loss_domain, loss_bc, terms
