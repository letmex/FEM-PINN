import torch
import torch.nn as nn

from compute_energy import field_grads
from thermo_mech_model import nodal_to_element


def compute_phase_loss(
    inp,
    d,
    d_prev,
    He,
    area_elem,
    T_conn,
    thermo_model,
    dt,
    irreversibility_weight=1.0,
    He_elem=None,
    return_components=False,
    scale_dict=None,
):
    """
    Incremental phase-field loss.
    He must be the current-step candidate driving force:
    He = max(HI_n, psi_I) + (GcI/GcII) * max(HII_n, psi_II)
    """
    grad_dx, grad_dy = field_grads(inp, d, area_elem, T_conn)
    d_elem = nodal_to_element(d, T_conn)
    d_prev_elem = nodal_to_element(d_prev, T_conn)
    if He_elem is None:
        He_elem = nodal_to_element(He, T_conn)

    # Split phase-domain functional into explicit components without changing total loss:
    # 1) positive crack-surface part (gradient + local d^2 term)
    # 2) He-driven reaction part
    crack_density = (
        0.5 * thermo_model.GcI * thermo_model.l0 * (grad_dx ** 2 + grad_dy ** 2)
        + 0.5 * (thermo_model.GcI / thermo_model.l0) * d_elem ** 2
    )
    reaction = He_elem * (d_elem ** 2 - 2.0 * d_elem)
    viscosity = 0.5 * thermo_model.etaPF / dt * (d_elem - d_prev_elem) ** 2

    E_crack_density = torch.sum(crack_density * area_elem)
    E_reaction = torch.sum(reaction * area_elem)
    E_viscosity = torch.sum(viscosity * area_elem)
    loss_domain = E_crack_density + E_reaction + E_viscosity

    if irreversibility_weight > 0:
        irreversibility = nn.ReLU()(d_prev_elem - d_elem)
        loss_irrev = torch.mean(irreversibility ** 2)
    else:
        loss_irrev = torch.tensor(0.0, device=inp.device)

    if scale_dict is None:
        phase_loss_ref = torch.tensor(1.0, device=inp.device, dtype=loss_domain.dtype)
        irrev_ref = torch.tensor(1.0, device=inp.device, dtype=loss_domain.dtype)
    else:
        phase_loss_ref = torch.tensor(
            float(scale_dict.get("phase_loss_ref", 1.0)),
            device=inp.device,
            dtype=loss_domain.dtype,
        )
        irrev_ref = torch.tensor(
            float(scale_dict.get("irrev_ref", 1.0)),
            device=inp.device,
            dtype=loss_domain.dtype,
        )
    phase_loss_ref = torch.clamp(phase_loss_ref, min=torch.tensor(1e-18, device=inp.device, dtype=loss_domain.dtype))
    irrev_ref = torch.clamp(irrev_ref, min=torch.tensor(1e-18, device=inp.device, dtype=loss_domain.dtype))

    crack_density_nd = E_crack_density / phase_loss_ref
    reaction_nd = E_reaction / phase_loss_ref
    viscosity_nd = E_viscosity / phase_loss_ref
    phase_domain_nd = loss_domain / phase_loss_ref
    ir_nd = loss_irrev / irrev_ref

    if return_components:
        components = {
            "E_crack_density": E_crack_density,
            "E_reaction": E_reaction,
            "E_viscosity": E_viscosity,
            "E_phase_domain": loss_domain,
            "crack_density_raw": E_crack_density,
            "reaction_raw": E_reaction,
            "viscosity_raw": E_viscosity,
            "phase_domain_raw": loss_domain,
            "ir_raw": loss_irrev,
            "crack_density_nd": crack_density_nd,
            "reaction_nd": reaction_nd,
            "viscosity_nd": viscosity_nd,
            "phase_domain_nd": phase_domain_nd,
            "ir_nd": ir_nd,
            "crack_density_mag_nd": torch.abs(crack_density_nd.detach()),
            "reaction_mag_nd": torch.abs(reaction_nd.detach()),
            "viscosity_mag_nd": torch.abs(viscosity_nd.detach()),
            "ir_mag_nd": torch.abs(ir_nd.detach()),
        }
        return loss_domain, loss_irrev, components

    return loss_domain, loss_irrev
