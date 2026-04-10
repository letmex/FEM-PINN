import csv
import time
from pathlib import Path

import numpy as np
import torch

from compute_thermal_loss import get_thermal_dirichlet_target
from input_data_from_mesh import prep_input_data_tm
from loss_logger import LossLogger
from solver_orchestrator import SolverOrchestrator
from state_manager import AcceptedState, PathDependentStateManager
from thermo_mech_model import element_to_nodal, nodal_to_element


def _extract_boundary_edges(T_conn, boundary_nodes):
    nodes = set(boundary_nodes.detach().cpu().numpy().tolist())
    conn = T_conn.detach().cpu().numpy()
    edge_to_elem = {}
    for elem_idx, tri in enumerate(conn):
        tri_edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for a, b in tri_edges:
            if int(a) in nodes and int(b) in nodes:
                edge = tuple(sorted((int(a), int(b))))
                edge_to_elem.setdefault(edge, []).append(elem_idx)

    boundary_edges, boundary_elems = [], []
    for edge, elems in edge_to_elem.items():
        if len(elems) == 1:
            boundary_edges.append(edge)
            boundary_elems.append(elems[0])
    return boundary_edges, boundary_elems


def _map_edges_to_boundary_elements(T_conn, edges):
    conn = T_conn.detach().cpu().numpy()
    edge_to_elem = {}
    for elem_idx, tri in enumerate(conn):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = tuple(sorted((int(a), int(b))))
            edge_to_elem.setdefault(edge, []).append(elem_idx)

    boundary_edges, boundary_elems = [], []
    for e in edges:
        edge = tuple(sorted((int(e[0]), int(e[1]))))
        adj = edge_to_elem.get(edge, [])
        if len(adj) == 1:
            boundary_edges.append(edge)
            boundary_elems.append(adj[0])
    return boundary_edges, boundary_elems


def _reaction_force(sig_yy_elem, boundary_edges, boundary_elems, inp, thk):
    if len(boundary_edges) == 0:
        return torch.tensor(0.0, device=inp.device)
    force = torch.tensor(0.0, device=inp.device)
    for edge, elem_id in zip(boundary_edges, boundary_elems):
        i, j = edge
        seg_len = torch.norm(inp[i, -2:] - inp[j, -2:])
        force = force + sig_yy_elem[elem_id] * seg_len * thk
    return force


def _save_field_csv(file_name, inp, fields):
    base_keys = ["T", "ux", "uy", "d", "HI", "HII", "He"]
    optional_keys = [
        "psi_I",
        "psi_II",
        "ep2",
        "ep2_legacy",
        "ep2_norm",
        "ep2_shear",
        "psi_II_norm_part",
        "psi_II_shear_part",
    ]

    cols = [inp[:, 0], inp[:, 1]]
    header_keys = ["x", "y"]

    for k in base_keys:
        cols.append(fields[k])
        header_keys.append(k)
    for k in optional_keys:
        if k in fields:
            cols.append(fields[k])
            header_keys.append(k)

    data = np.column_stack(cols)
    header = ",".join(header_keys)
    np.savetxt(file_name, data, delimiter=",", header=header, comments="")


def _write_dict_rows(file_name, fieldnames, rows):
    with open(file_name, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_loss_trace(loss_trace_file, branch, step_idx, loss_list, iter_counter, inner_iter=0):
    if len(loss_list) == 0:
        return iter_counter
    with open(loss_trace_file, "a", newline="") as file:
        writer = csv.writer(file)
        for loss_val in loss_list:
            iter_counter += 1
            writer.writerow([iter_counter, step_idx, inner_iter, branch, float(loss_val)])
    return iter_counter


def _read_dict_rows(file_name):
    rows = []
    if not Path(file_name).exists():
        return rows
    with open(file_name, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row_out = {}
            for key, value in row.items():
                if value is None or value == "":
                    row_out[key] = value
                    continue
                try:
                    row_out[key] = float(value)
                except ValueError:
                    row_out[key] = value
            rows.append(row_out)
    return rows


def _last_trace_iter(loss_trace_file):
    if not Path(loss_trace_file).exists():
        return 0
    with open(loss_trace_file, "r", newline="") as file:
        lines = file.read().strip().splitlines()
    if len(lines) <= 1:
        return 0
    try:
        return int(lines[-1].split(",")[0])
    except (ValueError, IndexError):
        return 0


def _load_field_state(field_csv, device):
    data = np.genfromtxt(field_csv, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)

    def _to_tensor(name):
        return torch.tensor(data[name], dtype=torch.float32, device=device)

    return {
        "T": _to_tensor("T"),
        "ux": _to_tensor("ux"),
        "uy": _to_tensor("uy"),
        "d": _to_tensor("d"),
        "HI": _to_tensor("HI"),
        "HII": _to_tensor("HII"),
    }


def _find_latest_step(field_path):
    latest_step = -1
    for field_file in field_path.glob("field_step_*.csv"):
        try:
            step = int(field_file.stem.split("_")[-1])
        except ValueError:
            continue
        latest_step = max(latest_step, step)
    return latest_step


def _infer_last_completed_step(field_path, loss_per_step_file, trainedModel_path):
    rows = _read_dict_rows(loss_per_step_file)
    steps_in_loss = set()
    for row in rows:
        try:
            steps_in_loss.add(int(float(row.get("step", -1))))
        except Exception:
            continue

    latest = -1
    for field_file in field_path.glob("field_step_*.csv"):
        try:
            step = int(field_file.stem.split("_")[-1])
        except Exception:
            continue
        if step <= 0:
            latest = max(latest, step)
            continue
        if step not in steps_in_loss:
            continue
        ckpt = trainedModel_path / Path(f"trained_unified_{step:04d}.pt")
        if not ckpt.exists():
            continue
        if field_file.stat().st_size <= 0:
            continue
        latest = max(latest, step)
    return latest


def _build_runtime_scale_dict(scale_cfg, inp, area_T, thermo_model, thermal_prop, time_dict, field_comp):
    x = inp[:, -2]
    y = inp[:, -1]
    Lx = float(torch.max(x) - torch.min(x))
    Ly = float(torch.max(y) - torch.min(y))
    L_ref = max(float(scale_cfg.get("L_ref", max(Lx, Ly))), 1e-12)
    uy_rate = float(field_comp.uy_rate.detach().cpu().item()) if torch.is_tensor(field_comp.uy_rate) else float(field_comp.uy_rate)
    U_ref_default = max(uy_rate * float(time_dict["t_end"]), 1e-12)
    U_ref = max(float(scale_cfg.get("U_ref", U_ref_default)), 1e-12)
    T_ref = float(scale_cfg.get("T_ref", float(thermal_prop.Tref.detach().cpu().item())))
    DT_ref = max(
        float(scale_cfg.get("DT_ref", abs(float(thermal_prop.T0.detach().cpu().item()) - T_ref))),
        1e-12,
    )

    eps_ref = float(scale_cfg.get("eps_ref", U_ref / L_ref))
    E0 = float(thermo_model.E0.detach().cpu().item())
    sigma_ref = float(scale_cfg.get("sigma_ref", E0 * eps_ref))
    psi_ref = float(scale_cfg.get("psi_ref", E0 * eps_ref ** 2))
    domain_area = max(float(torch.sum(area_T).detach().cpu().item()), 1e-18)
    dt_ref = max(float(time_dict["dt"]), 1e-12)
    rho = float(thermal_prop.rho.detach().cpu().item())
    c = float(thermal_prop.c.detach().cpu().item())
    k0 = float(thermal_prop.k0.detach().cpu().item())
    thermal_density_ref = rho * c * (DT_ref ** 2) / dt_ref + k0 * (DT_ref / L_ref) ** 2

    return {
        "L_ref": L_ref,
        "U_ref": U_ref,
        "T_ref": T_ref,
        "DT_ref": DT_ref,
        "eps_ref": eps_ref,
        "sigma_ref": sigma_ref,
        "psi_ref": psi_ref,
        "H_ref": float(scale_cfg.get("H_ref", psi_ref)),
        "area_ref": float(scale_cfg.get("area_ref", L_ref ** 2)),
        "thermal_loss_ref": max(float(scale_cfg.get("thermal_loss_ref", thermal_density_ref * domain_area)), 1e-18),
        "mech_loss_ref": max(float(scale_cfg.get("mech_loss_ref", psi_ref * domain_area)), 1e-18),
        "phase_loss_ref": max(float(scale_cfg.get("phase_loss_ref", psi_ref * domain_area)), 1e-18),
        "T_lock_ref": max(float(scale_cfg.get("T_lock_ref", DT_ref ** 2)), 1e-18),
        "bc_u_ref": max(float(scale_cfg.get("bc_u_ref", U_ref ** 2)), 1e-18),
        "disp_reg_ref": max(float(scale_cfg.get("disp_reg_ref", U_ref ** 2)), 1e-18),
        "irrev_ref": max(float(scale_cfg.get("irrev_ref", 1.0)), 1e-18),
    }


def _infer_notch_tip_region_masks(inp, bc_dict):
    x = inp[:, -2]
    y = inp[:, -1]
    bottom_nodes = bc_dict.get("mechanical_bottom_nodes", bc_dict.get("bottom_nodes", None))
    notch_nodes = bc_dict.get("notch_face_nodes", None)
    fixed_nodes = bc_dict.get("fixed_point_nodes", bc_dict.get("point1_node", None))

    if bottom_nodes is None or bottom_nodes.numel() == 0 or notch_nodes is None or notch_nodes.numel() == 0:
        return None, None

    notch_x = x[notch_nodes]
    notch_y = y[notch_nodes]
    x_tip = torch.max(notch_x)
    y_tip = torch.median(notch_y)

    x_bottom = x[bottom_nodes]
    if fixed_nodes is not None and fixed_nodes.numel() > 0:
        x_fix = torch.mean(x[fixed_nodes])
    else:
        x_fix = torch.mean(x_bottom)

    lx = torch.clamp(torch.max(x) - torch.min(x), min=torch.tensor(1e-12, device=inp.device, dtype=inp.dtype))
    ly = torch.clamp(torch.max(y) - torch.min(y), min=torch.tensor(1e-12, device=inp.device, dtype=inp.dtype))

    notch_rx = 0.03 * lx
    notch_ry = 0.02 * ly
    bottom_rx = 0.03 * lx
    bottom_ry = 0.02 * ly

    notch_mask = (torch.abs(x - x_tip) <= notch_rx) & (torch.abs(y - y_tip) <= notch_ry)
    bottom_mask = (torch.abs(x - x_fix) <= bottom_rx) & (torch.abs(y - torch.min(y)) <= bottom_ry)

    return notch_mask, bottom_mask


def _safe_region_mean(val, mask):
    if mask is None:
        return torch.tensor(0.0, device=val.device, dtype=val.dtype)
    if torch.any(mask):
        return torch.mean(val[mask])
    return torch.tensor(0.0, device=val.device, dtype=val.dtype)


def _parse_step_index(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _is_valid_field_csv(field_csv):
    field_csv = Path(field_csv)
    if (not field_csv.exists()) or (field_csv.stat().st_size <= 0):
        return False
    try:
        data = np.genfromtxt(field_csv, delimiter=",", names=True, max_rows=1)
    except Exception:
        return False
    if data is None:
        return False
    names = tuple(data.dtype.names or ())
    required = ("x", "y", "T", "ux", "uy", "d", "HI", "HII", "He")
    return all(name in names for name in required)


def _scan_completed_steps(loss_per_step_file, field_path, trainedModel_path, require_checkpoint=True):
    loss_steps = set()
    for row in _read_dict_rows(loss_per_step_file):
        step = _parse_step_index(row.get("step"))
        if step is None or step < 1:
            continue
        loss_steps.add(step)

    completed_steps = set()
    partial_steps = set()
    for step in sorted(loss_steps):
        field_csv = Path(field_path) / Path(f"field_step_{step:04d}.csv")
        ckpt_file = Path(trainedModel_path) / Path(f"trained_unified_{step:04d}.pt")
        ok_field = _is_valid_field_csv(field_csv)
        ok_ckpt = (not require_checkpoint) or (ckpt_file.exists() and ckpt_file.stat().st_size > 0)
        if ok_field and ok_ckpt:
            completed_steps.add(step)
        else:
            partial_steps.add(step)

    last_completed_step = 0
    while (last_completed_step + 1) in completed_steps:
        last_completed_step += 1

    return {
        "last_completed_step": int(last_completed_step),
        "completed_steps": sorted(completed_steps),
        "partial_steps": sorted(partial_steps),
        "loss_steps": sorted(loss_steps),
    }


def train_tm(
    field_comp,
    thermo_model,
    thermal_prop,
    crack_dict,
    numr_dict,
    optimizer_dict,
    training_dict,
    time_dict,
    mesh_file,
    device,
    trainedModel_path,
    results_path,
    writer,
    boundary_tag_dict=None,
    return_run_meta=False,
):
    """
    Stateful TM-phase training driver.
    Main path:
      accepted_state -> (thermal, mechanical, history, phase) -> candidate -> accept/reject
    """

    results_path.mkdir(parents=True, exist_ok=True)
    field_path = results_path / Path("field_data")
    field_path.mkdir(parents=True, exist_ok=True)
    curve_path = results_path / Path("curves")
    curve_path.mkdir(parents=True, exist_ok=True)
    loss_path = results_path / Path("losses")
    loss_path.mkdir(parents=True, exist_ok=True)
    loss_logger = LossLogger(loss_path)

    inp, T_conn, area_T, bc_dict, d0 = prep_input_data_tm(
        tm_model=thermo_model,
        crack_dict=crack_dict,
        mesh_file=mesh_file,
        device=device,
        length_scale=numr_dict.get("length_scale", 1.0),
        boundary_tag_dict=boundary_tag_dict,
    )

    if (
        bc_dict.get("thermal_dirichlet_value", None) is None
        and "thermal_dirichlet_fn" not in bc_dict
        and "thermal_dirichlet_values" not in bc_dict
    ):
        bc_dict["thermal_dirichlet_value"] = thermal_prop.T0.detach().clone()

    field_comp.net = field_comp.net.to(device)
    inp = inp.to(device)
    T_conn = T_conn.to(device)
    area_T = area_T.to(device)

    runtime_scale = _build_runtime_scale_dict(
        scale_cfg=training_dict.get("scale_dict", {}),
        inp=inp,
        area_T=area_T,
        thermo_model=thermo_model,
        thermal_prop=thermal_prop,
        time_dict=time_dict,
        field_comp=field_comp,
    )
    bc_dict["scale_dict"] = runtime_scale
    field_comp.T_shift = torch.tensor(runtime_scale["T_ref"], device=device, dtype=inp.dtype)
    field_comp.T_scale = torch.tensor(runtime_scale["DT_ref"], device=device, dtype=inp.dtype)
    field_comp.U_scale = torch.tensor(runtime_scale["U_ref"], device=device, dtype=inp.dtype)

    mech_top_nodes = bc_dict.get("mechanical_top_nodes", bc_dict.get("top_nodes"))
    mech_bottom_nodes = bc_dict.get("mechanical_bottom_nodes", bc_dict.get("bottom_nodes"))
    fixed_point_nodes = bc_dict.get("fixed_point_nodes", bc_dict.get("point1_node"))
    use_fixed_point_anchor = bool(training_dict.get("use_fixed_point_anchor", True))
    if not use_fixed_point_anchor:
        fixed_point_nodes = torch.empty((0,), dtype=torch.long, device=device)

    print(
        "Boundary sets:",
        f"source={bc_dict.get('boundary_source', 'unknown')},",
        f"physical_available={bc_dict.get('physical_groups_available', False)},",
        f"top={mech_top_nodes.numel()}, bottom={mech_bottom_nodes.numel()}, notch={bc_dict['notch_face_nodes'].numel()}, fixed_point={fixed_point_nodes.numel()}",
    )

    n_pts = inp.shape[0]
    t_start = time_dict["t_start"]
    t_end = time_dict["t_end"]
    dt = time_dict["dt"]
    time_arr = np.arange(t_start, t_end + 0.5 * dt, dt)
    target_final_step = len(time_arr) - 1

    top_edges_labeled = bc_dict.get("mechanical_top_edges", None)
    if top_edges_labeled is not None and top_edges_labeled.numel() > 0:
        top_edges_np = top_edges_labeled.detach().cpu().numpy().astype(np.int64)
        top_edges, top_elems = _map_edges_to_boundary_elements(T_conn, top_edges_np)
    else:
        top_edges, top_elems = _extract_boundary_edges(T_conn, mech_top_nodes)

    curve_file = curve_path / Path("reaction_displacement_macro_stress_strain.csv")
    loss_per_step_file = loss_path / Path("loss_per_step.csv")
    loss_trace_file = loss_path / Path("loss_trace.csv")
    diagnostics_file = loss_path / Path("diagnostics_window_step1_10_20_64.csv")
    physics_consistency_file = loss_path / Path("physics_consistency_per_step.csv")

    T_prev = torch.full((n_pts,), float(thermal_prop.T0.item()), device=device)
    ux_prev = torch.zeros((n_pts,), device=device)
    uy_prev = torch.zeros((n_pts,), device=device)
    d_prev = d0.detach()
    HI_elem_prev = torch.zeros_like(area_T)
    HII_elem_prev = torch.zeros_like(area_T)

    def _history_nodes_from_elem(HI_elem, HII_elem):
        HI_nodes = element_to_nodal(HI_elem, T_conn, n_pts, area_elem=area_T)
        HII_nodes = element_to_nodal(HII_elem, T_conn, n_pts, area_elem=area_T)
        He_nodes = HI_nodes + thermo_model.gc_ratio * HII_nodes
        return HI_nodes, HII_nodes, He_nodes

    HI_prev_nodes, HII_prev_nodes, He_prev_nodes = _history_nodes_from_elem(HI_elem_prev, HII_elem_prev)

    curve_rows = []
    loss_per_step_rows = _read_dict_rows(loss_per_step_file)
    diagnostics_rows = _read_dict_rows(diagnostics_file)
    physics_rows = _read_dict_rows(physics_consistency_file)
    trace_iter = _last_trace_iter(loss_trace_file)
    start_step = 1
    train_executed = False
    require_resume_ckpt = bool(training_dict.get("resume_require_checkpoint", True))

    progress_before = _scan_completed_steps(
        loss_per_step_file=loss_per_step_file,
        field_path=field_path,
        trainedModel_path=trainedModel_path,
        require_checkpoint=require_resume_ckpt,
    )
    steps_before = int(progress_before["last_completed_step"])
    if len(progress_before.get("partial_steps", [])) > 0:
        print(
            "Detected partial/stale trailing steps:",
            progress_before.get("partial_steps", []),
            f"(resume baseline step={steps_before})",
        )

    if training_dict.get("resume_if_available", False):
        latest_step = int(progress_before["last_completed_step"])
        if latest_step >= 1:
            ckpt = trainedModel_path / Path(f"trained_unified_{latest_step:04d}.pt")
            field_csv = field_path / Path(f"field_step_{latest_step:04d}.csv")
            if ckpt.exists() and field_csv.exists():
                field_state = _load_field_state(field_csv, device=device)
                T_prev = field_state["T"].detach()
                ux_prev = field_state["ux"].detach()
                uy_prev = field_state["uy"].detach()
                d_prev = field_state["d"].detach()

                ckpt_state = torch.load(ckpt, map_location=device, weights_only=False)
                if isinstance(ckpt_state, dict) and "net_state_dict" in ckpt_state:
                    field_comp.net.load_state_dict(ckpt_state["net_state_dict"])
                    if ("HI_elem_prev" in ckpt_state) and ("HII_elem_prev" in ckpt_state):
                        HI_elem_prev = ckpt_state["HI_elem_prev"].to(device=device, dtype=area_T.dtype).detach()
                        HII_elem_prev = ckpt_state["HII_elem_prev"].to(device=device, dtype=area_T.dtype).detach()
                    else:
                        HI_elem_prev = nodal_to_element(field_state["HI"], T_conn).detach()
                        HII_elem_prev = nodal_to_element(field_state["HII"], T_conn).detach()
                else:
                    field_comp.net.load_state_dict(ckpt_state)
                    HI_elem_prev = nodal_to_element(field_state["HI"], T_conn).detach()
                    HII_elem_prev = nodal_to_element(field_state["HII"], T_conn).detach()

                HI_prev_nodes, HII_prev_nodes, He_prev_nodes = _history_nodes_from_elem(HI_elem_prev, HII_elem_prev)
                curve_rows = _read_dict_rows(curve_file)
                start_step = latest_step + 1
                print(f"Resuming stateful training from step {latest_step}.")

    if start_step == 1:
        fields_np = {
            "T": T_prev.detach().cpu().numpy(),
            "ux": ux_prev.detach().cpu().numpy(),
            "uy": uy_prev.detach().cpu().numpy(),
            "d": d_prev.detach().cpu().numpy(),
            "HI": HI_prev_nodes.detach().cpu().numpy(),
            "HII": HII_prev_nodes.detach().cpu().numpy(),
            "He": He_prev_nodes.detach().cpu().numpy(),
        }
        _save_field_csv(field_path / Path("field_step_0000.csv"), inp.detach().cpu().numpy(), fields_np)
        torch.save(
            {
                "net_state_dict": field_comp.net.state_dict(),
                "HI_elem_prev": HI_elem_prev.detach().cpu(),
                "HII_elem_prev": HII_elem_prev.detach().cpu(),
                "step": 0,
            },
            trainedModel_path / Path("trained_unified_0000.pt"),
        )
        curve_rows = [
            {
                "step": 0,
                "time": float(time_arr[0]),
                "uy_top": 0.0,
                "reaction_force": 0.0,
                "macro_strain": 0.0,
                "macro_stress": 0.0,
            }
        ]
        _write_dict_rows(
            file_name=curve_file,
            fieldnames=["step", "time", "uy_top", "reaction_force", "macro_strain", "macro_stress"],
            rows=curve_rows,
        )
        loss_logger.init_iteration_trace(reset=True)
        trace_iter = 0
    else:
        loss_logger.init_iteration_trace(reset=False)

    if start_step >= len(time_arr):
        print("All time steps are already available. Skipping training.")
        run_meta = {
            "train_executed": False,
            "steps_before": steps_before,
            "steps_after": steps_before,
            "new_steps_generated": 0,
            "eligible_for_escalation": False,
            "target_final_step": target_final_step,
            "partial_steps_before": progress_before.get("partial_steps", []),
            "partial_steps_after": progress_before.get("partial_steps", []),
        }
        if return_run_meta:
            return inp, T_conn, area_T, bc_dict, run_meta
        return inp, T_conn, area_T, bc_dict

    notch_mask, bottom_mask = _infer_notch_tip_region_masks(inp, bc_dict)
    diagnostic_steps = set(int(s) for s in training_dict.get("diagnostic_steps", [1, 10, 20, 64]))

    state_mgr = PathDependentStateManager(
        thermo_model=thermo_model,
        T_conn=T_conn,
        area_elem=area_T,
        n_nodes=n_pts,
        history_update_mode=str(training_dict.get("history_update_mode", "step_end")),
    )
    state_mgr.load_from_tensors(HI_elem_prev, HII_elem_prev)
    accepted_state = AcceptedState(
        step=max(0, start_step - 1),
        time=float(time_arr[max(0, start_step - 1)]),
        T=T_prev.detach().clone(),
        ux=ux_prev.detach().clone(),
        uy=uy_prev.detach().clone(),
        d=d_prev.detach().clone(),
        HI_elem=HI_elem_prev.detach().clone(),
        HII_elem=HII_elem_prev.detach().clone(),
        He_elem=(HI_elem_prev + thermo_model.gc_ratio * HII_elem_prev).detach().clone(),
    )
    branch_weight_state = {"thermal": None, "mechanical": None, "phase": None}
    orchestrator = SolverOrchestrator(
        field_comp=field_comp,
        thermo_model=thermo_model,
        thermal_prop=thermal_prop,
        inp=inp,
        T_conn=T_conn,
        area_T=area_T,
        bc_dict=bc_dict,
        optimizer_dict=optimizer_dict,
        training_dict=training_dict,
        runtime_scale=runtime_scale,
        writer=writer,
    )

    width = torch.clamp(
        torch.max(inp[:, -2]) - torch.min(inp[:, -2]),
        min=torch.tensor(1e-12, device=device, dtype=inp.dtype),
    )
    height = torch.clamp(
        torch.max(inp[:, -1]) - torch.min(inp[:, -1]),
        min=torch.tensor(1e-12, device=device, dtype=inp.dtype),
    )

    for step_idx in range(start_step, len(time_arr)):
        train_executed = True
        t_now = torch.tensor(time_arr[step_idx], device=device, dtype=inp.dtype)
        dt_now = torch.tensor(time_arr[step_idx] - time_arr[step_idx - 1], device=device, dtype=inp.dtype)

        res = orchestrator.advance_step(
            step_idx=step_idx,
            t_now=t_now,
            dt_now=dt_now,
            accepted_state=accepted_state,
            state_mgr=state_mgr,
            branch_weight_state=branch_weight_state,
        )
        accepted_state = res.accepted_state
        branch_weight_state = res.branch_weight_state

        instant = res.instant_state
        mech_state = instant.mech_state
        HI_nodes, HII_nodes, He_nodes = state_mgr.nodal_state(use_candidate=False)
        psi_I_nodes = element_to_nodal(instant.psi_I_elem, T_conn, n_pts, area_elem=area_T)
        psi_II_nodes = element_to_nodal(instant.psi_II_elem, T_conn, n_pts, area_elem=area_T)

        optional_fields = {}
        if "ep2" in mech_state:
            optional_fields["ep2"] = element_to_nodal(mech_state["ep2"], T_conn, n_pts, area_elem=area_T).detach().cpu().numpy()
        if "ep2_legacy" in mech_state:
            optional_fields["ep2_legacy"] = element_to_nodal(mech_state["ep2_legacy"], T_conn, n_pts, area_elem=area_T).detach().cpu().numpy()
        if "ep2_norm" in mech_state:
            optional_fields["ep2_norm"] = element_to_nodal(mech_state["ep2_norm"], T_conn, n_pts, area_elem=area_T).detach().cpu().numpy()
        if "ep2_shear" in mech_state:
            optional_fields["ep2_shear"] = element_to_nodal(mech_state["ep2_shear"], T_conn, n_pts, area_elem=area_T).detach().cpu().numpy()
        if "psi_II_norm_part" in mech_state:
            optional_fields["psi_II_norm_part"] = element_to_nodal(
                mech_state["psi_II_norm_part"], T_conn, n_pts, area_elem=area_T
            ).detach().cpu().numpy()
        if "psi_II_shear_part" in mech_state:
            optional_fields["psi_II_shear_part"] = element_to_nodal(
                mech_state["psi_II_shear_part"], T_conn, n_pts, area_elem=area_T
            ).detach().cpu().numpy()

        fields_np = {
            "T": accepted_state.T.detach().cpu().numpy(),
            "ux": accepted_state.ux.detach().cpu().numpy(),
            "uy": accepted_state.uy.detach().cpu().numpy(),
            "d": accepted_state.d.detach().cpu().numpy(),
            "HI": HI_nodes.detach().cpu().numpy(),
            "HII": HII_nodes.detach().cpu().numpy(),
            "He": He_nodes.detach().cpu().numpy(),
            "psi_I": psi_I_nodes.detach().cpu().numpy(),
            "psi_II": psi_II_nodes.detach().cpu().numpy(),
            **optional_fields,
        }
        _save_field_csv(
            field_path / Path(f"field_step_{step_idx:04d}.csv"),
            inp.detach().cpu().numpy(),
            fields_np,
        )

        sig_yy_elem = mech_state["stress"]["sig_yy"]
        reaction_force = _reaction_force(sig_yy_elem, top_edges, top_elems, inp, thermal_prop.thk)
        uy_top = torch.mean(accepted_state.uy[mech_top_nodes]) if mech_top_nodes.numel() > 0 else torch.tensor(0.0, device=device)
        macro_strain = uy_top / height
        macro_stress = reaction_force / (thermal_prop.thk * width)
        curve_rows.append(
            {
                "step": int(step_idx),
                "time": float(time_arr[step_idx]),
                "uy_top": float(uy_top.item()),
                "reaction_force": float(reaction_force.item()),
                "macro_strain": float(macro_strain.item()),
                "macro_stress": float(macro_stress.item()),
            }
        )
        _write_dict_rows(
            file_name=curve_file,
            fieldnames=["step", "time", "uy_top", "reaction_force", "macro_strain", "macro_stress"],
            rows=curve_rows,
        )

        loss_row = dict(res.loss_row)
        loss_row.update(
            {
                "reaction_top": float(reaction_force.item()),
                "macro_stress": float(macro_stress.item()),
                "macro_strain": float(macro_strain.item()),
                "max_HI": float(torch.max(HI_nodes).item()),
                "max_HII": float(torch.max(HII_nodes).item()),
                "max_He": float(torch.max(He_nodes).item()),
            }
        )
        loss_per_step_rows.append(loss_row)
        loss_logger.write_step_rows(loss_per_step_rows)
        loss_logger.save_step_loss_artifacts(step_idx=step_idx, branch_losses=res.step_loss_series)

        trace_iter = _append_loss_trace(
            loss_trace_file=loss_trace_file,
            branch="thermal",
            step_idx=step_idx,
            loss_list=res.step_loss_series.get("thermal", []),
            iter_counter=trace_iter,
        )
        trace_iter = _append_loss_trace(
            loss_trace_file=loss_trace_file,
            branch="mech",
            step_idx=step_idx,
            loss_list=res.step_loss_series.get("mech", []),
            iter_counter=trace_iter,
        )
        trace_iter = _append_loss_trace(
            loss_trace_file=loss_trace_file,
            branch="phase",
            step_idx=step_idx,
            loss_list=res.step_loss_series.get("phase", []),
            iter_counter=trace_iter,
        )

        torch.save(
            {
                "net_state_dict": field_comp.net.state_dict(),
                "HI_elem_prev": accepted_state.HI_elem.detach().cpu(),
                "HII_elem_prev": accepted_state.HII_elem.detach().cpu(),
                "step": int(step_idx),
            },
            trainedModel_path / Path(f"trained_unified_{step_idx:04d}.pt"),
        )

        T_bc_target = get_thermal_dirichlet_target(
            bc_dict=bc_dict,
            thermal_prop=thermal_prop,
            time_value=t_now,
            idx_dirichlet=bc_dict["thermal_dirichlet_nodes"],
            device=inp.device,
            dtype=inp.dtype,
        )
        T_bc_nodes = bc_dict["thermal_dirichlet_nodes"]
        T_bc_err = torch.tensor(0.0, device=inp.device, dtype=inp.dtype)
        if T_bc_nodes is not None and T_bc_nodes.numel() > 0:
            if T_bc_target.numel() == 1:
                T_bc_err = torch.mean(torch.abs(accepted_state.T[T_bc_nodes] - T_bc_target.reshape(1)))
            else:
                T_bc_err = torch.mean(torch.abs(accepted_state.T[T_bc_nodes] - T_bc_target.reshape(-1)))
        uy_top_target = field_comp.uy_rate * field_comp.time
        uy_top_err = torch.mean(torch.abs(accepted_state.uy[mech_top_nodes] - uy_top_target))
        uy_bottom_err = torch.mean(torch.abs(accepted_state.uy[mech_bottom_nodes]))
        g_elem = thermo_model.degradation(nodal_to_element(accepted_state.d, T_conn)).detach()
        sigyy_abs = torch.abs(mech_state["stress"]["sig_yy"]).detach()
        stress_degrade_corr = torch.tensor(0.0, device=inp.device, dtype=inp.dtype)
        if g_elem.numel() > 1 and sigyy_abs.numel() > 1:
            g0 = g_elem - torch.mean(g_elem)
            s0 = sigyy_abs - torch.mean(sigyy_abs)
            denom = torch.sqrt(torch.sum(g0 ** 2) * torch.sum(s0 ** 2) + 1e-16)
            stress_degrade_corr = torch.sum(g0 * s0) / denom

        d_drop_max = torch.max(torch.clamp(d_prev - accepted_state.d, min=0.0))
        HI_drop_max = torch.max(torch.clamp(HI_prev_nodes - HI_nodes, min=0.0))
        HII_drop_max = torch.max(torch.clamp(HII_prev_nodes - HII_nodes, min=0.0))
        physics_rows.append(
            {
                "step": int(step_idx),
                "time": float(time_arr[step_idx]),
                "T_bc_l1": float(T_bc_err.item()),
                "uy_top_l1": float(uy_top_err.item()),
                "uy_bottom_l1": float(uy_bottom_err.item()),
                "stress_degrade_corr": float(stress_degrade_corr.item()),
                "max_d_drop": float(d_drop_max.item()),
                "max_HI_drop": float(HI_drop_max.item()),
                "max_HII_drop": float(HII_drop_max.item()),
                "macro_stress": float(macro_stress.item()),
                "macro_strain": float(macro_strain.item()),
                "softening_started": int(step_idx > 1),
            }
        )
        _write_dict_rows(
            file_name=physics_consistency_file,
            fieldnames=list(physics_rows[-1].keys()),
            rows=physics_rows,
        )

        if step_idx in diagnostic_steps:
            exy_elem = mech_state["exy_e"].detach()
            exy_nodes = element_to_nodal(exy_elem, T_conn, n_pts, area_elem=area_T)
            d_nodes = accepted_state.d.detach()
            exy_notch = _safe_region_mean(torch.abs(exy_nodes), notch_mask).item()
            exy_bottom = _safe_region_mean(torch.abs(exy_nodes), bottom_mask).item()
            psi_notch = _safe_region_mean(psi_II_nodes, notch_mask).item()
            psi_bottom = _safe_region_mean(psi_II_nodes, bottom_mask).item()
            hii_notch = _safe_region_mean(HII_nodes, notch_mask).item()
            hii_bottom = _safe_region_mean(HII_nodes, bottom_mask).item()
            he_notch = _safe_region_mean(He_nodes, notch_mask).item()
            he_bottom = _safe_region_mean(He_nodes, bottom_mask).item()
            d_notch = _safe_region_mean(d_nodes, notch_mask).item()
            d_bottom = _safe_region_mean(d_nodes, bottom_mask).item()
            diagnostics_rows.append(
                {
                    "step": int(step_idx),
                    "time": float(time_arr[step_idx]),
                    "exy_notch": exy_notch,
                    "exy_bottom": exy_bottom,
                    "R_exy": exy_bottom / (exy_notch + 1e-16),
                    "psiII_notch": psi_notch,
                    "psiII_bottom": psi_bottom,
                    "R_psiII": psi_bottom / (psi_notch + 1e-16),
                    "HII_notch": hii_notch,
                    "HII_bottom": hii_bottom,
                    "R_HII": hii_bottom / (hii_notch + 1e-16),
                    "He_notch": he_notch,
                    "He_bottom": he_bottom,
                    "R_He": he_bottom / (he_notch + 1e-16),
                    "d_notch": d_notch,
                    "d_bottom": d_bottom,
                    "R_d": d_bottom / (d_notch + 1e-16),
                    "phase_weight": float(loss_row.get("phase_weight", 1.0)),
                }
            )
            _write_dict_rows(
                file_name=diagnostics_file,
                fieldnames=list(diagnostics_rows[-1].keys()),
                rows=diagnostics_rows,
            )

        T_prev = accepted_state.T.detach().clone()
        ux_prev = accepted_state.ux.detach().clone()
        uy_prev = accepted_state.uy.detach().clone()
        d_prev = accepted_state.d.detach().clone()
        HI_prev_nodes = HI_nodes.detach().clone()
        HII_prev_nodes = HII_nodes.detach().clone()
        He_prev_nodes = He_nodes.detach().clone()

    progress_after = _scan_completed_steps(
        loss_per_step_file=loss_per_step_file,
        field_path=field_path,
        trainedModel_path=trainedModel_path,
        require_checkpoint=require_resume_ckpt,
    )
    steps_after = int(progress_after["last_completed_step"])
    new_steps_generated = max(0, steps_after - steps_before)
    run_meta = {
        "train_executed": bool(train_executed),
        "steps_before": steps_before,
        "steps_after": steps_after,
        "new_steps_generated": int(new_steps_generated),
        "eligible_for_escalation": bool(train_executed and (new_steps_generated > 0)),
        "target_final_step": target_final_step,
        "partial_steps_before": progress_before.get("partial_steps", []),
        "partial_steps_after": progress_after.get("partial_steps", []),
    }
    if return_run_meta:
        return inp, T_conn, area_T, bc_dict, run_meta
    return inp, T_conn, area_T, bc_dict
