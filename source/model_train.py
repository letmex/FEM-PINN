import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from input_data_from_mesh import prep_input_data, prep_input_data_tm
from fit import fit, fit_with_early_stopping, fit_tm
from compute_thermal_loss import compute_thermal_loss, get_thermal_dirichlet_target
from compute_mech_loss import compute_mech_loss
from compute_phase_loss import compute_phase_loss
from thermo_mech_model import element_to_nodal, nodal_to_element
from loss_logger import LossLogger
from optim import get_optimizer


def train(
    field_comp,
    disp,
    pffmodel,
    matprop,
    crack_dict,
    numr_dict,
    optimizer_dict,
    training_dict,
    coarse_mesh_file,
    fine_mesh_file,
    device,
    trainedModel_path,
    intermediateModel_path,
    writer,
):
    """
    Legacy training pipeline for the original non-TM example.
    Kept for repository backward compatibility.
    """

    inp, T_conn, area_T, hist_alpha = prep_input_data(
        matprop,
        pffmodel,
        crack_dict,
        numr_dict,
        mesh_file=coarse_mesh_file,
        device=device,
    )
    outp = torch.zeros(inp.shape[0], 1).to(device)
    training_set = DataLoader(torch.utils.data.TensorDataset(inp, outp), batch_size=inp.shape[0], shuffle=False)
    field_comp.lmbda = torch.tensor(disp[0]).to(device)

    loss_data = []
    start = time.time()

    n_epochs = max(optimizer_dict["n_epochs_LBFGS"], 1)
    optimizer = get_optimizer(field_comp.net.parameters(), "LBFGS")
    loss_data += fit(
        field_comp,
        training_set,
        T_conn,
        area_T,
        hist_alpha,
        matprop,
        pffmodel,
        optimizer_dict["weight_decay"],
        num_epochs=n_epochs,
        optimizer=optimizer,
        intermediateModel_path=None,
        writer=writer,
        training_dict=training_dict,
    )

    n_epochs = optimizer_dict["n_epochs_RPROP"]
    optimizer = get_optimizer(field_comp.net.parameters(), "RPROP")
    loss_data += fit_with_early_stopping(
        field_comp,
        training_set,
        T_conn,
        area_T,
        hist_alpha,
        matprop,
        pffmodel,
        optimizer_dict["weight_decay"],
        num_epochs=n_epochs,
        optimizer=optimizer,
        min_delta=optimizer_dict["optim_rel_tol_pretrain"],
        intermediateModel_path=None,
        writer=writer,
        training_dict=training_dict,
    )

    end = time.time()
    print(f"Execution time: {(end-start)/60:.03f}minutes")

    torch.save(field_comp.net.state_dict(), trainedModel_path / Path("trained_1NN_initTraining.pt"))
    with open(trainedModel_path / Path("trainLoss_1NN_initTraining.npy"), "wb") as file:
        np.save(file, np.asarray(loss_data))

    inp, T_conn, area_T, hist_alpha = prep_input_data(
        matprop,
        pffmodel,
        crack_dict,
        numr_dict,
        mesh_file=fine_mesh_file,
        device=device,
    )
    outp = torch.zeros(inp.shape[0], 1).to(device)
    training_set = DataLoader(torch.utils.data.TensorDataset(inp, outp), batch_size=inp.shape[0], shuffle=False)

    for j, disp_i in enumerate(disp):
        field_comp.lmbda = torch.tensor(disp_i).to(device)
        print(f"idx: {j}; displacement: {field_comp.lmbda}")
        loss_data = []
        start = time.time()

        if j == 0 or optimizer_dict["n_epochs_LBFGS"] > 0:
            n_epochs = max(optimizer_dict["n_epochs_LBFGS"], 1)
            optimizer = get_optimizer(field_comp.net.parameters(), "LBFGS")
            loss_data += fit(
                field_comp,
                training_set,
                T_conn,
                area_T,
                hist_alpha,
                matprop,
                pffmodel,
                optimizer_dict["weight_decay"],
                num_epochs=n_epochs,
                optimizer=optimizer,
                intermediateModel_path=None,
                writer=writer,
                training_dict=training_dict,
            )

        if optimizer_dict["n_epochs_RPROP"] > 0:
            n_epochs = optimizer_dict["n_epochs_RPROP"]
            optimizer = get_optimizer(field_comp.net.parameters(), "RPROP")
            loss_data += fit_with_early_stopping(
                field_comp,
                training_set,
                T_conn,
                area_T,
                hist_alpha,
                matprop,
                pffmodel,
                optimizer_dict["weight_decay"],
                num_epochs=n_epochs,
                optimizer=optimizer,
                min_delta=optimizer_dict["optim_rel_tol"],
                intermediateModel_path=intermediateModel_path,
                writer=writer,
                training_dict=training_dict,
            )

        end = time.time()
        print(f"Execution time: {(end-start)/60:.03f}minutes")

        hist_alpha = field_comp.update_hist_alpha(inp)
        torch.save(field_comp.net.state_dict(), trainedModel_path / Path("trained_1NN_" + str(j) + ".pt"))
        with open(trainedModel_path / Path("trainLoss_1NN_" + str(j) + ".npy"), "wb") as file:
            np.save(file, np.asarray(loss_data))


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
    data = np.column_stack(
        (
            inp[:, 0],
            inp[:, 1],
            fields["T"],
            fields["ux"],
            fields["uy"],
            fields["d"],
            fields["HI"],
            fields["HII"],
            fields["He"],
        )
    )
    header = "x,y,T,ux,uy,d,HI,HII,He"
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
        "He": _to_tensor("He"),
    }


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


def _relative_change(curr, prev, eps=1e-12, floor=1e-8):
    num = torch.norm(curr - prev)
    den = torch.maximum(
        torch.maximum(torch.norm(prev), torch.norm(curr)),
        torch.tensor(floor, device=curr.device, dtype=curr.dtype),
    )
    return (num / (den + eps)).item()


def _transform_loss(value, mode="raw", eps=1e-12):
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


def _to_float(val, default=0.0):
    if torch.is_tensor(val):
        return float(val.detach().cpu().item())
    if val is None:
        return float(default)
    return float(val)


def _build_runtime_scale_dict(scale_cfg, inp, area_T, thermo_model, thermal_prop, time_dict, field_comp):
    x = inp[:, -2]
    y = inp[:, -1]
    Lx = float(torch.max(x) - torch.min(x))
    Ly = float(torch.max(y) - torch.min(y))
    L_ref = max(float(scale_cfg.get("L_ref", max(Lx, Ly))), 1e-12)
    uy_rate = _to_float(field_comp.uy_rate, default=0.0)
    U_ref_default = max(uy_rate * float(time_dict["t_end"]), 1e-12)
    U_ref = max(float(scale_cfg.get("U_ref", U_ref_default)), 1e-12)
    T_ref = float(scale_cfg.get("T_ref", _to_float(thermal_prop.Tref)))
    DT_ref = max(
        float(scale_cfg.get("DT_ref", abs(_to_float(thermal_prop.T0) - T_ref))),
        1e-12,
    )
    d_ref = max(float(scale_cfg.get("d_ref", 1.0)), 1e-12)

    eps_ref = float(scale_cfg.get("eps_ref", U_ref / L_ref))
    sigma_ref = float(scale_cfg.get("sigma_ref", _to_float(thermo_model.E0) * eps_ref))
    psi_ref = float(scale_cfg.get("psi_ref", _to_float(thermo_model.E0) * eps_ref ** 2))
    H_ref = float(scale_cfg.get("H_ref", psi_ref))
    area_ref = float(scale_cfg.get("area_ref", L_ref ** 2))
    domain_area = max(float(torch.sum(area_T).detach().cpu().item()), 1e-18)
    dt_ref = max(float(time_dict["dt"]), 1e-12)
    thermal_density_ref = (
        _to_float(thermal_prop.rho) * _to_float(thermal_prop.c) * (DT_ref ** 2) / dt_ref
        + _to_float(thermal_prop.k0) * (DT_ref / L_ref) ** 2
    )
    thermal_loss_ref = float(scale_cfg.get("thermal_loss_ref", thermal_density_ref * domain_area))
    mech_loss_ref = float(scale_cfg.get("mech_loss_ref", psi_ref * domain_area))
    phase_density_ref = max(_to_float(thermo_model.GcI) / max(_to_float(thermo_model.l0), 1e-18), psi_ref)
    phase_loss_ref = float(scale_cfg.get("phase_loss_ref", phase_density_ref * domain_area))
    T_lock_ref = float(scale_cfg.get("T_lock_ref", DT_ref ** 2))
    bc_u_ref = float(scale_cfg.get("bc_u_ref", U_ref ** 2))
    disp_reg_ref = float(scale_cfg.get("disp_reg_ref", U_ref ** 2))
    irrev_ref = float(scale_cfg.get("irrev_ref", d_ref ** 2))

    runtime_scale = {
        "L_ref": L_ref,
        "U_ref": U_ref,
        "T_ref": T_ref,
        "DT_ref": DT_ref,
        "d_ref": d_ref,
        "eps_ref": eps_ref,
        "sigma_ref": sigma_ref,
        "psi_ref": psi_ref,
        "H_ref": H_ref,
        "area_ref": area_ref,
        "thermal_loss_ref": max(thermal_loss_ref, 1e-18),
        "mech_loss_ref": max(mech_loss_ref, 1e-18),
        "phase_loss_ref": max(phase_loss_ref, 1e-18),
        "T_lock_ref": max(T_lock_ref, 1e-18),
        "bc_u_ref": max(bc_u_ref, 1e-18),
        "disp_reg_ref": max(disp_reg_ref, 1e-18),
        "irrev_ref": max(irrev_ref, 1e-18),
    }
    return runtime_scale


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
        mean_w = max(float(np.mean(list(w_init.values()))), eps)
        for k in w_init:
            w_init[k] = w_init[k] / mean_w

    if prev_weights is None:
        return w_init

    w_new = {}
    for k in names:
        prev = float(prev_weights.get(k, w_init[k]))
        w_new[k] = beta * prev + (1.0 - beta) * w_init[k]

    if normalize and len(w_new) > 0:
        mean_w = max(float(np.mean(list(w_new.values()))), eps)
        for k in w_new:
            w_new[k] = w_new[k] / mean_w
    return w_new


def _weighted_sum_nd(term_map, weight_map):
    total = None
    for k, v in term_map.items():
        coeff = float(weight_map.get(k, 1.0))
        term = coeff * v
        total = term if total is None else (total + term)
    if total is None:
        raise ValueError("No terms provided for weighted sum.")
    return total


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

    if x_bottom.numel() >= 3:
        x_sorted, _ = torch.sort(x_bottom)
        dx = x_sorted[1:] - x_sorted[:-1]
        dx = dx[dx > 0]
        if dx.numel() > 0:
            dx_ref = torch.median(dx)
        else:
            dx_ref = torch.tensor(0.0, device=inp.device, dtype=inp.dtype)
    else:
        dx_ref = torch.tensor(0.0, device=inp.device, dtype=inp.dtype)

    lx = torch.clamp(
        torch.max(x) - torch.min(x),
        min=torch.tensor(1e-12, device=inp.device, dtype=inp.dtype),
    )
    ly = torch.clamp(
        torch.max(y) - torch.min(y),
        min=torch.tensor(1e-12, device=inp.device, dtype=inp.dtype),
    )

    notch_rx = torch.maximum(0.03 * lx, 3.0 * dx_ref)
    notch_ry = torch.maximum(0.02 * ly, 2.0 * dx_ref)
    bottom_rx = torch.maximum(0.03 * lx, 3.0 * dx_ref)
    bottom_ry = torch.maximum(0.02 * ly, 2.0 * dx_ref)

    notch_mask = (torch.abs(x - x_tip) <= notch_rx) & (torch.abs(y - y_tip) <= notch_ry)
    bottom_mask = (torch.abs(x - x_fix) <= bottom_rx) & (torch.abs(y - torch.min(y)) <= bottom_ry)

    return notch_mask, bottom_mask


def _safe_region_mean(val, mask):
    if mask is None:
        return torch.tensor(0.0, device=val.device, dtype=val.dtype)
    if torch.any(mask):
        return torch.mean(val[mask])
    return torch.tensor(0.0, device=val.device, dtype=val.dtype)


def _diagnostic_failed(diag_row, fail_hi=1.05):
    r_psi = float(diag_row.get("R_psiII", 0.0))
    r_he = float(diag_row.get("R_He", 0.0))
    r_hii = float(diag_row.get("R_HII", 0.0))
    return (r_psi >= fail_hi) or (r_he >= fail_hi) or (r_hii >= fail_hi)


def _diagnostic_good(diag_row, success_lo=0.95):
    r_psi = float(diag_row.get("R_psiII", 0.0))
    r_he = float(diag_row.get("R_He", 0.0))
    r_hii = float(diag_row.get("R_HII", 0.0))
    return (r_psi <= success_lo) and (r_he <= success_lo) and (r_hii <= success_lo)


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
    Monolithic-net thermo-mechanical mixed-mode phase-field training.
    Step order is fixed:
      thermal -> mechanical -> history candidate -> phase
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

    # Runtime physical scales for nondimensionalized optimization.
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

    mech_top_nodes = bc_dict.get("mechanical_top_nodes", bc_dict["top_nodes"])
    mech_bottom_nodes = bc_dict.get("mechanical_bottom_nodes", bc_dict["bottom_nodes"])
    fixed_point_nodes = bc_dict.get("fixed_point_nodes", bc_dict["point1_node"])
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

    T_prev = torch.full((n_pts,), thermal_prop.T0.item(), device=device)
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
    loss_per_step_rows = []
    diagnostics_rows = []
    physics_rows = []
    trace_iter = 0
    start_step = 1
    target_final_step = len(time_arr) - 1
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
    train_executed = False

    resume_enabled = training_dict.get("resume_if_available", False)
    if resume_enabled:
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
                loss_per_step_rows = _read_dict_rows(loss_per_step_file)
                diagnostics_rows = _read_dict_rows(diagnostics_file)
                physics_rows = _read_dict_rows(physics_consistency_file)
                trace_iter = _last_trace_iter(loss_trace_file)
                start_step = latest_step + 1
                print(f"Resuming unified training from step {latest_step}.")

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
        with open(loss_trace_file, "w", newline="") as file:
            writer_trace = csv.writer(file)
            writer_trace.writerow(["iter", "step", "inner_iter", "branch", "loss"])
        trace_iter = 0
    elif not Path(loss_trace_file).exists():
        with open(loss_trace_file, "w", newline="") as file:
            writer_trace = csv.writer(file)
            writer_trace.writerow(["iter", "step", "inner_iter", "branch", "loss"])

    if start_step >= len(time_arr):
        print("All time steps are already available. Skipping training.")
        run_meta = {
            "train_executed": False,
            "steps_before": steps_before,
            "steps_after": steps_before,
            "new_steps_generated": 0,
            "eligible_for_escalation": False,
            "target_final_step": target_final_step,
            "partial_steps": progress_before.get("partial_steps", []),
        }
        if return_run_meta:
            return inp, T_conn, area_T, bc_dict, run_meta
        return inp, T_conn, area_T, bc_dict

    tol_loss = training_dict.get("tol_loss", 1e-4)
    tol_field_T = training_dict.get("tol_field_T", 1e-3)
    tol_field_u = training_dict.get("tol_field_u", 1e-3)
    tol_field_d = training_dict.get("tol_field_d", 1e-3)
    tol_hist = training_dict.get("tol_hist", 1e-4)
    max_inner_iters = int(training_dict.get("max_inner_iters", 3))
    conv_patience = int(training_dict.get("conv_patience", 2))
    auto_weight_cfg = dict(training_dict.get("auto_weight_dict", {}))
    auto_weight_enabled = bool(auto_weight_cfg.get("enabled", False))
    branch_auto_cfg = dict(training_dict.get("branch_auto_weight_dict", {}))
    thermal_auto_enabled = auto_weight_enabled and bool(branch_auto_cfg.get("thermal", True))
    mechanical_auto_enabled = auto_weight_enabled and bool(branch_auto_cfg.get("mechanical", True))
    phase_auto_enabled = auto_weight_enabled and bool(branch_auto_cfg.get("phase", True))
    history_update_mode = str(training_dict.get("history_update_mode", "inner_accumulate")).lower()
    phase_balance_mode = str(training_dict.get("phase_balance_mode", "none")).lower()
    phase_balance_target_ratio = float(training_dict.get("phase_balance_target_ratio", 0.2))
    phase_balance_min = float(training_dict.get("phase_balance_min", 1.0))
    phase_balance_max = float(training_dict.get("phase_balance_max", 1e6))
    loss_transform_mode = str(training_dict.get("loss_transform_mode", "raw")).lower()
    if history_update_mode not in ("inner_accumulate", "step_end"):
        raise ValueError("training_dict['history_update_mode'] must be 'inner_accumulate' or 'step_end'.")

    notch_mask, bottom_mask = _infer_notch_tip_region_masks(inp, bc_dict)
    diagnostic_steps = set(int(s) for s in training_dict.get("diagnostic_steps", [1, 10, 20, 64]))
    auto_rebalance_enabled = bool(training_dict.get("auto_rebalance_enabled", True))
    auto_rebalance_fail_streak = int(training_dict.get("auto_rebalance_fail_streak", 2))
    auto_rebalance_growth = float(training_dict.get("auto_rebalance_growth", 1.6))
    auto_rebalance_max_target_ratio = float(training_dict.get("auto_rebalance_max_target_ratio", 5.0))
    auto_rebalance_good_streak = int(training_dict.get("auto_rebalance_good_streak", 3))
    auto_rebalance_decay = float(training_dict.get("auto_rebalance_decay", 0.85))
    auto_rebalance_min_target_ratio = float(training_dict.get("auto_rebalance_min_target_ratio", 0.1))
    rebalance_fail_hi = float(training_dict.get("rebalance_fail_hi", 1.05))
    rebalance_success_lo = float(training_dict.get("rebalance_success_lo", 0.95))
    diagnostic_fail_streak = 0
    diagnostic_good_streak = 0

    max_macro_stress_so_far = 0.0
    branch_weight_state = {"thermal": None, "mechanical": None, "phase": None}

    for step_idx in range(start_step, len(time_arr)):
        train_executed = True
        t_now = torch.tensor(time_arr[step_idx], device=device)
        dt_now = torch.tensor(time_arr[step_idx] - time_arr[step_idx - 1], device=device)
        field_comp.set_time(t_now)

        inner_prev_total = None
        conv_count = 0
        T_iter_prev = T_prev.detach().clone()
        ux_iter_prev = ux_prev.detach().clone()
        uy_iter_prev = uy_prev.detach().clone()
        d_iter_prev = d_prev.detach().clone()
        HI_elem_iter_prev = HI_elem_prev.detach().clone()
        HII_elem_iter_prev = HII_elem_prev.detach().clone()
        accepted_state = None
        phase_weight_curr = float(training_dict.get("w_phase_global", 1.0))

        step_loss_series = {"thermal": [], "mech": [], "phase": []}

        for inner_iter in range(1, max_inner_iters + 1):
            thermal_weights_inner = None
            mech_weights_inner = None
            phase_weights_inner = None

            def _current_T_bc_target():
                return get_thermal_dirichlet_target(
                    bc_dict=bc_dict,
                    thermal_prop=thermal_prop,
                    time_value=t_now,
                    idx_dirichlet=bc_dict["thermal_dirichlet_nodes"],
                    device=inp.device,
                    dtype=inp.dtype,
                )

            def thermal_loss_fn():
                nonlocal thermal_weights_inner
                T_bc_target = _current_T_bc_target()
                T_pred, ux_pred, uy_pred, _ = field_comp.fieldCalculation_tm(
                    inp=inp,
                    thermal_dirichlet_nodes=bc_dict["thermal_dirichlet_nodes"],
                    T_bc_value=T_bc_target,
                    mechanical_top_nodes=mech_top_nodes,
                    mechanical_bottom_nodes=mech_bottom_nodes,
                    fixed_point_nodes=fixed_point_nodes,
                )
                loss_dom, loss_bc, thermal_terms = compute_thermal_loss(
                    inp=inp,
                    T=T_pred,
                    T_prev=T_prev,
                    d_prev=d_iter_prev,
                    area_elem=area_T,
                    T_conn=T_conn,
                    thermo_model=thermo_model,
                    thermal_prop=thermal_prop,
                    dt=dt_now,
                    bc_dict=bc_dict,
                    time_value=t_now,
                )
                loss_disp_reg = torch.mean((ux_pred - ux_iter_prev) ** 2 + (uy_pred - uy_iter_prev) ** 2)
                disp_reg_ref = torch.tensor(
                    float(runtime_scale.get("disp_reg_ref", 1.0)),
                    device=inp.device,
                    dtype=inp.dtype,
                )
                disp_reg_ref = torch.clamp(
                    disp_reg_ref,
                    min=torch.tensor(1e-18, device=inp.device, dtype=inp.dtype),
                )
                disp_reg_nd = loss_disp_reg / disp_reg_ref

                nd_terms = {
                    "pde_nd": thermal_terms["pde_nd"],
                    "bc_nd": thermal_terms["bc_nd"],
                    "disp_reg_nd": disp_reg_nd,
                }
                if thermal_auto_enabled:
                    if thermal_weights_inner is None:
                        thermal_weights_inner = _make_auto_weights(
                            term_map=nd_terms,
                            cfg=auto_weight_cfg,
                            prev_weights=branch_weight_state["thermal"],
                        )
                    loss_train_nd = _weighted_sum_nd(nd_terms, thermal_weights_inner)
                    w_pde = thermal_weights_inner["pde_nd"]
                    w_bc = thermal_weights_inner["bc_nd"]
                    w_disp = thermal_weights_inner["disp_reg_nd"]
                else:
                    w_pde = 1.0
                    w_bc = float(training_dict["w_bc_T"])
                    w_disp = float(training_dict["w_disp_reg"])
                    loss_train_nd = w_pde * nd_terms["pde_nd"] + w_bc * nd_terms["bc_nd"] + w_disp * nd_terms["disp_reg_nd"]

                loss_total = _transform_loss(loss_train_nd, mode=loss_transform_mode)
                return loss_total, {
                    "domain": loss_dom,
                    "bc": loss_bc,
                    "disp_reg": loss_disp_reg,
                    "thermal_pde_raw": thermal_terms["pde_raw"],
                    "thermal_bc_raw": thermal_terms["bc_raw"],
                    "thermal_pde_nd": thermal_terms["pde_nd"],
                    "thermal_bc_nd": thermal_terms["bc_nd"],
                    "thermal_disp_reg_nd": disp_reg_nd,
                    "w_th_pde": torch.tensor(w_pde, device=inp.device, dtype=inp.dtype),
                    "w_th_bc": torch.tensor(w_bc, device=inp.device, dtype=inp.dtype),
                    "w_th_disp": torch.tensor(w_disp, device=inp.device, dtype=inp.dtype),
                }

            loss_thermal = fit_tm(
                loss_fn=thermal_loss_fn,
                params=field_comp.net.parameters(),
                optimizer_dict=optimizer_dict,
                n_epochs_lbfgs=optimizer_dict["n_epochs_LBFGS_thermal"],
                n_epochs_rprop=optimizer_dict["n_epochs_RPROP_thermal"],
                min_delta=optimizer_dict["optim_rel_tol"],
                writer=writer,
                writer_tag=f"thermal/step_{step_idx}/iter_{inner_iter}",
            )
            step_loss_series["thermal"].extend(loss_thermal)
            trace_iter = _append_loss_trace(
                loss_trace_file=loss_trace_file,
                branch="thermal",
                step_idx=step_idx,
                loss_list=loss_thermal,
                iter_counter=trace_iter,
                inner_iter=inner_iter,
            )
            if thermal_auto_enabled and (thermal_weights_inner is not None):
                branch_weight_state["thermal"] = dict(thermal_weights_inner)

            with torch.no_grad():
                T_bc_target = _current_T_bc_target()
                T_curr, _, _, _ = field_comp.fieldCalculation_tm(
                    inp=inp,
                    thermal_dirichlet_nodes=bc_dict["thermal_dirichlet_nodes"],
                    T_bc_value=T_bc_target,
                    mechanical_top_nodes=mech_top_nodes,
                    mechanical_bottom_nodes=mech_bottom_nodes,
                    fixed_point_nodes=fixed_point_nodes,
                )

            def mech_loss_fn():
                nonlocal mech_weights_inner
                T_curr_detached = T_curr.detach()
                T_bc_target = _current_T_bc_target()
                T_pred, ux_pred, uy_pred, _ = field_comp.fieldCalculation_tm(
                    inp=inp,
                    thermal_dirichlet_nodes=bc_dict["thermal_dirichlet_nodes"],
                    T_bc_value=T_bc_target,
                    mechanical_top_nodes=mech_top_nodes,
                    mechanical_bottom_nodes=mech_bottom_nodes,
                    fixed_point_nodes=fixed_point_nodes,
                )
                loss_mech, mech_state_local = compute_mech_loss(
                    inp=inp,
                    ux=ux_pred,
                    uy=uy_pred,
                    T_phys=T_curr_detached,
                    d_prev=d_iter_prev,
                    area_elem=area_T,
                    T_conn=T_conn,
                    thermo_model=thermo_model,
                    thermal_prop=thermal_prop,
                    scale_dict=runtime_scale,
                )
                loss_T_lock = torch.mean((T_pred - T_curr_detached) ** 2)
                bottom_fix_mode = str(training_dict.get("bottom_fix_mode", "uxuy")).lower()
                if bottom_fix_mode in ("uy_only", "y_only", "roller"):
                    loss_bc_u = torch.mean(uy_pred[mech_bottom_nodes] ** 2)
                else:
                    loss_bc_u = torch.mean(ux_pred[mech_bottom_nodes] ** 2 + uy_pred[mech_bottom_nodes] ** 2)
                if fixed_point_nodes is not None and fixed_point_nodes.numel() > 0:
                    loss_bc_u = loss_bc_u + torch.mean(
                        ux_pred[fixed_point_nodes] ** 2 + uy_pred[fixed_point_nodes] ** 2
                    )
                T_lock_ref = torch.tensor(
                    float(runtime_scale.get("T_lock_ref", 1.0)),
                    device=inp.device,
                    dtype=inp.dtype,
                )
                bc_u_ref = torch.tensor(
                    float(runtime_scale.get("bc_u_ref", 1.0)),
                    device=inp.device,
                    dtype=inp.dtype,
                )
                T_lock_ref = torch.clamp(
                    T_lock_ref,
                    min=torch.tensor(1e-18, device=inp.device, dtype=inp.dtype),
                )
                bc_u_ref = torch.clamp(
                    bc_u_ref,
                    min=torch.tensor(1e-18, device=inp.device, dtype=inp.dtype),
                )
                T_lock_nd = loss_T_lock / T_lock_ref
                bc_u_nd = loss_bc_u / bc_u_ref
                nd_terms = {"mech_nd": mech_state_local["mech_nd"], "T_lock_nd": T_lock_nd, "bc_u_nd": bc_u_nd}

                if mechanical_auto_enabled:
                    if mech_weights_inner is None:
                        mech_weights_inner = _make_auto_weights(
                            term_map=nd_terms,
                            cfg=auto_weight_cfg,
                            prev_weights=branch_weight_state["mechanical"],
                        )
                    loss_train_nd = _weighted_sum_nd(nd_terms, mech_weights_inner)
                    w_mech = mech_weights_inner["mech_nd"]
                    w_tlock = mech_weights_inner["T_lock_nd"]
                    w_bcu = mech_weights_inner["bc_u_nd"]
                else:
                    w_mech = 1.0
                    w_tlock = float(training_dict["w_T_lock"])
                    w_bcu = float(training_dict["w_bc_u"])
                    loss_train_nd = w_mech * nd_terms["mech_nd"] + w_tlock * nd_terms["T_lock_nd"] + w_bcu * nd_terms["bc_u_nd"]

                loss_total = _transform_loss(loss_train_nd, mode=loss_transform_mode)
                return loss_total, {
                    "mech": loss_mech,
                    "T_lock": loss_T_lock,
                    "bc_u": loss_bc_u,
                    "mech_raw": mech_state_local["mech_raw"],
                    "mech_nd": mech_state_local["mech_nd"],
                    "T_lock_nd": T_lock_nd,
                    "bc_u_nd": bc_u_nd,
                    "w_mech": torch.tensor(w_mech, device=inp.device, dtype=inp.dtype),
                    "w_Tlock": torch.tensor(w_tlock, device=inp.device, dtype=inp.dtype),
                    "w_bcu": torch.tensor(w_bcu, device=inp.device, dtype=inp.dtype),
                }

            loss_mech = fit_tm(
                loss_fn=mech_loss_fn,
                params=field_comp.net.parameters(),
                optimizer_dict=optimizer_dict,
                n_epochs_lbfgs=optimizer_dict["n_epochs_LBFGS_mech"],
                n_epochs_rprop=optimizer_dict["n_epochs_RPROP_mech"],
                min_delta=optimizer_dict["optim_rel_tol"],
                writer=writer,
                writer_tag=f"mech/step_{step_idx}/iter_{inner_iter}",
            )
            step_loss_series["mech"].extend(loss_mech)
            trace_iter = _append_loss_trace(
                loss_trace_file=loss_trace_file,
                branch="mech",
                step_idx=step_idx,
                loss_list=loss_mech,
                iter_counter=trace_iter,
                inner_iter=inner_iter,
            )
            if mechanical_auto_enabled and (mech_weights_inner is not None):
                branch_weight_state["mechanical"] = dict(mech_weights_inner)

            with torch.no_grad():
                T_curr_detached = T_curr.detach()
                T_bc_target = _current_T_bc_target()
                T_curr, ux_curr, uy_curr, _ = field_comp.fieldCalculation_tm(
                    inp=inp,
                    thermal_dirichlet_nodes=bc_dict["thermal_dirichlet_nodes"],
                    T_bc_value=T_bc_target,
                    mechanical_top_nodes=mech_top_nodes,
                    mechanical_bottom_nodes=mech_bottom_nodes,
                    fixed_point_nodes=fixed_point_nodes,
                )
                _, mech_state = compute_mech_loss(
                    inp=inp,
                    ux=ux_curr,
                    uy=uy_curr,
                    T_phys=T_curr_detached,
                    d_prev=d_iter_prev,
                    area_elem=area_T,
                    T_conn=T_conn,
                    thermo_model=thermo_model,
                    thermal_prop=thermal_prop,
                    scale_dict=runtime_scale,
                )
                psi_I_elem = mech_state["psi_I"]
                psi_II_elem = mech_state["psi_II"]
                if history_update_mode == "step_end":
                    # COMSOL-like timing: candidate uses step-start frozen history.
                    HI_hist_ref = HI_elem_prev
                    HII_hist_ref = HII_elem_prev
                else:
                    # Legacy PINN timing: candidate accumulates with inner-iteration history.
                    HI_hist_ref = HI_elem_iter_prev
                    HII_hist_ref = HII_elem_iter_prev
                HI_elem_cand = torch.maximum(HI_hist_ref, psi_I_elem)
                HII_elem_cand = torch.maximum(HII_hist_ref, psi_II_elem)
                He_elem_curr = HI_elem_cand + thermo_model.gc_ratio * HII_elem_cand
                HI_nodes_cand, HII_nodes_cand, He_nodes_cand = _history_nodes_from_elem(HI_elem_cand, HII_elem_cand)

            field_comp.set_prev_damage(d_iter_prev)

            def phase_loss_fn():
                nonlocal phase_weights_inner
                d_pred, _ = field_comp.fieldCalculation_phase(inp=inp, d_prev=d_iter_prev, return_raw=True)
                loss_pf, loss_ir, phase_components = compute_phase_loss(
                    inp=inp,
                    d=d_pred,
                    d_prev=d_iter_prev,
                    He=He_nodes_cand,
                    He_elem=He_elem_curr,
                    area_elem=area_T,
                    T_conn=T_conn,
                    thermo_model=thermo_model,
                    dt=dt_now,
                    irreversibility_weight=training_dict.get("w_irrev", 1.0),
                    return_components=True,
                    scale_dict=runtime_scale,
                )
                nd_terms = {
                    "crack_density_nd": phase_components["crack_density_nd"],
                    "reaction_nd": phase_components["reaction_nd"],
                    "viscosity_nd": phase_components["viscosity_nd"],
                    "ir_nd": phase_components["ir_nd"],
                }
                mag_terms = {
                    "crack_density_nd": phase_components["crack_density_mag_nd"],
                    "reaction_nd": phase_components["reaction_mag_nd"],
                    "viscosity_nd": phase_components["viscosity_mag_nd"],
                    "ir_nd": phase_components["ir_mag_nd"],
                }
                if phase_auto_enabled:
                    if phase_weights_inner is None:
                        phase_weights_inner = _make_auto_weights(
                            term_map=nd_terms,
                            cfg=auto_weight_cfg,
                            prev_weights=branch_weight_state["phase"],
                            magnitude_map=mag_terms,
                        )
                    loss_train_nd = _weighted_sum_nd(nd_terms, phase_weights_inner)
                    w_cd = phase_weights_inner["crack_density_nd"]
                    w_reac = phase_weights_inner["reaction_nd"]
                    w_visc = phase_weights_inner["viscosity_nd"]
                    w_ir = phase_weights_inner["ir_nd"]
                else:
                    w_cd = 1.0
                    w_reac = 1.0
                    w_visc = 1.0
                    w_ir = float(training_dict["w_irrev"])
                    loss_train_nd = (
                        w_cd * nd_terms["crack_density_nd"]
                        + w_reac * nd_terms["reaction_nd"]
                        + w_visc * nd_terms["viscosity_nd"]
                        + w_ir * nd_terms["ir_nd"]
                    )
                loss_total = _transform_loss(phase_weight_curr * loss_train_nd, mode=loss_transform_mode)
                return loss_total, {
                    "pf": loss_pf,
                    "ir": loss_ir,
                    "w_phase": torch.tensor(phase_weight_curr, device=inp.device, dtype=inp.dtype),
                    "w_cd": torch.tensor(w_cd, device=inp.device, dtype=inp.dtype),
                    "w_reac": torch.tensor(w_reac, device=inp.device, dtype=inp.dtype),
                    "w_visc": torch.tensor(w_visc, device=inp.device, dtype=inp.dtype),
                    "w_ir": torch.tensor(w_ir, device=inp.device, dtype=inp.dtype),
                    **phase_components,
                }

            loss_phase = fit_tm(
                loss_fn=phase_loss_fn,
                params=field_comp.net.parameters(),
                optimizer_dict=optimizer_dict,
                n_epochs_lbfgs=optimizer_dict["n_epochs_LBFGS_phase"],
                n_epochs_rprop=optimizer_dict["n_epochs_RPROP_phase"],
                min_delta=optimizer_dict["optim_rel_tol"],
                writer=writer,
                writer_tag=f"phase/step_{step_idx}/iter_{inner_iter}",
            )
            step_loss_series["phase"].extend(loss_phase)
            trace_iter = _append_loss_trace(
                loss_trace_file=loss_trace_file,
                branch="phase",
                step_idx=step_idx,
                loss_list=loss_phase,
                iter_counter=trace_iter,
                inner_iter=inner_iter,
            )
            if phase_auto_enabled and (phase_weights_inner is not None):
                branch_weight_state["phase"] = dict(phase_weights_inner)

            with torch.no_grad():
                d_curr = field_comp.fieldCalculation_phase(inp=inp, d_prev=d_iter_prev, return_raw=False)
                thermal_total_eval, thermal_terms_eval = thermal_loss_fn()
                mech_total_eval, mech_terms_eval = mech_loss_fn()
                phase_total_eval, phase_terms_eval = phase_loss_fn()
                total_loss_eval = (thermal_total_eval + mech_total_eval + phase_total_eval).item()

                rel_loss = np.inf if inner_prev_total is None else abs(total_loss_eval - inner_prev_total) / (
                    abs(inner_prev_total) + np.finfo(float).eps
                )
                rel_T = _relative_change(T_curr, T_iter_prev)
                u_curr_vec = torch.stack((ux_curr, uy_curr), dim=1)
                u_prev_vec = torch.stack((ux_iter_prev, uy_iter_prev), dim=1)
                rel_u = _relative_change(u_curr_vec, u_prev_vec)
                rel_d = _relative_change(d_curr, d_iter_prev)
                rel_HI = _relative_change(HI_elem_cand, HI_elem_iter_prev)
                rel_HII = _relative_change(HII_elem_cand, HII_elem_iter_prev)

                converged_now = (
                    rel_loss < tol_loss
                    and rel_T < tol_field_T
                    and rel_u < tol_field_u
                    and rel_d < tol_field_d
                    and rel_HI < tol_hist
                    and rel_HII < tol_hist
                )
                conv_count = conv_count + 1 if converged_now else 0

                if phase_balance_mode == "adaptive":
                    mech_ref = abs(float(mech_total_eval.item()))
                    phase_ref = abs(float(phase_terms_eval.get("pf", phase_total_eval).item()))
                    denom = max(phase_ref, 1e-16)
                    target_w = (phase_balance_target_ratio * mech_ref) / denom
                    target_w = float(np.clip(target_w, phase_balance_min, phase_balance_max))
                    phase_weight_curr = target_w

                accepted_state = {
                    "inner_iters": inner_iter,
                    "converged": conv_count >= conv_patience,
                    "T_curr": T_curr.detach(),
                    "ux_curr": ux_curr.detach(),
                    "uy_curr": uy_curr.detach(),
                    "d_curr": d_curr.detach(),
                    "psi_I_elem": psi_I_elem.detach(),
                    "psi_II_elem": psi_II_elem.detach(),
                    "HI_nodes_cand": HI_nodes_cand.detach(),
                    "HII_nodes_cand": HII_nodes_cand.detach(),
                    "He_nodes_cand": He_nodes_cand.detach(),
                    "HI_elem_cand": HI_elem_cand.detach(),
                    "HII_elem_cand": HII_elem_cand.detach(),
                    "He_elem_curr": He_elem_curr.detach(),
                    "mech_state": mech_state,
                    "thermal_total": thermal_total_eval.item(),
                    "mech_total": mech_total_eval.item(),
                    "phase_total": phase_total_eval.item(),
                    "phase_weight": float(phase_weight_curr),
                    "thermal_terms": {k: v.item() for k, v in thermal_terms_eval.items()},
                    "mech_terms": {k: v.item() for k, v in mech_terms_eval.items()},
                    "phase_terms": {k: v.item() for k, v in phase_terms_eval.items()},
                    "total_loss": total_loss_eval,
                    "rel_loss": rel_loss,
                    "rel_T": rel_T,
                    "rel_u": rel_u,
                    "rel_d": rel_d,
                    "rel_HI": rel_HI,
                    "rel_HII": rel_HII,
                }

                inner_prev_total = total_loss_eval
                T_iter_prev = T_curr.detach()
                ux_iter_prev = ux_curr.detach()
                uy_iter_prev = uy_curr.detach()
                d_iter_prev = d_curr.detach()
                # Keep previous inner candidate for convergence diagnostics.
                # In "step_end" mode this does not change candidate history reference.
                HI_elem_iter_prev = HI_elem_cand.detach()
                HII_elem_iter_prev = HII_elem_cand.detach()

            if conv_count >= conv_patience:
                break

        if accepted_state is None:
            raise RuntimeError(f"Step {step_idx}: no valid state was produced.")

        if bool(training_dict.get("require_converged_step", False)) and (not accepted_state["converged"]):
            raise RuntimeError(
                f"Step {step_idx}: convergence criteria not met within max_inner_iters={max_inner_iters}."
            )

        with torch.no_grad():
            if history_update_mode == "step_end":
                # COMSOL-like step-end update: freeze history only from accepted step state.
                HI_elem_prev = torch.maximum(HI_elem_prev, accepted_state["psi_I_elem"]).detach()
                HII_elem_prev = torch.maximum(HII_elem_prev, accepted_state["psi_II_elem"]).detach()
            else:
                HI_elem_prev = accepted_state["HI_elem_cand"].detach()
                HII_elem_prev = accepted_state["HII_elem_cand"].detach()
            HI_curr = accepted_state["HI_nodes_cand"].detach()
            HII_curr = accepted_state["HII_nodes_cand"].detach()
            He_out = accepted_state["He_nodes_cand"].detach()

            T_curr = accepted_state["T_curr"]
            ux_curr = accepted_state["ux_curr"]
            uy_curr = accepted_state["uy_curr"]
            d_curr = accepted_state["d_curr"]
            mech_state = accepted_state["mech_state"]

            sig_yy_elem = mech_state["stress"]["sig_yy"]
            reaction_force = _reaction_force(sig_yy_elem, top_edges, top_elems, inp, thermal_prop.thk)
            uy_top = torch.mean(uy_curr[mech_top_nodes])
            macro_strain = uy_top / torch.tensor(10e-6, device=device)
            macro_stress = reaction_force / (thermal_prop.thk * torch.tensor(10e-6, device=device))

            curve_rows.append(
                {
                    "step": step_idx,
                    "time": float(time_arr[step_idx]),
                    "uy_top": uy_top.item(),
                    "reaction_force": reaction_force.item(),
                    "macro_strain": macro_strain.item(),
                    "macro_stress": macro_stress.item(),
                }
            )
            _write_dict_rows(
                file_name=curve_file,
                fieldnames=["step", "time", "uy_top", "reaction_force", "macro_strain", "macro_stress"],
                rows=curve_rows,
            )

            fields_np = {
                "T": T_curr.detach().cpu().numpy(),
                "ux": ux_curr.detach().cpu().numpy(),
                "uy": uy_curr.detach().cpu().numpy(),
                "d": d_curr.detach().cpu().numpy(),
                "HI": HI_curr.detach().cpu().numpy(),
                "HII": HII_curr.detach().cpu().numpy(),
                "He": He_out.detach().cpu().numpy(),
            }
            _save_field_csv(
                field_path / Path(f"field_step_{step_idx:04d}.csv"),
                inp.detach().cpu().numpy(),
                fields_np,
            )

            thermal_loss_final = accepted_state["thermal_total"]
            mech_loss_final = accepted_state["mech_total"]
            phase_loss_final = accepted_state["phase_total"]
            total_loss_final = accepted_state["total_loss"]
            irrev_loss = accepted_state["phase_terms"].get("ir", 0.0)
            boundary_loss = accepted_state["thermal_terms"].get("bc", 0.0) + accepted_state["mech_terms"].get("bc_u", 0.0)
            E_el = accepted_state["mech_terms"].get("mech", 0.0)
            E_phase_domain = accepted_state["phase_terms"].get("E_phase_domain", accepted_state["phase_terms"].get("pf", 0.0))
            E_crack_density = accepted_state["phase_terms"].get("E_crack_density", 0.0)
            E_reaction = accepted_state["phase_terms"].get("E_reaction", 0.0)
            E_viscosity = accepted_state["phase_terms"].get("E_viscosity", 0.0)

            loss_per_step_rows.append(
                {
                    "step": step_idx,
                    "time": float(time_arr[step_idx]),
                    "inner_iters": accepted_state["inner_iters"],
                    "converged": int(accepted_state["converged"]),
                    "thermal_loss": thermal_loss_final,
                    "mech_loss": mech_loss_final,
                    "phase_loss": phase_loss_final,
                    "phase_weight": accepted_state.get("phase_weight", float(training_dict.get("w_phase_global", 1.0))),
                    "irreversibility_loss": irrev_loss,
                    "boundary_loss": boundary_loss,
                    "E_el": E_el,
                    "E_phase_domain": E_phase_domain,
                    "E_crack_density": E_crack_density,
                    "E_reaction": E_reaction,
                    "E_viscosity": E_viscosity,
                    "total_loss": total_loss_final,
                    "loss_total": total_loss_final,
                    "loss_T": thermal_loss_final,
                    "loss_u": mech_loss_final,
                    "loss_d": phase_loss_final,
                    "loss_irrev": training_dict["w_irrev"] * irrev_loss,
                    "loss_bc": boundary_loss,
                    "reaction_top": reaction_force.item(),
                    "macro_stress": macro_stress.item(),
                    "macro_strain": macro_strain.item(),
                    "max_d": torch.max(d_curr).item(),
                    "max_HI": torch.max(HI_curr).item(),
                    "max_HII": torch.max(HII_curr).item(),
                    "rel_loss": accepted_state["rel_loss"],
                    "rel_T": accepted_state["rel_T"],
                    "rel_u": accepted_state["rel_u"],
                    "rel_d": accepted_state["rel_d"],
                    "rel_HI": accepted_state["rel_HI"],
                    "rel_HII": accepted_state["rel_HII"],
                    "thermal_pde_raw": accepted_state["thermal_terms"].get("thermal_pde_raw", accepted_state["thermal_terms"].get("domain", 0.0)),
                    "thermal_bc_raw": accepted_state["thermal_terms"].get("thermal_bc_raw", accepted_state["thermal_terms"].get("bc", 0.0)),
                    "thermal_pde_nd": accepted_state["thermal_terms"].get("thermal_pde_nd", 0.0),
                    "thermal_bc_nd": accepted_state["thermal_terms"].get("thermal_bc_nd", 0.0),
                    "thermal_disp_reg_nd": accepted_state["thermal_terms"].get("thermal_disp_reg_nd", 0.0),
                    "w_th_pde": accepted_state["thermal_terms"].get("w_th_pde", 1.0),
                    "w_th_bc": accepted_state["thermal_terms"].get("w_th_bc", 1.0),
                    "w_th_disp": accepted_state["thermal_terms"].get("w_th_disp", 1.0),
                    "mech_raw": accepted_state["mech_terms"].get("mech_raw", accepted_state["mech_terms"].get("mech", 0.0)),
                    "mech_nd": accepted_state["mech_terms"].get("mech_nd", 0.0),
                    "T_lock_raw": accepted_state["mech_terms"].get("T_lock", 0.0),
                    "T_lock_nd": accepted_state["mech_terms"].get("T_lock_nd", 0.0),
                    "bc_u_raw": accepted_state["mech_terms"].get("bc_u", 0.0),
                    "bc_u_nd": accepted_state["mech_terms"].get("bc_u_nd", 0.0),
                    "w_mech": accepted_state["mech_terms"].get("w_mech", 1.0),
                    "w_Tlock": accepted_state["mech_terms"].get("w_Tlock", 1.0),
                    "w_bcu": accepted_state["mech_terms"].get("w_bcu", 1.0),
                    "crack_density_raw": accepted_state["phase_terms"].get("crack_density_raw", 0.0),
                    "reaction_raw": accepted_state["phase_terms"].get("reaction_raw", 0.0),
                    "viscosity_raw": accepted_state["phase_terms"].get("viscosity_raw", 0.0),
                    "phase_domain_raw": accepted_state["phase_terms"].get("phase_domain_raw", accepted_state["phase_terms"].get("pf", 0.0)),
                    "crack_density_nd": accepted_state["phase_terms"].get("crack_density_nd", 0.0),
                    "reaction_nd": accepted_state["phase_terms"].get("reaction_nd", 0.0),
                    "viscosity_nd": accepted_state["phase_terms"].get("viscosity_nd", 0.0),
                    "phase_domain_nd": accepted_state["phase_terms"].get("phase_domain_nd", 0.0),
                    "ir_nd": accepted_state["phase_terms"].get("ir_nd", 0.0),
                    "w_cd": accepted_state["phase_terms"].get("w_cd", 1.0),
                    "w_reac": accepted_state["phase_terms"].get("w_reac", 1.0),
                    "w_visc": accepted_state["phase_terms"].get("w_visc", 1.0),
                    "w_ir": accepted_state["phase_terms"].get("w_ir", 1.0),
                }
            )
            loss_logger.write_step_rows(loss_per_step_rows)
            loss_logger.save_step_loss_artifacts(step_idx=step_idx, branch_losses=step_loss_series)

            # --- Physics consistency audit (no training-force, logging only) ---
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
                    T_bc_err = torch.mean(torch.abs(T_curr[T_bc_nodes] - T_bc_target.reshape(1)))
                else:
                    T_bc_err = torch.mean(torch.abs(T_curr[T_bc_nodes] - T_bc_target.reshape(-1)))

            uy_top_target = field_comp.uy_rate * field_comp.time
            uy_top_err = torch.mean(torch.abs(uy_curr[mech_top_nodes] - uy_top_target))
            uy_bottom_err = torch.mean(torch.abs(uy_curr[mech_bottom_nodes]))

            g_elem = thermo_model.degradation(nodal_to_element(d_curr, T_conn)).detach()
            sigyy_abs = torch.abs(mech_state["stress"]["sig_yy"]).detach()
            stress_degrade_corr = torch.tensor(0.0, device=inp.device, dtype=inp.dtype)
            if g_elem.numel() > 1 and sigyy_abs.numel() > 1:
                g0 = g_elem - torch.mean(g_elem)
                s0 = sigyy_abs - torch.mean(sigyy_abs)
                denom = torch.sqrt(torch.sum(g0 ** 2) * torch.sum(s0 ** 2) + 1e-16)
                stress_degrade_corr = torch.sum(g0 * s0) / denom

            d_drop_max = torch.max(torch.clamp(d_prev - d_curr, min=0.0))
            HI_drop_max = torch.max(torch.clamp(HI_prev_nodes - HI_curr, min=0.0))
            HII_drop_max = torch.max(torch.clamp(HII_prev_nodes - HII_curr, min=0.0))

            softening_started = 1 if (step_idx > 1 and macro_stress.item() < max_macro_stress_so_far) else 0
            max_macro_stress_so_far = max(max_macro_stress_so_far, macro_stress.item())

            physics_row = {
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
                "softening_started": int(softening_started),
            }
            physics_rows.append(physics_row)
            _write_dict_rows(
                file_name=physics_consistency_file,
                fieldnames=list(physics_rows[-1].keys()),
                rows=physics_rows,
            )

            if step_idx in diagnostic_steps:
                exy_elem = mech_state["exy_e"].detach()
                psi_ii_elem = accepted_state["psi_II_elem"].detach()
                hii_elem = accepted_state["HII_elem_cand"].detach()
                he_elem = accepted_state["He_elem_curr"].detach()
                exy_nodes = element_to_nodal(exy_elem, T_conn, n_pts, area_elem=area_T)
                psi_ii_nodes = element_to_nodal(psi_ii_elem, T_conn, n_pts, area_elem=area_T)
                hii_nodes = element_to_nodal(hii_elem, T_conn, n_pts, area_elem=area_T)
                he_nodes = element_to_nodal(he_elem, T_conn, n_pts, area_elem=area_T)
                d_nodes = d_curr.detach()

                exy_notch = _safe_region_mean(torch.abs(exy_nodes), notch_mask).item()
                exy_bottom = _safe_region_mean(torch.abs(exy_nodes), bottom_mask).item()
                psi_notch = _safe_region_mean(psi_ii_nodes, notch_mask).item()
                psi_bottom = _safe_region_mean(psi_ii_nodes, bottom_mask).item()
                hii_notch = _safe_region_mean(hii_nodes, notch_mask).item()
                hii_bottom = _safe_region_mean(hii_nodes, bottom_mask).item()
                he_notch = _safe_region_mean(he_nodes, notch_mask).item()
                he_bottom = _safe_region_mean(he_nodes, bottom_mask).item()
                d_notch = _safe_region_mean(d_nodes, notch_mask).item()
                d_bottom = _safe_region_mean(d_nodes, bottom_mask).item()
                diag_row = {
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
                    "phase_weight": accepted_state.get("phase_weight", float(training_dict.get("w_phase_global", 1.0))),
                    "phase_balance_target_ratio": float(phase_balance_target_ratio),
                    "diagnostic_fail_streak": int(diagnostic_fail_streak),
                    "diagnostic_good_streak": int(diagnostic_good_streak),
                    "rebalance_fail_hi": float(rebalance_fail_hi),
                    "rebalance_success_lo": float(rebalance_success_lo),
                    "auto_rebalance_applied": 0,
                    "auto_rebalance_up": 0,
                    "auto_rebalance_down": 0,
                }

                failed_now = _diagnostic_failed(diag_row, fail_hi=rebalance_fail_hi)
                good_now = _diagnostic_good(diag_row, success_lo=rebalance_success_lo)
                if failed_now:
                    diagnostic_fail_streak += 1
                    diagnostic_good_streak = 0
                elif good_now:
                    diagnostic_good_streak += 1
                    diagnostic_fail_streak = 0
                else:
                    diagnostic_fail_streak = 0
                    diagnostic_good_streak = 0

                rebalance_up_applied = 0
                rebalance_down_applied = 0
                if auto_rebalance_enabled and (diagnostic_fail_streak >= auto_rebalance_fail_streak):
                    old_ratio = float(phase_balance_target_ratio)
                    phase_balance_target_ratio = float(
                        min(auto_rebalance_max_target_ratio, old_ratio * auto_rebalance_growth)
                    )
                    rebalance_up_applied = 1 if phase_balance_target_ratio > old_ratio else 0
                    if rebalance_up_applied:
                        print(
                            f"[auto-rebalance-up] step={step_idx}: fail streak={diagnostic_fail_streak}, "
                            f"phase_balance_target_ratio {old_ratio:.4g} -> {phase_balance_target_ratio:.4g}"
                        )
                    diagnostic_fail_streak = 0

                if auto_rebalance_enabled and (diagnostic_good_streak >= auto_rebalance_good_streak):
                    old_ratio = float(phase_balance_target_ratio)
                    phase_balance_target_ratio = float(
                        max(auto_rebalance_min_target_ratio, old_ratio * auto_rebalance_decay)
                    )
                    rebalance_down_applied = 1 if phase_balance_target_ratio < old_ratio else 0
                    if rebalance_down_applied:
                        print(
                            f"[auto-rebalance-down] step={step_idx}: good streak={diagnostic_good_streak}, "
                            f"phase_balance_target_ratio {old_ratio:.4g} -> {phase_balance_target_ratio:.4g}"
                        )
                    diagnostic_good_streak = 0

                diag_row["diagnostic_fail_streak"] = int(diagnostic_fail_streak)
                diag_row["diagnostic_good_streak"] = int(diagnostic_good_streak)
                diag_row["auto_rebalance_applied"] = int(rebalance_up_applied or rebalance_down_applied)
                diag_row["auto_rebalance_up"] = int(rebalance_up_applied)
                diag_row["auto_rebalance_down"] = int(rebalance_down_applied)
                diag_row["phase_balance_target_ratio"] = float(phase_balance_target_ratio)

                diagnostics_rows.append(diag_row)
                _write_dict_rows(
                    file_name=diagnostics_file,
                    fieldnames=list(diagnostics_rows[-1].keys()),
                    rows=diagnostics_rows,
                )

            torch.save(
                {
                    "net_state_dict": field_comp.net.state_dict(),
                    "HI_elem_prev": HI_elem_prev.detach().cpu(),
                    "HII_elem_prev": HII_elem_prev.detach().cpu(),
                    "step": int(step_idx),
                },
                trainedModel_path / Path(f"trained_unified_{step_idx:04d}.pt"),
            )

            with open(loss_path / Path(f"loss_step_{step_idx:04d}.csv"), "w", newline="") as file:
                writer_loss = csv.writer(file)
                writer_loss.writerow(list(loss_per_step_rows[-1].keys()))
                writer_loss.writerow(list(loss_per_step_rows[-1].values()))

            T_prev = T_curr.detach()
            ux_prev = ux_curr.detach()
            uy_prev = uy_curr.detach()
            d_prev = d_curr.detach()
            HI_prev_nodes = HI_curr.detach()
            HII_prev_nodes = HII_curr.detach()
            He_prev_nodes = He_out.detach()

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


# -----------------------------------------------------------------------------
# Stateful orchestrated TM-phase training is the default main path.
# Keep the legacy implementation above for audit/backward reference only.
# -----------------------------------------------------------------------------
from model_train_stateful import train_tm as _train_tm_stateful

train_tm = _train_tm_stateful
