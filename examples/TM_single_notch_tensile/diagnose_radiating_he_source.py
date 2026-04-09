from runtime_env import configure_runtime_env

configure_runtime_env()

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import torch

_cli_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from config import (
    tm_model_dict,
    crack_dict,
    numr_dict,
    mesh_file,
    boundary_tag_dict,
    thermal_prop_dict,
)
sys.argv = _cli_argv

PATH_SOURCE = Path(__file__).parents[2]
sys.path.insert(0, str(PATH_SOURCE / Path("source")))

from thermo_mech_model import ThermoMechModel, nodal_to_element, element_to_nodal
from input_data_from_mesh import prep_input_data_tm
from compute_energy import field_grads


def _read_field_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    out = {k: np.asarray(data[k], dtype=np.float64) for k in data.dtype.names}
    return out


def _as_idx(x):
    if x is None:
        return np.array([], dtype=np.int64)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.int64).reshape(-1)
    return np.asarray(x, dtype=np.int64).reshape(-1)


def _region_masks_nodes(inp, bc_dict):
    x = inp[:, 0]
    y = inp[:, 1]
    Lx = max(float(np.max(x) - np.min(x)), 1e-16)
    Ly = max(float(np.max(y) - np.min(y)), 1e-16)

    bottom_nodes = _as_idx(bc_dict.get("mechanical_bottom_nodes", bc_dict.get("bottom_nodes", None)))
    notch_nodes = _as_idx(bc_dict.get("notch_face_nodes", None))
    fixed_nodes = _as_idx(bc_dict.get("fixed_point_nodes", bc_dict.get("point1_node", None)))
    top_nodes = _as_idx(bc_dict.get("mechanical_top_nodes", bc_dict.get("top_nodes", None)))

    if bottom_nodes.size > 2:
        xb = np.sort(x[bottom_nodes])
        dx = np.diff(xb)
        dx = dx[dx > 0]
        dx_ref = float(np.median(dx)) if dx.size > 0 else 0.0
    else:
        dx_ref = 0.0

    x_tip = float(np.max(x[notch_nodes])) if notch_nodes.size > 0 else float(np.median(x))
    y_tip = float(np.median(y[notch_nodes])) if notch_nodes.size > 0 else float(np.median(y))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    x_fix = float(np.mean(x[fixed_nodes])) if fixed_nodes.size > 0 else float(np.min(x))
    x_mid = 0.5 * (float(np.min(x)) + float(np.max(x)))

    notch_rx = max(0.03 * Lx, 3.0 * dx_ref)
    notch_ry = max(0.02 * Ly, 2.0 * dx_ref)
    bottom_rx = max(0.03 * Lx, 3.0 * dx_ref)
    bottom_ry = max(0.02 * Ly, 2.0 * dx_ref)
    mid_rx = max(0.12 * Lx, 6.0 * dx_ref)
    top_ry = max(0.10 * Ly, 3.0 * dx_ref)

    notch_tip = (np.abs(x - x_tip) <= notch_rx) & (np.abs(y - y_tip) <= notch_ry)
    bottom_near_fix = (np.abs(x - x_fix) <= bottom_rx) & (np.abs(y - y_min) <= bottom_ry)
    bottom_band = np.abs(y - y_min) <= bottom_ry
    bottom_mid = bottom_band & (np.abs(x - x_mid) <= mid_rx) & (~bottom_near_fix)
    top_band = np.abs(y - y_max) <= top_ry
    left_edge_mid = (np.abs(x - np.min(x)) <= bottom_rx) & (np.abs(y - 0.5 * (y_min + y_max)) <= 0.15 * Ly)
    bottom_band_minus_near_fix = bottom_band & (~bottom_near_fix)

    masks = {
        "notch_tip": notch_tip,
        "bottom_near_fix": bottom_near_fix,
        "bottom_mid": bottom_mid,
        "top_band": top_band,
        "left_edge_mid": left_edge_mid,
        "bottom_band_all": bottom_band,
        "bottom_band_minus_near_fix": bottom_band_minus_near_fix,
    }
    meta = {
        "x_tip": x_tip,
        "y_tip": y_tip,
        "x_fix": x_fix,
        "y_min": y_min,
        "y_max": y_max,
    }
    return masks, meta


def _to_elem_mask(node_mask, T_conn):
    tri = T_conn
    return np.any(node_mask[tri], axis=1)


def _label_region(xc, yc, masks_elem):
    for name in ["notch_tip", "bottom_near_fix", "bottom_mid", "top_band", "left_edge_mid"]:
        m = masks_elem[name]
        if np.any(m):
            # closest masked element center determines label
            return name
    return "other"


def _region_stats(values_elem, masks_elem):
    rows = []
    for name, mask in masks_elem.items():
        if not np.any(mask):
            rows.append(
                dict(
                    region=name,
                    count=0,
                    mean=np.nan,
                    std=np.nan,
                    p95=np.nan,
                    p99=np.nan,
                    max=np.nan,
                )
            )
            continue
        v = values_elem[mask]
        rows.append(
            dict(
                region=name,
                count=int(v.size),
                mean=float(np.mean(v)),
                std=float(np.std(v)),
                p95=float(np.percentile(v, 95)),
                p99=float(np.percentile(v, 99)),
                max=float(np.max(v)),
            )
        )
    return rows


def _anisotropy_index(values_elem_abs, xc, yc, n_sector=36, eps=1e-16):
    dx = xc - float(np.median(xc))
    dy = yc - float(np.median(yc))
    peak_idx = int(np.argmax(values_elem_abs))
    x0, y0 = float(xc[peak_idx]), float(yc[peak_idx])
    ang = np.arctan2(yc - y0, xc - x0)
    bins = np.linspace(-np.pi, np.pi, n_sector + 1)
    sec_means = []
    for i in range(n_sector):
        m = (ang >= bins[i]) & (ang < bins[i + 1])
        if np.any(m):
            sec_means.append(float(np.mean(values_elem_abs[m])))
    if len(sec_means) < 3:
        return np.nan, x0, y0
    sec_means = np.asarray(sec_means)
    idx = float(np.std(sec_means) / (np.mean(np.abs(sec_means)) + eps))
    return idx, x0, y0


def _plot_field_pair(step, name, node_val, elem_val, x, y, tri, notch_meta, out_dir):
    if name in ("du_x_dy", "du_y_dx", "exy", "ed", "eta", "epxy"):
        vmax = float(np.max(np.abs(np.concatenate([node_val, elem_val]))))
        vmin = -vmax
    else:
        vmin = float(np.min(np.concatenate([node_val, elem_val])))
        vmax = float(np.max(np.concatenate([node_val, elem_val])))
    if vmax <= vmin:
        vmax = vmin + 1e-16

    fig, ax = plt.subplots(figsize=(5, 4.2))
    tpc = ax.tripcolor(tri, node_val, shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.plot([0.0, notch_meta["x_tip"]], [notch_meta["y_tip"], notch_meta["y_tip"]], "w-", lw=2.0)
    ax.set_title(f"{name} node (step {step})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(tpc, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_node_step{step:04d}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    tpc = ax.tripcolor(tri, facecolors=elem_val, shading="flat", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.plot([0.0, notch_meta["x_tip"]], [notch_meta["y_tip"], notch_meta["y_tip"]], "w-", lw=2.0)
    ax.set_title(f"{name} elem const (step {step})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(tpc, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_elem_step{step:04d}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    tpc0 = axes[0].tripcolor(tri, node_val, shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].plot([0.0, notch_meta["x_tip"]], [notch_meta["y_tip"], notch_meta["y_tip"]], "w-", lw=2.0)
    axes[0].set_title(f"{name} node")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    tpc1 = axes[1].tripcolor(tri, facecolors=elem_val, shading="flat", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].plot([0.0, notch_meta["x_tip"]], [notch_meta["y_tip"], notch_meta["y_tip"]], "w-", lw=2.0)
    axes[1].set_title(f"{name} elem const")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    fig.colorbar(tpc1, ax=axes.ravel().tolist(), shrink=0.85)
    fig.suptitle(f"compare node/elem: {name} (step {step})")
    fig.tight_layout()
    fig.savefig(out_dir / f"compare_node_elem_{name}_step{step:04d}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        default="examples/TM_single_notch_tensile/run_short20_scale_autow",
    )
    parser.add_argument("--steps", type=str, default="1,10,20")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    field_dir = run_dir / "results" / "field_data"
    out_dir = run_dir / "results" / "losses" / "radiating_source_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    tm_model = ThermoMechModel(
        E0=torch.tensor(tm_model_dict["E0"]),
        v0=torch.tensor(tm_model_dict["v0"]),
        GcI=torch.tensor(tm_model_dict["GcI"]),
        GcII=torch.tensor(tm_model_dict["GcII"]),
        kappa=torch.tensor(tm_model_dict["kappa"]),
        l0=torch.tensor(tm_model_dict["l0"]),
        etaPF=torch.tensor(tm_model_dict["etaPF"]),
        eps_r=torch.tensor(tm_model_dict["eps_r"]),
    )
    inp, T_conn, area_T, bc_dict, _ = prep_input_data_tm(
        tm_model=tm_model,
        crack_dict=crack_dict,
        mesh_file=mesh_file,
        device=device,
        length_scale=numr_dict.get("length_scale", 1.0),
        boundary_tag_dict=boundary_tag_dict,
    )
    inp_np = inp.detach().cpu().numpy()
    tri_np = T_conn.detach().cpu().numpy()
    area_np = area_T.detach().cpu().numpy()
    x = inp_np[:, 0]
    y = inp_np[:, 1]
    tri = mtri.Triangulation(x, y, tri_np)

    node_masks, meta = _region_masks_nodes(inp_np, bc_dict)
    elem_masks = {k: _to_elem_mask(v, tri_np) for k, v in node_masks.items()}
    xc = np.mean(x[tri_np], axis=1)
    yc = np.mean(y[tri_np], axis=1)

    steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
    peak_rows = []
    stats_rows = []
    ani_rows = []

    for step in steps:
        fcsv = field_dir / f"field_step_{step:04d}.csv"
        if not fcsv.exists():
            continue
        f = _read_field_csv(fcsv)
        T = torch.tensor(f["T"], dtype=torch.float64)
        ux = torch.tensor(f["ux"], dtype=torch.float64)
        uy = torch.tensor(f["uy"], dtype=torch.float64)
        HI = torch.tensor(f["HI"], dtype=torch.float64)
        HII = torch.tensor(f["HII"], dtype=torch.float64)
        He = torch.tensor(f["He"], dtype=torch.float64)

        area_t = torch.tensor(area_np, dtype=torch.float64)
        conn_t = torch.tensor(tri_np, dtype=torch.long)
        inp_t = torch.tensor(inp_np, dtype=torch.float64)
        alpha = torch.tensor(thermal_prop_dict["alpha"], dtype=torch.float64)
        Tref = torch.tensor(thermal_prop_dict["Tref"], dtype=torch.float64)

        dux_dx, dux_dy = field_grads(inp_t, ux, area_t, conn_t)
        duy_dx, duy_dy = field_grads(inp_t, uy, area_t, conn_t)

        exx_e, eyy_e, exy_e, ezz_e = tm_model.kinematics(
            inp=inp_t,
            ux=ux,
            uy=uy,
            T=T,
            alpha=alpha,
            Tref=Tref,
            area_elem=area_t,
            T_conn=conn_t,
        )
        mixed = tm_model.mixed_mode_terms(exx_e, eyy_e, exy_e, ezz_e)

        variable_elem = {
            "du_x_dy": dux_dy.detach().cpu().numpy(),
            "du_y_dx": duy_dx.detach().cpu().numpy(),
            "exy": exy_e.detach().cpu().numpy(),
            "ed": mixed["ed"].detach().cpu().numpy(),
            "eta": mixed["eta"].detach().cpu().numpy(),
            "epxy": mixed["epxy"].detach().cpu().numpy(),
            "ep2": mixed["ep2"].detach().cpu().numpy(),
            "psi_II": mixed["psi_II"].detach().cpu().numpy(),
            "HII": nodal_to_element(HII, conn_t).detach().cpu().numpy(),
            "He": nodal_to_element(He, conn_t).detach().cpu().numpy(),
        }
        area_t32 = area_t.to(dtype=torch.float32)
        variable_node = {
            k: element_to_nodal(
                torch.tensor(v, dtype=torch.float32),
                conn_t,
                inp_np.shape[0],
                area_elem=area_t32,
            )
            .detach()
            .cpu()
            .numpy()
            for k, v in variable_elem.items()
        }

        for name in variable_elem.keys():
            node_v = variable_node[name]
            elem_v = variable_elem[name]
            _plot_field_pair(step, name, node_v, elem_v, x, y, tri, meta, out_dir)

            # peaks
            i_peak = int(np.argmax(np.abs(elem_v)))
            xpk = float(xc[i_peak])
            ypk = float(yc[i_peak])
            reg = "other"
            for rname, rmask in elem_masks.items():
                if rname in ("bottom_band_all", "bottom_band_minus_near_fix"):
                    continue
                if rmask[i_peak]:
                    reg = rname
                    break
            peak_rows.append(
                {
                    "step": step,
                    "variable": name,
                    "max_abs_value": float(np.max(np.abs(elem_v))),
                    "x_peak": xpk,
                    "y_peak": ypk,
                    "region_label": reg,
                }
            )

            # region stats
            for row in _region_stats(elem_v, elem_masks):
                row_out = {"step": step, "variable": name}
                row_out.update(row)
                stats_rows.append(row_out)

            # anisotropy
            ani, x0, y0 = _anisotropy_index(np.abs(elem_v), xc, yc, n_sector=36)
            ani_rows.append(
                {
                    "step": step,
                    "variable": name,
                    "radial_anisotropy_index": float(ani),
                    "source_x_peak_abs": float(x0),
                    "source_y_peak_abs": float(y0),
                }
            )

    if len(peak_rows) > 0:
        pd.DataFrame(peak_rows).to_csv(out_dir / "radiating_source_peak_locations.csv", index=False)
    if len(stats_rows) > 0:
        pd.DataFrame(stats_rows).to_csv(out_dir / "radiating_source_region_stats.csv", index=False)
    if len(ani_rows) > 0:
        pd.DataFrame(ani_rows).to_csv(out_dir / "radiating_source_anisotropy.csv", index=False)

    print(f"diagnostics_done: {out_dir}")


if __name__ == "__main__":
    main()
