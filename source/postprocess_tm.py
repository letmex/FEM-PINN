import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path


def _read_field_csv(field_csv):
    data = np.genfromtxt(field_csv, delimiter=",", names=True)
    return data


def _read_named_csv(csv_file):
    if not Path(csv_file).exists():
        return None
    data = np.genfromtxt(csv_file, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data is None or np.size(data) == 0:
        return None
    if np.ndim(data) == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def _plot_scalar_field_node(ax, triang, val, title):
    ax.set_aspect("equal")
    tpc = ax.tripcolor(triang, val, shading="gouraud", rasterized=True)
    cbar = plt.colorbar(tpc, ax=ax)
    cbar.formatter.set_powerlimits((0, 0))
    ax.set_title(title)


def _plot_scalar_field_elem(ax, triang, val_elem, title):
    ax.set_aspect("equal")
    tpc = ax.tripcolor(triang, facecolors=val_elem, shading="flat", rasterized=True)
    cbar = plt.colorbar(tpc, ax=ax)
    cbar.formatter.set_powerlimits((0, 0))
    ax.set_title(title)


def _get_col(data, key, default=0.0):
    if data is None:
        return None
    if key in data.dtype.names:
        return np.asarray(data[key], dtype=float)
    n = len(data)
    return np.full((n,), float(default), dtype=float)


def _plot_physics_consistency_report(results_path, fig_path, physics_file, dpi):
    data = _read_named_csv(physics_file)
    if data is None or "step" not in data.dtype.names:
        return

    step = np.asarray(data["step"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for key, label in [
        ("T_bc_l1", "T_bc_l1"),
        ("uy_top_l1", "uy_top_l1"),
        ("uy_bottom_l1", "uy_bottom_l1"),
    ]:
        if key in data.dtype.names:
            ax.semilogy(step, np.abs(np.asarray(data[key], dtype=float)) + 1e-16, linewidth=1.2, label=label)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Boundary residual (log)")
    ax.set_title("Boundary Consistency Audit")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="best")
    plt.savefig(fig_path / Path("physics_boundary_audit.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    summary = {}
    for key in ["T_bc_l1", "uy_top_l1", "uy_bottom_l1", "stress_degrade_corr", "max_d_drop", "max_HI_drop", "max_HII_drop"]:
        if key in data.dtype.names:
            vec = np.asarray(data[key], dtype=float)
            summary[f"{key}_mean"] = float(np.nanmean(vec))
            summary[f"{key}_max"] = float(np.nanmax(vec))

    if "softening_started" in data.dtype.names:
        s = np.asarray(data["softening_started"], dtype=int)
        idx = np.where(s > 0)[0]
        summary["softening_first_step"] = int(data["step"][idx[0]]) if idx.size > 0 else -1

    summary_file = Path(results_path) / Path("losses/physics_consistency_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}={v}\n")


def _write_physical_pass_fail_report(results_path, diagnostics_file, physics_file, crack_summary_file):
    diag = _read_named_csv(diagnostics_file)
    phys = _read_named_csv(physics_file)

    # Default fail-safe values.
    initiated_at_notch = 0
    front_monotonic_breaks = 10**9
    mean_abs_path_dev = np.inf
    if Path(crack_summary_file).exists():
        with open(crack_summary_file, "r", encoding="utf-8") as f:
            for line in f:
                if "=" not in line:
                    continue
                k, v = line.strip().split("=", 1)
                if k == "initiated_at_notch":
                    initiated_at_notch = int(float(v))
                elif k == "front_monotonic_breaks":
                    front_monotonic_breaks = int(float(v))
                elif k == "mean_abs_path_dev":
                    mean_abs_path_dev = float(v)

    r_he_ok = False
    if diag is not None and ("R_He" in diag.dtype.names):
        r_he = np.asarray(diag["R_He"], dtype=float)
        if r_he.size > 0:
            r_he_ok = bool(np.nanmean(r_he) < 1.0)

    irrev_ok = False
    bc_ok = False
    if phys is not None:
        if ("max_d_drop" in phys.dtype.names) and ("max_HI_drop" in phys.dtype.names) and ("max_HII_drop" in phys.dtype.names):
            d_drop = np.nanmax(np.asarray(phys["max_d_drop"], dtype=float))
            hi_drop = np.nanmax(np.asarray(phys["max_HI_drop"], dtype=float))
            hii_drop = np.nanmax(np.asarray(phys["max_HII_drop"], dtype=float))
            irrev_ok = (d_drop <= 1e-10) and (hi_drop <= 1e-10) and (hii_drop <= 1e-10)
        if ("T_bc_l1" in phys.dtype.names) and ("uy_top_l1" in phys.dtype.names) and ("uy_bottom_l1" in phys.dtype.names):
            tbc = float(np.nanmean(np.asarray(phys["T_bc_l1"], dtype=float)))
            uyt = float(np.nanmean(np.asarray(phys["uy_top_l1"], dtype=float)))
            uyb = float(np.nanmean(np.asarray(phys["uy_bottom_l1"], dtype=float)))
            bc_ok = (tbc <= 1e-4) and (uyt <= 1e-9) and (uyb <= 1e-9)

    path_ok = (initiated_at_notch == 1) and (front_monotonic_breaks <= 2) and np.isfinite(mean_abs_path_dev)
    overall_ok = bool(path_ok and r_he_ok and irrev_ok and bc_ok)

    out_txt = Path(results_path) / Path("losses/physical_correctness_report.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"overall_pass={int(overall_ok)}\n")
        f.write(f"path_ok={int(path_ok)}\n")
        f.write(f"r_he_ok={int(r_he_ok)}\n")
        f.write(f"irreversibility_ok={int(irrev_ok)}\n")
        f.write(f"boundary_ok={int(bc_ok)}\n")
        f.write(f"initiated_at_notch={initiated_at_notch}\n")
        f.write(f"front_monotonic_breaks={front_monotonic_breaks}\n")
        f.write(f"mean_abs_path_dev={mean_abs_path_dev}\n")


def _plot_diagnostic_rebalance_curves(fig_path, diagnostics_file, dpi):
    diag = _read_named_csv(diagnostics_file)
    if diag is None:
        return
    if "step" not in diag.dtype.names:
        return
    if "phase_balance_target_ratio" not in diag.dtype.names:
        return

    step = np.asarray(diag["step"], dtype=float)
    ratio = np.asarray(diag["phase_balance_target_ratio"], dtype=float)

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(step, ratio, "-o", linewidth=1.2, markersize=3.0, label="target_ratio")
    ax.set_xlabel("Time step")
    ax.set_ylabel("phase_balance_target_ratio")
    ax.set_title("Auto-Rebalance Target Ratio")
    ax.grid(alpha=0.25)

    if "auto_rebalance_up" in diag.dtype.names:
        up = np.asarray(diag["auto_rebalance_up"], dtype=int)
        idx_up = np.where(up > 0)[0]
        if idx_up.size > 0:
            ax.scatter(step[idx_up], ratio[idx_up], c="tab:red", s=24, label="rebalance up")

    if "auto_rebalance_down" in diag.dtype.names:
        dn = np.asarray(diag["auto_rebalance_down"], dtype=int)
        idx_dn = np.where(dn > 0)[0]
        if idx_dn.size > 0:
            ax.scatter(step[idx_dn], ratio[idx_dn], c="tab:green", s=24, label="rebalance down")

    ax.legend(loc="best")
    plt.savefig(fig_path / Path("target_ratio_vs_step.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_dict_rows(csv_file, fieldnames, rows):
    import csv

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _extract_notch_tip_from_bc(x, y, bc_dict):
    if bc_dict is None:
        return None, None
    nodes = _as_np_index(bc_dict.get("notch_face_nodes", None))
    if nodes.size == 0:
        return None, None
    return float(np.max(x[nodes])), float(np.median(y[nodes]))


def _evaluate_crack_path_metrics(results_path, fig_path, bc_dict, dpi=300):
    field_path = Path(results_path) / Path("field_data")
    field_files = sorted(field_path.glob("field_step_*.csv"))
    if len(field_files) == 0:
        return

    data0 = _read_field_csv(field_files[0])
    x = np.asarray(data0["x"], dtype=float)
    y = np.asarray(data0["y"], dtype=float)
    x_tip0, y_ref = _extract_notch_tip_from_bc(x, y, bc_dict)
    if x_tip0 is None:
        return

    d_th = 0.5
    front_rows = []
    initiated_step = None
    initiated_at_notch = 0
    tip_x_prev = x_tip0

    for field_csv in field_files:
        data = _read_field_csv(field_csv)
        d = np.asarray(data["d"], dtype=float)
        step = int(Path(field_csv).stem.split("_")[-1])

        mask = d >= d_th
        if np.any(mask):
            x_front = float(np.max(x[mask]))
            i_tip = int(np.argmax(np.where(mask, x, -np.inf)))
            y_front = float(y[i_tip])
            path_dev = abs(y_front - y_ref)
            mono_break = 1 if x_front + 1e-14 < tip_x_prev else 0
            tip_x_prev = max(tip_x_prev, x_front)

            if initiated_step is None:
                initiated_step = step
                Lx = max(np.max(x) - np.min(x), 1e-16)
                Ly = max(np.max(y) - np.min(y), 1e-16)
                rx = 0.03 * Lx
                ry = 0.03 * Ly
                initiated_at_notch = int((abs(x_front - x_tip0) <= rx) and (abs(y_front - y_ref) <= ry))
        else:
            x_front = np.nan
            y_front = np.nan
            path_dev = np.nan
            mono_break = 0

        front_rows.append(
            {
                "step": step,
                "d_threshold": d_th,
                "x_tip_ref": float(x_tip0),
                "y_ref": float(y_ref),
                "x_front": float(x_front) if np.isfinite(x_front) else np.nan,
                "y_front": float(y_front) if np.isfinite(y_front) else np.nan,
                "path_dev_abs": float(path_dev) if np.isfinite(path_dev) else np.nan,
                "front_monotonic_break": int(mono_break),
            }
        )

    eval_csv = Path(results_path) / Path("losses/crack_path_eval.csv")
    _save_dict_rows(eval_csv, list(front_rows[-1].keys()), front_rows)

    valid = np.asarray([r["path_dev_abs"] for r in front_rows], dtype=float)
    valid = valid[np.isfinite(valid)]
    mean_dev = float(np.mean(valid)) if valid.size > 0 else np.nan
    max_dev = float(np.max(valid)) if valid.size > 0 else np.nan
    n_breaks = int(np.sum([r["front_monotonic_break"] for r in front_rows]))

    summary_file = Path(results_path) / Path("losses/crack_path_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"initiated_step={initiated_step}\n")
        f.write(f"initiated_at_notch={initiated_at_notch}\n")
        f.write(f"mean_abs_path_dev={mean_dev}\n")
        f.write(f"max_abs_path_dev={max_dev}\n")
        f.write(f"front_monotonic_breaks={n_breaks}\n")

    step_vec = np.asarray([r["step"] for r in front_rows], dtype=float)
    x_front = np.asarray([r["x_front"] for r in front_rows], dtype=float)
    y_front = np.asarray([r["y_front"] for r in front_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    ax.plot(step_vec, x_front, "-o", linewidth=1.1, markersize=2.8, label="x_front")
    ax.axhline(x_tip0, color="k", linestyle="--", linewidth=0.9, label="x_tip_ref")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Crack front x (m)")
    ax.set_title("Crack Front Advance")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    plt.savefig(fig_path / Path("crack_front_x_vs_step.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    ax.plot(step_vec, y_front, "-o", linewidth=1.1, markersize=2.8, label="y_front")
    ax.axhline(y_ref, color="k", linestyle="--", linewidth=0.9, label="y_ref")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Crack front y (m)")
    ax.set_title("Crack Front Path vs Reference Midline")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    plt.savefig(fig_path / Path("crack_front_y_vs_step.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_loss_curves(fig_path, loss_per_step_file, loss_trace_file, dpi):
    loss_step = _read_named_csv(loss_per_step_file)
    if loss_step is not None:
        step = loss_step["step"] if "step" in loss_step.dtype.names else np.arange(len(loss_step))
        curve_keys = [
            (("thermal_loss", "loss_T"), "Thermal"),
            (("mech_loss", "loss_u"), "Mechanical"),
            (("phase_loss", "loss_d"), "Phase-field"),
            (("total_loss", "loss_total"), "Total"),
        ]

        def _pick_key(candidates):
            for k in candidates:
                if k in loss_step.dtype.names:
                    return k
            return None

        fig, ax = plt.subplots(figsize=(5, 3.4))
        for key_candidates, label in curve_keys:
            key = _pick_key(key_candidates)
            if key is not None:
                ax.plot(step, loss_step[key], label=label, linewidth=1.3)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Per Step")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        plt.savefig(fig_path / Path("loss_per_step_linear.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 3.4))
        for key_candidates, label in curve_keys:
            key = _pick_key(key_candidates)
            if key is not None:
                y = np.abs(loss_step[key]) + 1e-16
                ax.semilogy(step, y, label=label, linewidth=1.3)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title("Loss Per Step (Log)")
        ax.legend(loc="best")
        ax.grid(alpha=0.25, which="both")
        plt.savefig(fig_path / Path("loss_per_step_log.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        if "adaptive_passes" in loss_step.dtype.names:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(step, loss_step["adaptive_passes"], "-o", markersize=2.5, linewidth=1.0)
            ax.set_xlabel("Time step")
            ax.set_ylabel("Adaptive passes")
            ax.set_title("Adaptive Retraining Passes")
            ax.grid(alpha=0.25)
            plt.savefig(fig_path / Path("adaptive_passes.png"), dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        phase_domain_key = "E_phase_domain" if "E_phase_domain" in loss_step.dtype.names else ("E_d" if "E_d" in loss_step.dtype.names else None)
        if "E_el" in loss_step.dtype.names and phase_domain_key is not None:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(step, loss_step["E_el"], label="E_el", linewidth=1.2)
            ax.plot(step, loss_step[phase_domain_key], label="E_phase_domain", linewidth=1.2)
            ax.set_xlabel("Time step")
            ax.set_ylabel("Energy")
            ax.set_title("Elastic and Damage Energies")
            ax.legend(loc="best")
            ax.grid(alpha=0.25)
            plt.savefig(fig_path / Path("energy_el_damage.png"), dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        if (
            "E_crack_density" in loss_step.dtype.names
            and "E_reaction" in loss_step.dtype.names
            and "E_viscosity" in loss_step.dtype.names
            and phase_domain_key is not None
        ):
            fig, ax = plt.subplots(figsize=(5.6, 3.4))
            ax.plot(step, loss_step["E_crack_density"], label="E_crack_density", linewidth=1.2)
            ax.plot(step, loss_step["E_reaction"], label="E_reaction", linewidth=1.2)
            ax.plot(step, loss_step["E_viscosity"], label="E_viscosity", linewidth=1.2)
            ax.plot(step, loss_step[phase_domain_key], label="E_phase_domain", linewidth=1.4, color="k")
            ax.set_xlabel("Time step")
            ax.set_ylabel("Energy")
            ax.set_title("Phase Energy Components")
            ax.legend(loc="best")
            ax.grid(alpha=0.25)
            plt.savefig(fig_path / Path("phase_energy_components.png"), dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        if "max_d" in loss_step.dtype.names and "max_HI" in loss_step.dtype.names and "max_HII" in loss_step.dtype.names:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(step, loss_step["max_d"], label="max(d)", linewidth=1.2)
            ax.plot(step, loss_step["max_HI"], label="max(HI)", linewidth=1.2)
            ax.plot(step, loss_step["max_HII"], label="max(HII)", linewidth=1.2)
            ax.set_xlabel("Time step")
            ax.set_ylabel("Value")
            ax.set_title("Damage and History Maxima")
            ax.legend(loc="best")
            ax.grid(alpha=0.25)
            plt.savefig(fig_path / Path("history_damage_max.png"), dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        # Loss normalization by step-1 baseline for trend comparison.
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for key_candidates, label in curve_keys:
            key = _pick_key(key_candidates)
            if key is not None:
                y = np.asarray(loss_step[key], dtype=float)
                y0 = np.abs(y[0]) + 1e-16
                ax.plot(step, y / y0, label=label, linewidth=1.3)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Normalized loss (L/L_step1)")
        ax.set_title("Normalized Loss Per Step")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        plt.savefig(fig_path / Path("loss_per_step_normalized.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # Physics-vs-constraint decomposition.
        # If detailed weighted terms are present, use exact split.
        # Otherwise fallback to a proxy split from the available legacy columns.
        if "physics_loss" in loss_step.dtype.names and "constraint_loss" in loss_step.dtype.names:
            physics = np.asarray(loss_step["physics_loss"], dtype=float)
            constraint = np.asarray(loss_step["constraint_loss"], dtype=float)
            split_title = "Loss Decomposition (Exact)"
        else:
            thermal = _get_col(loss_step, "thermal_loss", default=0.0)
            eel = _get_col(loss_step, "E_el", default=0.0)
            if "E_phase_domain" in loss_step.dtype.names:
                ed = _get_col(loss_step, "E_phase_domain", default=0.0)
            else:
                ed = _get_col(loss_step, "E_d", default=0.0)
            boundary = _get_col(loss_step, "boundary_loss", default=0.0)
            irrev = _get_col(loss_step, "irreversibility_loss", default=0.0)
            physics = thermal + eel + ed
            constraint = boundary + irrev
            split_title = "Loss Decomposition (Proxy)"

        if "total_loss" in loss_step.dtype.names:
            total = _get_col(loss_step, "total_loss", default=0.0)
        else:
            total = _get_col(loss_step, "loss_total", default=0.0)
        residual_gap = total - (physics + constraint)

        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(step, total, label="Total", linewidth=1.6, color="tab:red")
        ax.plot(step, physics, label="Physics", linewidth=1.3, color="tab:blue")
        ax.plot(step, constraint, label="Constraint", linewidth=1.3, color="tab:green")
        ax.plot(step, residual_gap, label="Gap", linewidth=1.1, color="tab:gray", linestyle="--")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Loss")
        ax.set_title(split_title)
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        plt.savefig(fig_path / Path("loss_physics_constraint_split.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    loss_trace = _read_named_csv(loss_trace_file)
    if loss_trace is not None and "branch" in loss_trace.dtype.names:
        branches = np.unique(loss_trace["branch"])
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for branch in branches:
            mask = loss_trace["branch"] == branch
            if np.any(mask):
                x = loss_trace["iter"][mask] if "iter" in loss_trace.dtype.names else np.arange(np.sum(mask))
                y = np.abs(loss_trace["loss"][mask]) + 1e-16
                ax.semilogy(x, y, linewidth=1.0, label=str(branch))
        ax.set_xlabel("Optimizer iteration")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title("Loss Trace by Branch")
        ax.legend(loc="best")
        ax.grid(alpha=0.25, which="both")
        plt.savefig(fig_path / Path("loss_trace_by_branch.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # Step-wise first->last reduction indicator for each branch.
        # reduction = (first - last) / |first|; >0 means improved in-step.
        step_key = "step" if "step" in loss_trace.dtype.names else None
        iter_key = "iter" if "iter" in loss_trace.dtype.names else None
        if step_key is not None and iter_key is not None:
            branch_reduction = {}
            branch_ratio = {}
            for branch in branches:
                mask_b = loss_trace["branch"] == branch
                if not np.any(mask_b):
                    continue
                d_b = loss_trace[mask_b]
                steps_b = np.unique(d_b[step_key])
                red_list = []
                ratio_list = []
                step_list = []
                for s in steps_b:
                    mask_s = d_b[step_key] == s
                    d_bs = d_b[mask_s]
                    if d_bs.size == 0:
                        continue
                    order = np.argsort(d_bs[iter_key])
                    vals = np.asarray(d_bs["loss"][order], dtype=float)
                    first = vals[0]
                    last = vals[-1]
                    reduction = (first - last) / (np.abs(first) + 1e-16)
                    ratio = last / (np.abs(first) + 1e-16)
                    red_list.append(reduction)
                    ratio_list.append(ratio)
                    step_list.append(s)
                if len(step_list) > 0:
                    branch_reduction[str(branch)] = (np.asarray(step_list), np.asarray(red_list))
                    branch_ratio[str(branch)] = (np.asarray(step_list), np.asarray(ratio_list))

            if len(branch_reduction) > 0:
                fig, ax = plt.subplots(figsize=(5.6, 3.6))
                for branch, (svec, rvec) in branch_reduction.items():
                    ax.plot(svec, rvec, linewidth=1.1, label=branch)
                ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--")
                ax.set_xlabel("Time step")
                ax.set_ylabel("First->Last Reduction")
                ax.set_title("Step-wise Reduction by Branch")
                ax.legend(loc="best")
                ax.grid(alpha=0.25)
                plt.savefig(fig_path / Path("stepwise_first_last_reduction.png"), dpi=dpi, bbox_inches="tight")
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(5.6, 3.6))
                for branch, (svec, qvec) in branch_ratio.items():
                    ax.plot(svec, qvec, linewidth=1.1, label=branch)
                ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--")
                ax.set_xlabel("Time step")
                ax.set_ylabel("Last / First")
                ax.set_title("Step-wise Last/First Ratio")
                ax.legend(loc="best")
                ax.grid(alpha=0.25)
                plt.savefig(fig_path / Path("stepwise_last_first_ratio.png"), dpi=dpi, bbox_inches="tight")
                plt.close(fig)


def _as_np_index(arr):
    if arr is None:
        return np.empty((0,), dtype=np.int64)
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=np.int64).reshape(-1)


def _as_np_edges(arr):
    if arr is None:
        return np.empty((0, 2), dtype=np.int64)
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.int64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return arr.reshape(-1, 2)


def _infer_notch_segment(x, y, bc_dict):
    if bc_dict is None:
        return None
    edges = _as_np_edges(bc_dict.get("notch_face_edges", None))
    if edges.size > 0:
        xs = np.concatenate([x[edges[:, 0]], x[edges[:, 1]]])
        ys = np.concatenate([y[edges[:, 0]], y[edges[:, 1]]])
        return {
            "x_min": float(np.min(xs)),
            "x_tip": float(np.max(xs)),
            "y_tip": float(np.median(ys)),
            "n_edge": int(edges.shape[0]),
        }

    nodes = _as_np_index(bc_dict.get("notch_face_nodes", None))
    if nodes.size > 1:
        xs = x[nodes]
        ys = y[nodes]
        return {
            "x_min": float(np.min(xs)),
            "x_tip": float(np.max(xs)),
            "y_tip": float(np.median(ys)),
            "n_edge": 0,
        }
    return None


def _cross_notch_triangle_mask(x, y, triangles, notch_seg):
    if notch_seg is None or triangles.size == 0:
        return np.zeros((triangles.shape[0],), dtype=bool)
    x_min = notch_seg["x_min"]
    x_tip = notch_seg["x_tip"]
    y_tip = notch_seg["y_tip"]
    Lx = max(float(np.max(x) - np.min(x)), 1.0)
    Ly = max(float(np.max(y) - np.min(y)), 1.0)
    tol_x = 1e-6 * Lx
    tol_y = 1e-4 * Ly

    tx = x[triangles]
    ty = y[triangles]
    tri_xmin = np.min(tx, axis=1)
    tri_xmax = np.max(tx, axis=1)
    tri_ymin = np.min(ty, axis=1)
    tri_ymax = np.max(ty, axis=1)

    return (
        (tri_xmax <= x_tip + tol_x)
        & (tri_xmin >= x_min - tol_x)
        & (tri_ymin < y_tip - tol_y)
        & (tri_ymax > y_tip + tol_y)
    )


def _build_plot_triangulations(x, y, tri_conn, bc_dict):
    tri_conn = np.asarray(tri_conn, dtype=np.int64)
    tri_true = mtri.Triangulation(x, y, triangles=tri_conn)
    tri_auto = mtri.Triangulation(x, y)
    notch_seg = _infer_notch_segment(x, y, bc_dict)

    true_cross = _cross_notch_triangle_mask(x, y, tri_true.triangles, notch_seg)
    auto_cross = _cross_notch_triangle_mask(x, y, tri_auto.triangles, notch_seg)

    tri_true_masked = mtri.Triangulation(x, y, triangles=tri_conn)
    if np.any(true_cross):
        tri_true_masked.set_mask(true_cross)

    diag = {
        "notch_segment": notch_seg,
        "n_tri_true": int(tri_true.triangles.shape[0]),
        "n_tri_auto": int(tri_auto.triangles.shape[0]),
        "n_cross_true_before_mask": int(np.sum(true_cross)),
        "n_cross_auto": int(np.sum(auto_cross)),
        "n_cross_true_after_mask": 0,
    }
    return tri_auto, tri_true, tri_true_masked, diag


def _elem_from_nodal(val_node, tri):
    return np.mean(val_node[tri], axis=1)


def _plot_old_new_compare(fig_path, key, title, tri_auto, tri_true_masked, val_node, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    _plot_scalar_field_node(axes[0], tri_auto, val_node, f"{title} (old auto triang)")
    _plot_scalar_field_node(axes[1], tri_true_masked, val_node, f"{title} (new true T_conn)")
    fig.tight_layout()
    plt.savefig(fig_path / Path(f"compare_old_new_{key}.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_node_elem_compare(fig_path, key, title, tri_true_masked, val_node, val_elem, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    _plot_scalar_field_node(axes[0], tri_true_masked, val_node, f"{title} node (true T_conn)")
    _plot_scalar_field_elem(axes[1], tri_true_masked, val_elem, f"{title} elem const (true T_conn)")
    fig.tight_layout()
    plt.savefig(fig_path / Path(f"compare_node_elem_{key}.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_boundary_groups(fig_path, x, y, tri, bc_dict, dpi):
    top = _as_np_index(bc_dict.get("mechanical_top_nodes", bc_dict.get("top_nodes", None)))
    bottom = _as_np_index(bc_dict.get("mechanical_bottom_nodes", bc_dict.get("bottom_nodes", None)))
    thermal = _as_np_index(bc_dict.get("thermal_dirichlet_nodes", None))
    notch = _as_np_index(bc_dict.get("notch_face_nodes", None))
    fixed = _as_np_index(bc_dict.get("fixed_point_nodes", bc_dict.get("point1_node", None)))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_aspect("equal")
    ax.triplot(x, y, tri, color="0.85", linewidth=0.25)
    if top.size > 0:
        ax.scatter(x[top], y[top], s=5, c="tab:red", label="mech top")
    if bottom.size > 0:
        ax.scatter(x[bottom], y[bottom], s=5, c="tab:blue", label="mech bottom")
    if thermal.size > 0:
        ax.scatter(x[thermal], y[thermal], s=4, c="tab:orange", alpha=0.45, label="thermal Dirichlet")
    if notch.size > 0:
        ax.scatter(x[notch], y[notch], s=5, c="tab:purple", label="notch faces")
    if fixed.size > 0:
        ax.scatter(x[fixed], y[fixed], s=55, c="black", marker="*", label="fixed point")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Boundary Groups Used in Training")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    plt.savefig(fig_path / Path("boundary_groups.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def postprocess_tm(results_path, inp, T_conn, step_idx=-1, dpi=300, bc_dict=None):
    results_path = Path(results_path)
    field_path = results_path / Path("field_data")
    curve_file = results_path / Path("curves/reaction_displacement_macro_stress_strain.csv")
    loss_per_step_file = results_path / Path("losses/loss_per_step.csv")
    loss_trace_file = results_path / Path("losses/loss_trace.csv")
    diagnostics_file = results_path / Path("losses/diagnostics_window_step1_10_20_64.csv")
    physics_file = results_path / Path("losses/physics_consistency_per_step.csv")
    fig_path = results_path / Path("figures")
    fig_path.mkdir(parents=True, exist_ok=True)

    field_files = sorted(field_path.glob("field_step_*.csv"))
    if len(field_files) == 0:
        raise FileNotFoundError(f"No field data found in {field_path}")

    if step_idx < 0:
        field_csv = field_files[-1]
    else:
        field_csv = field_path / Path(f"field_step_{step_idx:04d}.csv")
        if not field_csv.exists():
            raise FileNotFoundError(f"No field data file: {field_csv}")

    data = _read_field_csv(field_csv)
    x = data["x"]
    y = data["y"]
    tri = T_conn.detach().cpu().numpy()
    tri_auto, tri_true, tri_true_masked, tri_diag = _build_plot_triangulations(x=x, y=y, tri_conn=tri, bc_dict=bc_dict)

    # Save triangulation diagnostics for anti-artifact auditing.
    diag_file = fig_path / Path("triangulation_diagnostics.txt")
    with open(diag_file, "w", encoding="utf-8") as f:
        f.write(f"n_tri_true={tri_diag['n_tri_true']}\n")
        f.write(f"n_tri_auto={tri_diag['n_tri_auto']}\n")
        f.write(f"n_cross_true_before_mask={tri_diag['n_cross_true_before_mask']}\n")
        f.write(f"n_cross_true_after_mask={tri_diag['n_cross_true_after_mask']}\n")
        f.write(f"n_cross_auto={tri_diag['n_cross_auto']}\n")
        f.write(f"notch_segment={tri_diag['notch_segment']}\n")

    figs = [
        ("T", "Temperature T (K)"),
        ("ux", "Displacement ux (m)"),
        ("uy", "Displacement uy (m)"),
        ("d", "Phase Field d"),
        ("HI", "History HI (Pa)"),
        ("HII", "History HII (Pa)"),
        ("He", "Driving Force He (Pa)"),
    ]

    for key, title in figs:
        fig, ax = plt.subplots(figsize=(4, 3))
        _plot_scalar_field_node(ax, tri_true_masked, data[key], title)
        plt.savefig(fig_path / Path(f"{key}.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # Element constant maps (no interpolation / no smoothing).
    HI_elem = _elem_from_nodal(data["HI"], tri)
    HII_elem = _elem_from_nodal(data["HII"], tri)
    He_elem = _elem_from_nodal(data["He"], tri)

    fig, ax = plt.subplots(figsize=(4, 3))
    _plot_scalar_field_elem(ax, tri_true_masked, HI_elem, "History HI elem (const)")
    plt.savefig(fig_path / Path("HI_elem.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3))
    _plot_scalar_field_elem(ax, tri_true_masked, HII_elem, "History HII elem (const)")
    plt.savefig(fig_path / Path("HII_elem.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3))
    _plot_scalar_field_elem(ax, tri_true_masked, He_elem, "Driving Force He elem (const)")
    plt.savefig(fig_path / Path("He_elem.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Old vs new (auto-triangulation vs true T_conn) comparisons.
    _plot_old_new_compare(fig_path, "HI", "History HI", tri_auto, tri_true_masked, data["HI"], dpi)
    _plot_old_new_compare(fig_path, "HII", "History HII", tri_auto, tri_true_masked, data["HII"], dpi)
    _plot_old_new_compare(fig_path, "He", "Driving Force He", tri_auto, tri_true_masked, data["He"], dpi)

    # Node vs element comparisons on true mesh.
    _plot_node_elem_compare(fig_path, "He", "Driving Force He", tri_true_masked, data["He"], He_elem, dpi)
    _plot_node_elem_compare(fig_path, "HII", "History HII", tri_true_masked, data["HII"], HII_elem, dpi)

    if curve_file.exists():
        curve = np.genfromtxt(curve_file, delimiter=",", names=True)

        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.plot(curve["uy_top"], curve["reaction_force"], "-k")
        ax1.set_xlabel("Top displacement uy (m)")
        ax1.set_ylabel("Reaction force (N)")
        ax1.set_title("Reaction-Displacement")
        plt.savefig(fig_path / Path("reaction_displacement.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(curve["macro_strain"], curve["macro_stress"], "-k")
        ax2.set_xlabel("Macro strain")
        ax2.set_ylabel("Macro stress (Pa)")
        ax2.set_title("Macro Stress-Strain")
        plt.savefig(fig_path / Path("macro_stress_strain.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig2)

    if bc_dict is not None:
        _plot_boundary_groups(fig_path=fig_path, x=x, y=y, tri=tri, bc_dict=bc_dict, dpi=dpi)

    _plot_loss_curves(
        fig_path=fig_path,
        loss_per_step_file=loss_per_step_file,
        loss_trace_file=loss_trace_file,
        dpi=dpi,
    )
    _plot_physics_consistency_report(
        results_path=results_path,
        fig_path=fig_path,
        physics_file=physics_file,
        dpi=dpi,
    )
    _plot_diagnostic_rebalance_curves(
        fig_path=fig_path,
        diagnostics_file=diagnostics_file,
        dpi=dpi,
    )
    _evaluate_crack_path_metrics(
        results_path=results_path,
        fig_path=fig_path,
        bc_dict=bc_dict,
        dpi=dpi,
    )
    _write_physical_pass_fail_report(
        results_path=results_path,
        diagnostics_file=diagnostics_file,
        physics_file=physics_file,
        crack_summary_file=results_path / Path("losses/crack_path_summary.txt"),
    )
