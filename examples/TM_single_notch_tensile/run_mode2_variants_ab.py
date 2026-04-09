from runtime_env import configure_runtime_env

configure_runtime_env()

import copy
import csv
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

_cli_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from config import *
sys.argv = _cli_argv

PATH_SOURCE = Path(__file__).parents[2]
sys.path.insert(0, str(PATH_SOURCE / Path("source")))

from construct_model import construct_tm_model
from field_computation_tm import MonolithicTMPhaseFieldComputation
from model_train import train_tm
from postprocess_tm import postprocess_tm


def _run_variant(
    variant_name,
    mode_ii_variant,
    w_mode2_norm,
    w_mode2_shear,
    t_end,
    run_root,
    resume_if_available=False,
    skip_if_exists=False,
):
    tm_cfg = copy.deepcopy(tm_model_dict)
    tr_cfg = copy.deepcopy(training_dict)
    opt_cfg = copy.deepcopy(optimizer_dict)
    t_cfg = copy.deepcopy(time_dict)

    tm_cfg["mode_ii_variant"] = mode_ii_variant
    tm_cfg["w_mode2_norm"] = float(w_mode2_norm)
    tm_cfg["w_mode2_shear"] = float(w_mode2_shear)

    tr_cfg["resume_if_available"] = bool(resume_if_available)
    tr_cfg["auto_physics_loop"] = False
    tr_cfg["precheck_before_training"] = False
    tr_cfg["stop_after_precheck"] = False

    # Keep strict semantics from current config without implicit auto-relax loop.
    t_cfg["t_end"] = float(t_end)

    thermo_model, thermal_prop, monolithic_net = construct_tm_model(
        tm_model_dict=tm_cfg,
        thermal_prop_dict=thermal_prop_dict,
        network_dict=network_dict,
        domain_extrema=domain_extrema,
        device=device,
    )

    field_comp = MonolithicTMPhaseFieldComputation(
        net=monolithic_net,
        domain_extrema=domain_extrema.to(device),
        time=torch.tensor(t_cfg["t_start"], device=device),
        uy_rate=uy_rate.to(device),
        T_shift=torch.tensor(thermal_prop_dict["T0"], device=device),
        T_scale=max(abs(thermal_prop_dict["T0"] - thermal_prop_dict["Tref"]), 1.0),
        bottom_fix_mode=tr_cfg.get("bottom_fix_mode", "uy_only"),
        enforce_nodal_clamp=tr_cfg.get("enforce_nodal_clamp", False),
        ux_roller_mode=tr_cfg.get("ux_roller_mode", "free_gauge"),
    )

    run_dir = run_root / Path(f"run_mode2_{variant_name}_step{int(round(t_end / t_cfg['dt'])):03d}")
    run_dir.mkdir(parents=True, exist_ok=True)
    trained = run_dir / Path("best_models")
    trained.mkdir(parents=True, exist_ok=True)
    results = run_dir / Path("results")
    results.mkdir(parents=True, exist_ok=True)

    done_flag = results / "losses" / "loss_per_step.csv"
    if skip_if_exists and done_flag.exists():
        print(f"[mode2-ab] skip existing run: {run_dir}")
        return run_dir

    inp, T_conn, area_T, bc_dict = train_tm(
        field_comp=field_comp,
        thermo_model=thermo_model,
        thermal_prop=thermal_prop,
        crack_dict=crack_dict,
        numr_dict=numr_dict,
        optimizer_dict=opt_cfg,
        training_dict=tr_cfg,
        time_dict=t_cfg,
        mesh_file=mesh_file,
        device=device,
        trainedModel_path=trained,
        results_path=results,
        writer=None,
        boundary_tag_dict=boundary_tag_dict,
    )

    step_idx = int(round((t_cfg["t_end"] - t_cfg["t_start"]) / t_cfg["dt"]))
    postprocess_tm(results_path=results, inp=inp, T_conn=T_conn, step_idx=step_idx, dpi=260, bc_dict=bc_dict)

    with open(run_dir / "mode2_variant_settings.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant_name",
                "mode_ii_variant",
                "w_mode2_norm",
                "w_mode2_shear",
                "t_end",
                "dt",
            ],
        )
        writer.writeheader()
        writer.writerow(
            dict(
                variant_name=variant_name,
                mode_ii_variant=mode_ii_variant,
                w_mode2_norm=float(w_mode2_norm),
                w_mode2_shear=float(w_mode2_shear),
                t_end=float(t_cfg["t_end"]),
                dt=float(t_cfg["dt"]),
            )
        )

    return run_dir


def _safe_read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _collect_key_rows(run_dir, key_steps=(1, 10, 20, 64, 100)):
    loss_dir = run_dir / "results" / "losses"
    loss_step = _safe_read_csv(loss_dir / "loss_per_step.csv")
    diag = _safe_read_csv(loss_dir / "diagnostics_window_step1_10_20_64.csv")
    mode2 = _safe_read_csv(loss_dir / "mode2_component_timeline.csv")
    ratio = _safe_read_csv(loss_dir / "hii_hi_ratio_timeline.csv")
    region = _safe_read_csv(loss_dir / "mode2_region_stats_step1_10_20_64_100.csv")

    out_rows = []
    for s in key_steps:
        row = {"step": int(s)}

        if not loss_step.empty and ("step" in loss_step.columns):
            ls = loss_step[loss_step["step"] == s]
            if not ls.empty:
                ls = ls.iloc[-1]
                for k in ["total_loss", "loss_T", "loss_u", "loss_d", "max_d", "max_HI", "max_HII", "max_He"]:
                    if k in ls.index:
                        row[k] = float(ls[k])

        if not diag.empty and ("step" in diag.columns):
            dg = diag[diag["step"] == s]
            if not dg.empty:
                dg = dg.iloc[-1]
                for k in ["R_exy", "R_psiII", "R_HII", "R_He", "R_d"]:
                    if k in dg.index:
                        row[k] = float(dg[k])

        if not ratio.empty and ("step" in ratio.columns):
            rr = ratio[ratio["step"] == s]
            if not rr.empty:
                rr = rr.iloc[-1]
                for k in ["max_HII_over_HI", "max_psiII_over_psiI", "max_HI", "max_HII", "max_psi_I", "max_psi_II"]:
                    if k in rr.index:
                        row[k] = float(rr[k])

        if not mode2.empty and ("step" in mode2.columns):
            mm = mode2[mode2["step"] == s]
            if not mm.empty:
                mm = mm.iloc[-1]
                for k in [
                    "ep2_norm_mean",
                    "ep2_shear_mean",
                    "ep2_total_used_mean",
                    "psi_II_norm_part_mean",
                    "psi_II_shear_part_mean",
                    "psi_II_total_mean",
                    "psi_I_mean",
                ]:
                    if k in mm.index:
                        row[k] = float(mm[k])

        # Region ratios from region table (bottom_near_fix / notch_tip)
        if (not region.empty) and ("step" in region.columns):
            reg = region[region["step"] == s]
            for var in ["ep2_norm", "ep2_shear", "psi_I", "psi_II", "HI", "HII", "He", "d"]:
                rv = reg[reg["variable"] == var]
                if rv.empty:
                    continue
                notch = rv[rv["region"] == "notch_tip"]["mean"]
                bottom = rv[rv["region"] == "bottom_near_fix"]["mean"]
                if (not notch.empty) and (not bottom.empty):
                    denom = float(notch.iloc[-1])
                    numer = float(bottom.iloc[-1])
                    row[f"R_{var}_bottom_over_notch"] = numer / (denom + 1e-16)

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="all", choices=["all", "legacy", "shear_only", "weighted_split"])
    parser.add_argument("--t_end", type=float, default=0.20)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    run_root = Path(__file__).parents[0]
    t_end_short = float(args.t_end)
    variants = [
        ("legacy", "legacy", 1.0, 1.0),
        ("shear_only", "shear_only", 0.0, 1.0),
        ("weighted_split", "weighted_split", 0.1, 1.0),
    ]
    if args.variant != "all":
        variants = [v for v in variants if v[0] == args.variant]

    run_dirs = []
    for name, variant, w_norm, w_shear in variants:
        print(f"[mode2-ab] running {name}: variant={variant}, w_norm={w_norm}, w_shear={w_shear}")
        run_dir = _run_variant(
            variant_name=name,
            mode_ii_variant=variant,
            w_mode2_norm=w_norm,
            w_mode2_shear=w_shear,
            t_end=t_end_short,
            run_root=run_root,
            resume_if_available=False,
            skip_if_exists=bool(args.skip_existing),
        )
        run_dirs.append((name, run_dir))
        print(f"[mode2-ab] done: {run_dir}")

    summary_rows = []
    summary_frames = []
    for name, run_dir in run_dirs:
        df = _collect_key_rows(run_dir=run_dir)
        if df.empty:
            continue
        df.insert(0, "variant_name", name)
        summary_frames.append(df)

        last = df[df["step"] == 20]
        if not last.empty:
            last = last.iloc[-1]
            summary_rows.append(
                {
                    "variant_name": name,
                    "run_dir": str(run_dir),
                    "step": 20,
                    "max_HII_over_HI": float(last.get("max_HII_over_HI", np.nan)),
                    "max_psiII_over_psiI": float(last.get("max_psiII_over_psiI", np.nan)),
                    "R_psiII": float(last.get("R_psiII", np.nan)),
                    "R_HII": float(last.get("R_HII", np.nan)),
                    "R_He": float(last.get("R_He", np.nan)),
                    "R_d": float(last.get("R_d", np.nan)),
                    "R_ep2_norm_bottom_over_notch": float(last.get("R_ep2_norm_bottom_over_notch", np.nan)),
                    "R_ep2_shear_bottom_over_notch": float(last.get("R_ep2_shear_bottom_over_notch", np.nan)),
                }
            )

    out_dir = run_root / "run_mode2_ab_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    if summary_frames:
        pd.concat(summary_frames, ignore_index=True).to_csv(out_dir / "mode2_ab_key_steps_summary.csv", index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "mode2_ab_step20_compact.csv", index=False)

    with open(out_dir / "mode2_ab_run_dirs.txt", "w", encoding="utf-8") as f:
        for name, run_dir in run_dirs:
            f.write(f"{name},{run_dir}\n")

    print(f"[mode2-ab] summary saved: {out_dir}")


if __name__ == "__main__":
    main()
