from runtime_env import configure_runtime_env

configure_runtime_env()

import csv

from config import *

PATH_SOURCE = Path(__file__).parents[2]
sys.path.insert(0, str(PATH_SOURCE / Path("source")))

from construct_model import construct_tm_model
from field_computation_tm import MonolithicTMPhaseFieldComputation
from model_train import train_tm
from postprocess_tm import postprocess_tm
from visualize_geometry import visualize_geometry


def _append_timeline_row(results_path, row):
    timeline_file = results_path / Path("losses/physical_correctness_timeline.csv")
    timeline_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not timeline_file.exists()) or (timeline_file.stat().st_size == 0)
    with open(timeline_file, "a", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer_csv.writeheader()
        writer_csv.writerow(row)


def _read_physical_report(report_file):
    out = {}
    if not report_file.exists():
        return out
    with open(report_file, "r", encoding="utf-8") as f:
        for line in f:
            if "=" not in line:
                continue
            k, v = line.strip().split("=", 1)
            out[k] = v
    return out


def _run_once_with_current_config():
    try:
        inp_, T_conn_, area_T_, bc_dict_, run_meta_ = train_tm(
            field_comp=field_comp,
            thermo_model=thermo_model,
            thermal_prop=thermal_prop,
            crack_dict=crack_dict,
            numr_dict=numr_dict,
            optimizer_dict=optimizer_dict,
            training_dict=training_dict,
            time_dict=time_dict,
            mesh_file=mesh_file,
            device=device,
            trainedModel_path=trainedModel_path,
            results_path=results_path,
            writer=writer,
            boundary_tag_dict=boundary_tag_dict,
            return_run_meta=True,
        )
    except RuntimeError as e:
        print(f"[auto-physics] run failed: {e}")
        return None, None, None, None, False, str(e), {}

    postprocess_tm(
        results_path=results_path,
        inp=inp_,
        T_conn=T_conn_,
        step_idx=-1,
        dpi=400,
        bc_dict=bc_dict_,
    )

    return inp_, T_conn_, area_T_, bc_dict_, True, "", run_meta_


# run as:
# python .\main.py hidden_layers neurons seed activation init_coeff


thermo_model, thermal_prop, monolithic_net = construct_tm_model(
    tm_model_dict=tm_model_dict,
    thermal_prop_dict=thermal_prop_dict,
    network_dict=network_dict,
    domain_extrema=domain_extrema,
    device=device,
)

field_comp = MonolithicTMPhaseFieldComputation(
    net=monolithic_net,
    domain_extrema=domain_extrema.to(device),
    time=torch.tensor(time_dict["t_start"], device=device),
    uy_rate=uy_rate.to(device),
    T_shift=torch.tensor(thermal_prop_dict["T0"], device=device),
    T_scale=max(abs(thermal_prop_dict["T0"] - thermal_prop_dict["Tref"]), 1.0),
    bottom_fix_mode=training_dict.get("bottom_fix_mode", "uy_only"),
    enforce_nodal_clamp=training_dict.get("enforce_nodal_clamp", False),
)


if __name__ == "__main__":
    if training_dict.get("precheck_before_training", True):
        visualize_geometry(
            output_dir=results_path / Path("precheck"),
            time_value=time_dict["t_end"],
            title_suffix=" (main run)",
        )
        if training_dict.get("stop_after_precheck", False):
            print("Precheck-only mode: training skipped.")
            raise SystemExit(0)

    auto_loop = bool(training_dict.get("auto_physics_loop", True))
    max_rounds = int(training_dict.get("auto_physics_max_rounds", 3))

    inp = T_conn = area_T = bc_dict = None
    for round_idx in range(1, max_rounds + 1):
        print(f"[auto-physics] round {round_idx}/{max_rounds} start")
        inp, T_conn, area_T, bc_dict, run_ok, fail_msg, run_meta = _run_once_with_current_config()

        report = _read_physical_report(results_path / Path("losses/physical_correctness_report.txt"))
        train_executed = bool(run_meta.get("train_executed", False))
        steps_before = int(run_meta.get("steps_before", 0))
        steps_after = int(run_meta.get("steps_after", steps_before))
        new_steps_generated = int(run_meta.get("new_steps_generated", 0))
        eligible_for_escalation = bool(
            run_meta.get("eligible_for_escalation", train_executed and (new_steps_generated > 0))
        )
        target_final_step = int(run_meta.get("target_final_step", -1))
        already_complete = (target_final_step >= 0) and (steps_after >= target_final_step)
        passed = run_ok and (int(report.get("overall_pass", "0")) == 1)

        _append_timeline_row(
            results_path,
            {
                "round": round_idx,
                "stage": "strict",
                "run_ok": int(run_ok),
                "passed": int(passed),
                "fail_msg": str(fail_msg),
                "require_converged_step": int(bool(training_dict.get("require_converged_step", True))),
                "max_inner_iters": int(training_dict.get("max_inner_iters", 6)),
                "tol_loss": float(training_dict.get("tol_loss", 2e-1)),
                "tol_hist": float(training_dict.get("tol_hist", 5e-2)),
                "phase_balance_target_ratio": float(training_dict.get("phase_balance_target_ratio", 0.2)),
                "phase_rprop": int(optimizer_dict.get("n_epochs_RPROP_phase", 120)),
                "w_irrev": float(training_dict.get("w_irrev", 1.0)),
                "train_executed": int(train_executed),
                "steps_before": int(steps_before),
                "steps_after": int(steps_after),
                "new_steps_generated": int(new_steps_generated),
                "eligible_for_escalation": int(eligible_for_escalation),
            },
        )

        if passed:
            print(f"[auto-physics] pass at round {round_idx}")
            break

        if (not eligible_for_escalation) or already_complete:
            if already_complete:
                print(
                    "[auto-physics] target steps already completed; "
                    "no training executed in this round, escalation disabled."
                )
            else:
                print(
                    "[auto-physics] no new trained steps generated in this round; "
                    "escalation disabled."
                )
            break

        if (not run_ok) and ("convergence criteria not met" in str(fail_msg)):
            # Relaxed stage: loosen convergence and retry in the same round.
            old_req = bool(training_dict.get("require_converged_step", True))
            old_inner = int(training_dict.get("max_inner_iters", 6))
            old_tol_loss = float(training_dict.get("tol_loss", 2e-1))
            old_tol_hist = float(training_dict.get("tol_hist", 5e-2))
            old_rprop_th = int(optimizer_dict.get("n_epochs_RPROP_thermal", 30))
            old_rprop_me = int(optimizer_dict.get("n_epochs_RPROP_mech", 60))
            old_rprop_ph = int(optimizer_dict.get("n_epochs_RPROP_phase", 80))
            training_dict["require_converged_step"] = False
            training_dict["max_inner_iters"] = min(20, old_inner + 4)
            training_dict["tol_loss"] = old_tol_loss * 2.2
            training_dict["tol_hist"] = old_tol_hist * 2.2
            optimizer_dict["n_epochs_RPROP_thermal"] = min(80, max(old_rprop_th + 4, round(old_rprop_th * 1.2)))
            optimizer_dict["n_epochs_RPROP_mech"] = min(160, max(old_rprop_me + 8, round(old_rprop_me * 1.2)))
            optimizer_dict["n_epochs_RPROP_phase"] = min(220, max(old_rprop_ph + 10, round(old_rprop_ph * 1.2)))
            print(
                "[auto-physics] relaxed retry: "
                f"require_converged_step {old_req}->{training_dict['require_converged_step']}, "
                f"max_inner_iters {old_inner}->{training_dict['max_inner_iters']}, "
                f"tol_loss {old_tol_loss:.4g}->{training_dict['tol_loss']:.4g}, "
                f"tol_hist {old_tol_hist:.4g}->{training_dict['tol_hist']:.4g}"
            )

            inp, T_conn, area_T, bc_dict, run_ok_relaxed, fail_msg_relaxed, run_meta_relaxed = _run_once_with_current_config()
            report_relaxed = _read_physical_report(results_path / Path("losses/physical_correctness_report.txt"))
            train_executed_relaxed = bool(run_meta_relaxed.get("train_executed", False))
            steps_before_relaxed = int(run_meta_relaxed.get("steps_before", 0))
            steps_after_relaxed = int(run_meta_relaxed.get("steps_after", steps_before_relaxed))
            new_steps_relaxed = int(run_meta_relaxed.get("new_steps_generated", 0))
            eligible_relaxed = bool(
                run_meta_relaxed.get(
                    "eligible_for_escalation",
                    train_executed_relaxed and (new_steps_relaxed > 0),
                )
            )
            passed_relaxed = run_ok_relaxed and (int(report_relaxed.get("overall_pass", "0")) == 1)

            _append_timeline_row(
                results_path,
                {
                    "round": round_idx,
                    "stage": "relaxed",
                    "run_ok": int(run_ok_relaxed),
                    "passed": int(passed_relaxed),
                    "fail_msg": str(fail_msg_relaxed),
                    "require_converged_step": int(bool(training_dict.get("require_converged_step", True))),
                    "max_inner_iters": int(training_dict.get("max_inner_iters", 6)),
                    "tol_loss": float(training_dict.get("tol_loss", 2e-1)),
                    "tol_hist": float(training_dict.get("tol_hist", 5e-2)),
                    "phase_balance_target_ratio": float(training_dict.get("phase_balance_target_ratio", 0.2)),
                    "phase_rprop": int(optimizer_dict.get("n_epochs_RPROP_phase", 120)),
                    "w_irrev": float(training_dict.get("w_irrev", 1.0)),
                    "train_executed": int(train_executed_relaxed),
                    "steps_before": int(steps_before_relaxed),
                    "steps_after": int(steps_after_relaxed),
                    "new_steps_generated": int(new_steps_relaxed),
                    "eligible_for_escalation": int(eligible_relaxed),
                },
            )

            if passed_relaxed:
                print(f"[auto-physics] pass at round {round_idx} (relaxed)")
                break

            if not eligible_relaxed:
                print(
                    "[auto-physics] relaxed retry produced no new trained steps; "
                    "escalation disabled."
                )
                break

        if not auto_loop:
            print("[auto-physics] disabled; stop after first run")
            break

        # Escalation for next round (physics-driven, no path hard-coding)
        old_inner = int(training_dict.get("max_inner_iters", 6))
        old_tol_loss = float(training_dict.get("tol_loss", 2e-1))
        old_tol_hist = float(training_dict.get("tol_hist", 5e-2))
        training_dict["max_inner_iters"] = min(18, old_inner + 2)
        training_dict["tol_loss"] = old_tol_loss * 1.5
        training_dict["tol_hist"] = old_tol_hist * 1.5

        old_ratio = float(training_dict.get("phase_balance_target_ratio", 0.2))
        new_ratio = min(
            float(training_dict.get("auto_phase_ratio_cap", 6.0)),
            old_ratio * float(training_dict.get("auto_phase_ratio_growth", 1.4)),
        )
        training_dict["phase_balance_target_ratio"] = new_ratio

        old_rprop = int(optimizer_dict.get("n_epochs_RPROP_phase", 120))
        new_rprop = int(
            min(
                int(training_dict.get("auto_phase_rprop_cap", 300)),
                max(old_rprop + 1, round(old_rprop * float(training_dict.get("auto_phase_rprop_growth", 1.25)))),
            )
        )
        optimizer_dict["n_epochs_RPROP_phase"] = new_rprop

        old_wir = float(training_dict.get("w_irrev", 1.0))
        new_wir = min(
            float(training_dict.get("auto_irrev_cap", 5.0)),
            old_wir * float(training_dict.get("auto_irrev_growth", 1.2)),
        )
        training_dict["w_irrev"] = new_wir

        print(
            "[auto-physics] escalate-next-round: "
            f"max_inner_iters {old_inner}->{training_dict['max_inner_iters']}, "
            f"tol_loss {old_tol_loss:.4g}->{training_dict['tol_loss']:.4g}, "
            f"tol_hist {old_tol_hist:.4g}->{training_dict['tol_hist']:.4g}, "
            f"phase_balance_target_ratio {old_ratio:.4g}->{new_ratio:.4g}, "
            f"phase_rprop {old_rprop}->{new_rprop}, "
            f"w_irrev {old_wir:.4g}->{new_wir:.4g}"
        )

    else:
        print("[auto-physics] max rounds reached without pass")
