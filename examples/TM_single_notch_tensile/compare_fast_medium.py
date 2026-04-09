from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _read_named_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if np.ndim(data) == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def _scalar(data, key, default=np.nan):
    if key in data.dtype.names:
        return float(data[key][-1])
    return float(default)


def _maxv(data, key, default=np.nan):
    if key in data.dtype.names:
        return float(np.max(data[key]))
    return float(default)


def _minv(data, key, default=np.nan):
    if key in data.dtype.names:
        return float(np.min(data[key]))
    return float(default)


def main():
    root = Path(__file__).parents[0]
    fast = root / Path("run_full_once_fast/results")
    medium = root / Path("run_full_once_medium/results")

    cmp_dir = root / Path("compare_fast_medium")
    cmp_dir.mkdir(parents=True, exist_ok=True)

    curve_fast = _read_named_csv(fast / Path("curves/reaction_displacement_macro_stress_strain.csv"))
    curve_med = _read_named_csv(medium / Path("curves/reaction_displacement_macro_stress_strain.csv"))
    loss_fast = _read_named_csv(fast / Path("losses/loss_per_step.csv"))
    loss_med = _read_named_csv(medium / Path("losses/loss_per_step.csv"))

    # Reaction-displacement
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(curve_fast["uy_top"], curve_fast["reaction_force"], label="fast", linewidth=1.2)
    ax.plot(curve_med["uy_top"], curve_med["reaction_force"], label="medium", linewidth=1.2)
    ax.set_xlabel("Top displacement uy (m)")
    ax.set_ylabel("Reaction force (N)")
    ax.set_title("Reaction-Displacement: Fast vs Medium")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(cmp_dir / Path("compare_reaction_displacement.png"), dpi=300)
    plt.close(fig)

    # Macro stress-strain
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(curve_fast["macro_strain"], curve_fast["macro_stress"], label="fast", linewidth=1.2)
    ax.plot(curve_med["macro_strain"], curve_med["macro_stress"], label="medium", linewidth=1.2)
    ax.set_xlabel("Macro strain")
    ax.set_ylabel("Macro stress (Pa)")
    ax.set_title("Macro Stress-Strain: Fast vs Medium")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(cmp_dir / Path("compare_macro_stress_strain.png"), dpi=300)
    plt.close(fig)

    # Total loss per step
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    if "step" in loss_fast.dtype.names and "total_loss" in loss_fast.dtype.names:
        ax.plot(loss_fast["step"], np.abs(loss_fast["total_loss"]) + 1e-16, label="fast", linewidth=1.2)
    if "step" in loss_med.dtype.names and "total_loss" in loss_med.dtype.names:
        ax.plot(loss_med["step"], np.abs(loss_med["total_loss"]) + 1e-16, label="medium", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Time step")
    ax.set_ylabel("|total_loss| (log)")
    ax.set_title("Total Loss Per Step: Fast vs Medium")
    ax.legend(loc="best")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(cmp_dir / Path("compare_total_loss_log.png"), dpi=300)
    plt.close(fig)

    # Key scalar summary
    rows = []
    for tag, cdat, ldat in (("fast", curve_fast, loss_fast), ("medium", curve_med, loss_med)):
        rows.append(
            {
                "run": tag,
                "n_steps_curve": int(len(cdat)),
                "n_steps_loss": int(len(ldat)),
                "final_uy_top": _scalar(cdat, "uy_top"),
                "final_reaction_force": _scalar(cdat, "reaction_force"),
                "final_macro_strain": _scalar(cdat, "macro_strain"),
                "final_macro_stress": _scalar(cdat, "macro_stress"),
                "min_reaction_force": _minv(cdat, "reaction_force"),
                "max_reaction_force": _maxv(cdat, "reaction_force"),
                "min_macro_stress": _minv(cdat, "macro_stress"),
                "max_macro_stress": _maxv(cdat, "macro_stress"),
                "final_total_loss": _scalar(ldat, "total_loss"),
                "final_mech_loss": _scalar(ldat, "mech_loss"),
                "final_thermal_loss": _scalar(ldat, "thermal_loss"),
                "final_phase_loss": _scalar(ldat, "phase_loss"),
                "final_max_d": _scalar(ldat, "max_d"),
                "final_max_HI": _scalar(ldat, "max_HI"),
                "final_max_HII": _scalar(ldat, "max_HII"),
            }
        )

    keys = list(rows[0].keys())
    cmp_csv = cmp_dir / Path("compare_summary.csv")
    with open(cmp_csv, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    print("COMPARE_DONE", cmp_dir)


if __name__ == "__main__":
    main()

