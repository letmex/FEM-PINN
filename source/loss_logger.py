import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class LossLogger:
    """
    Utility logger for thermo-mechanical phase-field training.
    Stores:
    - iteration-level loss trace
    - step-level summary metrics
    - per-step loss artifacts (csv/npy/png)
    """

    def __init__(self, loss_dir):
        self.loss_dir = Path(loss_dir)
        self.loss_dir.mkdir(parents=True, exist_ok=True)
        self.iter_file = self.loss_dir / Path("loss_trace.csv")
        self.step_file = self.loss_dir / Path("loss_per_step.csv")

    def init_iteration_trace(self, reset=False):
        if reset or (not self.iter_file.exists()):
            with open(self.iter_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["iter", "step", "inner_iter", "branch", "loss"])

    def append_iteration_rows(self, rows):
        if len(rows) == 0:
            return
        with open(self.iter_file, "a", newline="") as file:
            writer = csv.writer(file)
            for row in rows:
                writer.writerow(
                    [
                        int(row["iter"]),
                        int(row["step"]),
                        int(row.get("inner_iter", 0)),
                        str(row["branch"]),
                        float(row["loss"]),
                    ]
                )

    def write_step_rows(self, rows):
        if len(rows) == 0:
            return
        fieldnames = list(rows[0].keys())
        with open(self.step_file, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        np.save(self.loss_dir / Path("loss_per_step.npy"), np.asarray(rows, dtype=object), allow_pickle=True)

    def save_step_loss_artifacts(self, step_idx, branch_losses):
        """
        branch_losses: dict(branch_name -> list(float))
        """
        # CSV
        step_csv = self.loss_dir / Path(f"loss_step_{step_idx:04d}_iter.csv")
        max_len = max([len(v) for v in branch_losses.values()] + [0])
        branch_names = list(branch_losses.keys())
        with open(step_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["iter"] + branch_names)
            for i in range(max_len):
                row = [i + 1]
                for b in branch_names:
                    vals = branch_losses[b]
                    row.append(vals[i] if i < len(vals) else "")
                writer.writerow(row)

        # NPY
        step_npy = self.loss_dir / Path(f"loss_step_{step_idx:04d}_iter.npy")
        np.save(step_npy, branch_losses, allow_pickle=True)

        # PNG
        step_png = self.loss_dir / Path(f"loss_step_{step_idx:04d}_iter.png")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for b in branch_names:
            vals = np.asarray(branch_losses[b], dtype=float)
            if vals.size == 0:
                continue
            ax.plot(np.arange(1, vals.size + 1), vals, label=b, linewidth=1.0)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(f"Step {step_idx} Loss")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(step_png, dpi=250, bbox_inches="tight")
        plt.close(fig)
