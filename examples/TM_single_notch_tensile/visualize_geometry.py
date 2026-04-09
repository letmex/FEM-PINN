from runtime_env import configure_runtime_env

configure_runtime_env()

from config import *

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

PATH_SOURCE = Path(__file__).parents[2]
sys.path.insert(0, str(PATH_SOURCE / Path("source")))

from construct_model import construct_tm_model
from input_data_from_mesh import prep_input_data_tm


def _to_um(arr):
    return np.asarray(arr) * 1e6


def _draw_notch_void(ax, x, y, notch_nodes):
    """
    Draw the real pre-crack void explicitly from notch-face boundary nodes.
    This avoids the optical illusion from dense triplot edge lines.
    """
    if notch_nodes.size < 4:
        return None

    xn = x[notch_nodes]
    yn = y[notch_nodes]
    y_mid = 0.5 * (np.min(yn) + np.max(yn))
    upper = yn >= y_mid
    lower = yn < y_mid
    if not np.any(upper) or not np.any(lower):
        return None

    y_low = np.max(yn[lower])
    y_high = np.min(yn[upper])
    if y_high <= y_low:
        return None

    x_start = np.min(xn)
    x_tip = np.max(xn)
    rect = Rectangle(
        (_to_um(x_start), _to_um(y_low)),
        _to_um(x_tip - x_start),
        _to_um(y_high - y_low),
        facecolor="white",
        edgecolor="crimson",
        linewidth=1.0,
        zorder=6,
        alpha=0.95,
    )
    ax.add_patch(rect)
    return (x_start, x_tip, y_low, y_high)


def visualize_geometry(output_dir=None, time_value=None, title_suffix=""):
    thermo_model, thermal_prop, _ = construct_tm_model(
        tm_model_dict=tm_model_dict,
        thermal_prop_dict=thermal_prop_dict,
        network_dict=network_dict,
        domain_extrema=domain_extrema,
        device="cpu",
    )

    inp, T_conn, area_T, bc_dict, d0 = prep_input_data_tm(
        tm_model=thermo_model,
        crack_dict=crack_dict,
        mesh_file=mesh_file,
        device="cpu",
        length_scale=numr_dict.get("length_scale", 1.0),
        boundary_tag_dict=boundary_tag_dict,
    )

    x = inp[:, 0].detach().cpu().numpy()
    y = inp[:, 1].detach().cpu().numpy()
    tri = T_conn.detach().cpu().numpy()
    d0_np = d0.detach().cpu().numpy()

    top_nodes = bc_dict["mechanical_top_nodes"].detach().cpu().numpy()
    bottom_nodes = bc_dict["mechanical_bottom_nodes"].detach().cpu().numpy()
    notch_nodes = bc_dict["notch_face_nodes"].detach().cpu().numpy()
    point1_node = bc_dict["fixed_point_nodes"].detach().cpu().numpy()
    thermal_nodes = bc_dict["thermal_dirichlet_nodes"].detach().cpu().numpy()

    if output_dir is None:
        geo_dir = results_path / Path("geometry")
    else:
        geo_dir = Path(output_dir)
    geo_dir.mkdir(parents=True, exist_ok=True)

    t_show = float(time_dict["t_start"] if time_value is None else time_value)
    uy_show = float(uy_rate.item()) * t_show
    bottom_fix_mode = training_dict.get("bottom_fix_mode", "uy_only")

    # 1) Geometry + mesh. Real notch is represented by geometric boundary.
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.set_aspect("equal")
    ax1.triplot(_to_um(x), _to_um(y), tri, color="0.6", linewidth=0.35)
    notch_box = _draw_notch_void(ax1, x, y, notch_nodes)
    if notch_box is not None:
        ax1.text(
            _to_um(0.5 * (notch_box[0] + notch_box[1])),
            _to_um(notch_box[3]) + 0.08,
            "pre-crack void",
            color="crimson",
            fontsize=8,
            ha="center",
            va="bottom",
        )

    ax1.set_xlabel("x (um)")
    ax1.set_ylabel("y (um)")
    ax1.set_title("Geometry and Mesh (Real Pre-crack Boundary)")
    plt.tight_layout()
    fig1.savefig(geo_dir / Path("geometry_mesh_notch.png"), dpi=300)
    plt.close(fig1)

    # 2) Boundary-condition nodes
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.set_aspect("equal")
    ax2.triplot(_to_um(x), _to_um(y), tri, color="0.85", linewidth=0.25)
    ax2.scatter(
        _to_um(x[top_nodes]),
        _to_um(y[top_nodes]),
        s=5,
        c="tab:red",
        label=f"top uy=4e-8 t (t={t_show:.3g}s, uy={uy_show:.3e}m)",
    )
    ax2.scatter(
        _to_um(x[bottom_nodes]),
        _to_um(y[bottom_nodes]),
        s=5,
        c="tab:blue",
        label=f"bottom support ({bottom_fix_mode})",
    )
    if notch_nodes.size > 0:
        ax2.scatter(_to_um(x[notch_nodes]), _to_um(y[notch_nodes]), s=5, c="tab:purple", label="notch faces")
    ax2.scatter(_to_um(x[thermal_nodes]), _to_um(y[thermal_nodes]), s=4, c="tab:orange", alpha=0.5, label="thermal T=T0")
    ax2.scatter(
        _to_um(x[point1_node]),
        _to_um(y[point1_node]),
        s=80,
        marker="*",
        c="black",
        label="fixed point 1",
        zorder=10,
    )
    ax2.set_xlabel("x (um)")
    ax2.set_ylabel("y (um)")
    ax2.set_title(f"Boundary Nodes Used in PINN Ansatz/Loss{title_suffix}")
    ax2.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig2.savefig(geo_dir / Path("geometry_boundary_sets.png"), dpi=300)
    plt.close(fig2)

    # 2c) Dedicated load/constraint precheck figure before training
    fig2c, ax2c = plt.subplots(figsize=(7, 6))
    ax2c.set_aspect("equal")
    ax2c.triplot(_to_um(x), _to_um(y), tri, color="0.85", linewidth=0.25)
    ax2c.scatter(_to_um(x[top_nodes]), _to_um(y[top_nodes]), s=6, c="tab:red", label="top displacement BC")
    ax2c.scatter(_to_um(x[bottom_nodes]), _to_um(y[bottom_nodes]), s=6, c="tab:blue", label="bottom support BC")
    ax2c.scatter(_to_um(x[thermal_nodes]), _to_um(y[thermal_nodes]), s=4, c="tab:orange", alpha=0.5, label="thermal Dirichlet")
    if notch_nodes.size > 0:
        ax2c.scatter(_to_um(x[notch_nodes]), _to_um(y[notch_nodes]), s=6, c="tab:purple", label="notch faces (natural BC)")
    ax2c.scatter(_to_um(x[point1_node]), _to_um(y[point1_node]), s=85, marker="*", c="black", label="fixed point")

    if top_nodes.size > 1 and abs(uy_show) > 0:
        top_x_um = _to_um(x[top_nodes])
        top_y_um = _to_um(y[top_nodes])
        sample = np.linspace(0, top_nodes.size - 1, num=min(12, top_nodes.size), dtype=int)
        xs = top_x_um[sample]
        ys = top_y_um[sample]
        dy_um = np.sign(uy_show) * np.full_like(xs, 0.15)
        ax2c.quiver(
            xs,
            ys,
            np.zeros_like(xs),
            dy_um,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            color="tab:red",
            alpha=0.9,
            zorder=9,
        )

    ax2c.set_xlabel("x (um)")
    ax2c.set_ylabel("y (um)")
    ax2c.set_title(f"Pre-Training BC/Load Check{title_suffix}")
    ax2c.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig2c.savefig(geo_dir / Path("precheck_boundary_loads.png"), dpi=300)
    plt.close(fig2c)

    # 2b) Notch zoom to make the physical gap directly visible
    if notch_box is not None:
        fig2b, ax2b = plt.subplots(figsize=(8, 3.2))
        ax2b.set_aspect("equal")
        ax2b.triplot(_to_um(x), _to_um(y), tri, color="0.6", linewidth=0.2)
        _draw_notch_void(ax2b, x, y, notch_nodes)
        ax2b.scatter(_to_um(x[notch_nodes]), _to_um(y[notch_nodes]), s=5, c="tab:red", label="notch faces")
        ax2b.set_xlim(_to_um(notch_box[0]) - 0.2, _to_um(notch_box[1]) + 0.3)
        ax2b.set_ylim(_to_um(notch_box[2]) - 0.1, _to_um(notch_box[3]) + 0.1)
        ax2b.set_xlabel("x (um)")
        ax2b.set_ylabel("y (um)")
        ax2b.set_title("Pre-crack Zoom (Void Between Two Notch Faces)")
        ax2b.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        fig2b.savefig(geo_dir / Path("geometry_notch_zoom.png"), dpi=300)
        plt.close(fig2b)

    # 3) Initial phase field in material domain (d0 = 0)
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.set_aspect("equal")
    tpc = ax3.tripcolor(_to_um(x), _to_um(y), tri, d0_np, shading="gouraud")
    cbar = fig3.colorbar(tpc, ax=ax3)
    cbar.set_label("initial d")
    ax3.set_xlabel("x (um)")
    ax3.set_ylabel("y (um)")
    ax3.set_title(f"Initial Phase Field (d0 = 0){title_suffix}")
    plt.tight_layout()
    fig3.savefig(geo_dir / Path("geometry_initial_seed.png"), dpi=300)
    plt.close(fig3)

    print("Saved:")
    print(f"boundary source: {bc_dict.get('boundary_source', 'unknown')}")
    print(
        "counts:",
        f"top={top_nodes.size}, bottom={bottom_nodes.size}, left={bc_dict['left_nodes'].numel()},",
        f"right={bc_dict['right_nodes'].numel()}, notch={notch_nodes.size}, thermal={thermal_nodes.size}, fixed_point={point1_node.size}",
    )
    print(f"load formula: uy_top(t)=4e-8*t (m), preview t={t_show:.3g}s => uy_top={uy_show:.3e} m")
    print(f"bottom_fix_mode={bottom_fix_mode}")
    print(geo_dir / Path("geometry_mesh_notch.png"))
    print(geo_dir / Path("geometry_boundary_sets.png"))
    print(geo_dir / Path("precheck_boundary_loads.png"))
    if notch_box is not None:
        print(geo_dir / Path("geometry_notch_zoom.png"))
    print(geo_dir / Path("geometry_initial_seed.png"))


if __name__ == "__main__":
    visualize_geometry()
