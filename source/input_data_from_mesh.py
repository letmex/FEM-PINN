import torch
import numpy as np
from pathlib import Path
from utils import parse_mesh, hist_alpha_init

# Prepare input data
def prep_input_data(matprop, pffmodel, crack_dict, numr_dict, mesh_file, device):
    '''
    Input data is prepared from the .msh file.
    If gradient_type = numerical:  
        X, Y = nodal coordinates
        T_conn = connectivity
    If gradient_type = autodiff:   
        X, Y = coordinate of the Gauss point in one point Gauss quadrature
        T_conn = None
    area_T: area of elements

    hist_alpha = initial alpha field

    '''
    assert Path(mesh_file).suffix == '.msh', "Mesh file should be a .msh file"
    
    X, Y, T_conn, area_T = parse_mesh(filename = mesh_file, gradient_type=numr_dict["gradient_type"])

    inp = torch.from_numpy(np.column_stack((X, Y))).to(torch.float).to(device)
    T_conn = torch.from_numpy(T_conn).to(torch.long).to(device)
    area_T = torch.from_numpy(area_T).to(torch.float).to(device)
    if numr_dict["gradient_type"] == 'autodiff':
        T_conn = None

    hist_alpha = hist_alpha_init(inp, matprop, pffmodel, crack_dict)

    return inp, T_conn, area_T, hist_alpha


def _select_boundary_nodes(X, Y, tol):
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)

    top_nodes = np.where(np.abs(Y - y_max) <= tol)[0]
    bottom_nodes = np.where(np.abs(Y - y_min) <= tol)[0]
    left_nodes = np.where(np.abs(X - x_min) <= tol)[0]
    right_nodes = np.where(np.abs(X - x_max) <= tol)[0]

    point1 = np.argmin((X - x_min) ** 2 + (Y - y_min) ** 2)

    return {
        "top_nodes": top_nodes,
        "bottom_nodes": bottom_nodes,
        "left_nodes": left_nodes,
        "right_nodes": right_nodes,
        "point1_node": np.asarray([point1], dtype=np.int64),
    }


def _unique_nodes(nodes):
    if nodes is None:
        return np.empty((0,), dtype=np.int64)
    arr = np.asarray(nodes, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return np.empty((0,), dtype=np.int64)
    return np.unique(arr)


def _unique_edges(edges):
    if edges is None:
        return np.empty((0, 2), dtype=np.int64)
    arr = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    arr = np.sort(arr, axis=1)
    return np.unique(arr, axis=0)


def _boundary_edges_from_connectivity(T_conn):
    edge_count = {}
    for tri in T_conn:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b in ((i, j), (j, k), (k, i)):
            edge = (a, b) if a < b else (b, a)
            edge_count[edge] = edge_count.get(edge, 0) + 1
    boundary_edges = np.asarray([e for e, cnt in edge_count.items() if cnt == 1], dtype=np.int64)
    return _unique_edges(boundary_edges)


def _edges_to_nodes(edges):
    edges = _unique_edges(edges)
    if edges.size == 0:
        return np.empty((0,), dtype=np.int64)
    return np.unique(edges.reshape(-1))


def _edges_from_node_set(boundary_edges, node_set):
    boundary_edges = _unique_edges(boundary_edges)
    node_set = set(_unique_nodes(node_set).tolist())
    if boundary_edges.size == 0 or len(node_set) == 0:
        return np.empty((0, 2), dtype=np.int64)
    mask = np.asarray([(int(a) in node_set) and (int(b) in node_set) for a, b in boundary_edges], dtype=bool)
    return boundary_edges[mask]


def _edge_difference(edges_a, edges_b):
    edges_a = _unique_edges(edges_a)
    edges_b = _unique_edges(edges_b)
    if edges_a.size == 0:
        return edges_a
    if edges_b.size == 0:
        return edges_a
    set_b = set(map(tuple, edges_b.tolist()))
    keep = [e for e in edges_a.tolist() if tuple(e) not in set_b]
    return _unique_edges(keep)


def _extract_boundary_nodes_from_connectivity(X, Y, T_conn):
    """
    Extract true geometric boundary nodes from triangular connectivity.
    Boundary edges are edges that belong to exactly one triangle.
    This avoids classifying interior partition lines as notch boundaries.
    """
    boundary_edges = _boundary_edges_from_connectivity(T_conn)
    if boundary_edges.size == 0:
        return None

    boundary_nodes = _edges_to_nodes(boundary_edges)

    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)
    tol = 1e-9 * max(x_max - x_min, y_max - y_min, 1.0)

    top_nodes = boundary_nodes[np.abs(Y[boundary_nodes] - y_max) <= tol]
    bottom_nodes = boundary_nodes[np.abs(Y[boundary_nodes] - y_min) <= tol]
    left_nodes = boundary_nodes[np.abs(X[boundary_nodes] - x_min) <= tol]
    right_nodes = boundary_nodes[np.abs(X[boundary_nodes] - x_max) <= tol]

    outer_nodes = np.union1d(np.union1d(top_nodes, bottom_nodes), np.union1d(left_nodes, right_nodes))
    notch_nodes = np.setdiff1d(boundary_nodes, outer_nodes)

    return {
        "top_nodes": top_nodes.astype(np.int64),
        "bottom_nodes": bottom_nodes.astype(np.int64),
        "left_nodes": left_nodes.astype(np.int64),
        "right_nodes": right_nodes.astype(np.int64),
        "notch_face_nodes": notch_nodes.astype(np.int64),
        "boundary_edges": boundary_edges.astype(np.int64),
    }


def _keywords_hit(name, keywords):
    if name is None:
        return False
    lname = str(name).strip().lower()
    return any(k in lname for k in keywords)


def _classify_boundary_group(group_tag, group_name, rule_tags):
    categories = []
    if _keywords_hit(group_name, ("top", "upper", "load", "displacement", "uy")):
        categories.append("mechanical_top")
    if _keywords_hit(group_name, ("bottom", "lower")):
        categories.append("mechanical_bottom")
    if _keywords_hit(group_name, ("fix", "support", "clamp", "constraint")):
        categories.append("mechanical_bottom")
    if _keywords_hit(group_name, ("notch", "crack", "precrack", "pre-crack", "slot")):
        categories.append("notch_faces")
    if _keywords_hit(group_name, ("thermal", "temperature", "isothermal", "t0", "dirichlet")):
        categories.append("thermal_dirichlet")
    if _keywords_hit(group_name, ("insulated", "insulation", "adiabatic", "heatflux0", "neumann")):
        categories.append("thermal_insulated")
    if _keywords_hit(group_name, ("point1", "point_1", "fixed_point")):
        categories.append("fixed_point")

    # Numeric defaults from report (physical tags).
    if group_tag in rule_tags["mechanical_top"]:
        categories.append("mechanical_top")
    if group_tag in rule_tags["mechanical_bottom"]:
        categories.append("mechanical_bottom")
    if group_tag in rule_tags["fixed_point"]:
        categories.append("fixed_point")
    if group_tag in rule_tags["thermal_dirichlet"]:
        categories.append("thermal_dirichlet")
    if group_tag in rule_tags["notch_faces"]:
        categories.append("notch_faces")
    if group_tag in rule_tags["thermal_insulated"]:
        categories.append("thermal_insulated")

    return list(dict.fromkeys(categories))


def _bc_from_physical_groups(X, Y, boundary_data, boundary_tag_dict=None):
    """
    Build boundary sets from .msh Physical Groups (preferred path).
    Returns None when no useful physical group information is available.
    """
    if boundary_data is None:
        return None

    has_phys = bool(boundary_data.get("has_physical_groups", False))
    phys_edges = boundary_data.get("physical_edges_by_tag", {})
    phys_points = boundary_data.get("physical_points_by_tag", {})
    phys_name_map = boundary_data.get("physical_name_map", {})
    if (not has_phys) or (len(phys_edges) == 0 and len(phys_points) == 0):
        return None

    tag_rules = {
        "mechanical_top": set([10]),
        "mechanical_bottom": set([2]),
        "fixed_point": set([1]),
        "thermal_dirichlet": set([12, 13, 14]),
        "notch_faces": set([]),
        "thermal_insulated": set([]),
    }
    if boundary_tag_dict is not None:
        for key in tag_rules.keys():
            if key in boundary_tag_dict:
                tag_rules[key] = set([int(v) for v in boundary_tag_dict[key]])

    mech_top_edges = []
    mech_bottom_edges = []
    thermal_dir_edges = []
    thermal_ins_edges = []
    notch_edges = []
    fixed_points = []

    for ptag, edges in phys_edges.items():
        ptag_i = int(ptag)
        name = phys_name_map.get((1, ptag_i), f"physical_{ptag_i}")
        categories = _classify_boundary_group(ptag_i, name, tag_rules)
        if len(categories) == 0:
            continue
        if "mechanical_top" in categories:
            mech_top_edges.append(edges)
        if "mechanical_bottom" in categories:
            mech_bottom_edges.append(edges)
        if "thermal_dirichlet" in categories:
            thermal_dir_edges.append(edges)
        if "thermal_insulated" in categories:
            thermal_ins_edges.append(edges)
        if "notch_faces" in categories:
            notch_edges.append(edges)

    for ptag, nodes in phys_points.items():
        ptag_i = int(ptag)
        name = phys_name_map.get((0, ptag_i), f"physical_{ptag_i}")
        categories = _classify_boundary_group(ptag_i, name, tag_rules)
        if "fixed_point" in categories:
            fixed_points.append(nodes)

    mech_top_edges = _unique_edges(np.vstack(mech_top_edges)) if len(mech_top_edges) > 0 else np.empty((0, 2), dtype=np.int64)
    mech_bottom_edges = _unique_edges(np.vstack(mech_bottom_edges)) if len(mech_bottom_edges) > 0 else np.empty((0, 2), dtype=np.int64)
    thermal_dir_edges = _unique_edges(np.vstack(thermal_dir_edges)) if len(thermal_dir_edges) > 0 else np.empty((0, 2), dtype=np.int64)
    thermal_ins_edges = _unique_edges(np.vstack(thermal_ins_edges)) if len(thermal_ins_edges) > 0 else np.empty((0, 2), dtype=np.int64)
    notch_edges = _unique_edges(np.vstack(notch_edges)) if len(notch_edges) > 0 else np.empty((0, 2), dtype=np.int64)
    fixed_points = _unique_nodes(np.concatenate(fixed_points)) if len(fixed_points) > 0 else np.empty((0,), dtype=np.int64)

    # If physical groups exist but none mapped to categories, return None to use geometric fallback.
    if (
        mech_top_edges.size == 0
        and mech_bottom_edges.size == 0
        and thermal_dir_edges.size == 0
        and notch_edges.size == 0
        and fixed_points.size == 0
    ):
        return None

    out = {
        "mechanical_top_edges": mech_top_edges,
        "mechanical_bottom_edges": mech_bottom_edges,
        "thermal_dirichlet_edges": thermal_dir_edges,
        "thermal_insulated_edges": thermal_ins_edges,
        "notch_face_edges": notch_edges,
        "mechanical_top_nodes": _edges_to_nodes(mech_top_edges),
        "mechanical_bottom_nodes": _edges_to_nodes(mech_bottom_edges),
        "thermal_dirichlet_nodes": _edges_to_nodes(thermal_dir_edges),
        "notch_face_nodes": _edges_to_nodes(notch_edges),
        "fixed_point_nodes": fixed_points,
        "source": "physical_groups",
    }
    return out


def prep_input_data_tm(
    tm_model,
    crack_dict,
    mesh_file,
    device,
    length_scale=1e-3,
    boundary_tag_dict=None,
):
    """
    Mesh-based input preparation for thermo-mechanical mixed-mode training.
    Real pre-crack is represented by geometric boundary, not by phase-field seed.
    crack_dict is accepted for interface compatibility and is not used here.
    """
    assert Path(mesh_file).suffix == ".msh", "Mesh file should be a .msh file"

    X, Y, T_conn, area_T, boundary_data = parse_mesh(
        filename=mesh_file,
        gradient_type="numerical",
        return_boundary_data=True,
    )
    X = X * length_scale
    Y = Y * length_scale
    area_T = np.abs(area_T) * (length_scale ** 2)

    inp = torch.from_numpy(np.column_stack((X, Y))).to(torch.float).to(device)
    T_conn = torch.from_numpy(T_conn).to(torch.long).to(device)
    area_T = torch.from_numpy(area_T).to(torch.float).to(device)

    boundary_edges_from_conn = _boundary_edges_from_connectivity(T_conn)
    bnd_from_mesh = _extract_boundary_nodes_from_connectivity(X, Y, T_conn)
    if bnd_from_mesh is None:
        tol = 1e-10 * max(np.max(np.abs(X)), np.max(np.abs(Y)), 1.0)
        bnd_geom = _select_boundary_nodes(X, Y, tol)
        top_geom = bnd_geom["top_nodes"]
        bottom_geom = bnd_geom["bottom_nodes"]
        left_geom = bnd_geom["left_nodes"]
        right_geom = bnd_geom["right_nodes"]
        notch_geom = np.asarray([], dtype=np.int64)
    else:
        top_geom = bnd_from_mesh["top_nodes"]
        bottom_geom = bnd_from_mesh["bottom_nodes"]
        left_geom = bnd_from_mesh["left_nodes"]
        right_geom = bnd_from_mesh["right_nodes"]
        notch_geom = bnd_from_mesh["notch_face_nodes"]

    bc_phys = _bc_from_physical_groups(
        X=X,
        Y=Y,
        boundary_data=boundary_data,
        boundary_tag_dict=boundary_tag_dict,
    )

    if bc_phys is None:
        mechanical_top_nodes = _unique_nodes(top_geom)
        mechanical_bottom_nodes = _unique_nodes(bottom_geom)
        notch_face_nodes = _unique_nodes(notch_geom)
        fixed_point_nodes = np.empty((0,), dtype=np.int64)
        # COMSOL-aligned fallback thermal BC:
        # only right boundary is Dirichlet T=T0, others are insulated.
        thermal_dirichlet_nodes = _unique_nodes(right_geom)
        thermal_dirichlet_edges = _edges_from_node_set(boundary_edges_from_conn, thermal_dirichlet_nodes)
        mechanical_top_edges = _edges_from_node_set(boundary_edges_from_conn, mechanical_top_nodes)
        mechanical_bottom_edges = _edges_from_node_set(boundary_edges_from_conn, mechanical_bottom_nodes)
        notch_face_edges = _edges_from_node_set(boundary_edges_from_conn, notch_face_nodes)
        thermal_insulated_edges = _edge_difference(boundary_edges_from_conn, thermal_dirichlet_edges)
        boundary_source = "geometry_fallback"
    else:
        mechanical_top_nodes = _unique_nodes(bc_phys["mechanical_top_nodes"])
        mechanical_bottom_nodes = _unique_nodes(bc_phys["mechanical_bottom_nodes"])
        notch_face_nodes = _unique_nodes(bc_phys["notch_face_nodes"])
        fixed_point_nodes = _unique_nodes(bc_phys["fixed_point_nodes"])
        thermal_dirichlet_nodes = _unique_nodes(bc_phys["thermal_dirichlet_nodes"])

        line_edges_all = _unique_edges(boundary_data.get("line_edges_all", np.empty((0, 2), dtype=np.int64)))
        boundary_edges_all = line_edges_all if line_edges_all.size > 0 else boundary_edges_from_conn

        mechanical_top_edges = _unique_edges(bc_phys["mechanical_top_edges"])
        mechanical_bottom_edges = _unique_edges(bc_phys["mechanical_bottom_edges"])
        thermal_dirichlet_edges = _unique_edges(bc_phys["thermal_dirichlet_edges"])
        notch_face_edges = _unique_edges(bc_phys["notch_face_edges"])
        thermal_insulated_edges = _unique_edges(bc_phys["thermal_insulated_edges"])

        if mechanical_top_edges.size == 0 and mechanical_top_nodes.size > 0:
            mechanical_top_edges = _edges_from_node_set(boundary_edges_all, mechanical_top_nodes)
        if mechanical_bottom_edges.size == 0 and mechanical_bottom_nodes.size > 0:
            mechanical_bottom_edges = _edges_from_node_set(boundary_edges_all, mechanical_bottom_nodes)
        if thermal_dirichlet_edges.size == 0 and thermal_dirichlet_nodes.size > 0:
            thermal_dirichlet_edges = _edges_from_node_set(boundary_edges_all, thermal_dirichlet_nodes)
        if notch_face_edges.size == 0 and notch_face_nodes.size > 0:
            notch_face_edges = _edges_from_node_set(boundary_edges_all, notch_face_nodes)
        if thermal_insulated_edges.size == 0:
            thermal_insulated_edges = _edge_difference(boundary_edges_all, thermal_dirichlet_edges)

        # fill missing essential groups from geometry fallback
        if mechanical_top_nodes.size == 0:
            mechanical_top_nodes = _unique_nodes(top_geom)
        if mechanical_bottom_nodes.size == 0:
            mechanical_bottom_nodes = _unique_nodes(bottom_geom)
        if notch_face_nodes.size == 0:
            notch_face_nodes = _unique_nodes(notch_geom)
            notch_face_edges = _edges_from_node_set(boundary_edges_all, notch_face_nodes)
        if thermal_dirichlet_nodes.size == 0:
            # COMSOL-aligned fallback thermal BC:
            # only right boundary is Dirichlet T=T0, others are insulated.
            thermal_dirichlet_nodes = _unique_nodes(right_geom)
            thermal_dirichlet_edges = _edges_from_node_set(boundary_edges_all, thermal_dirichlet_nodes)
            thermal_insulated_edges = _edge_difference(boundary_edges_all, thermal_dirichlet_edges)

        boundary_source = bc_phys["source"]

    x_min, y_min = np.min(X), np.min(Y)
    point1 = np.argmin((X - x_min) ** 2 + (Y - y_min) ** 2)
    if fixed_point_nodes.size == 0:
        fixed_point_nodes = np.asarray([point1], dtype=np.int64)

    def _to_torch_nodes(arr):
        return torch.from_numpy(_unique_nodes(arr)).to(torch.long).to(device)

    def _to_torch_edges(arr):
        arr_u = _unique_edges(arr)
        return torch.from_numpy(arr_u).to(torch.long).to(device)

    bc_dict = {
        # Preferred keys (physical-group-ready)
        "mechanical_top_nodes": _to_torch_nodes(mechanical_top_nodes),
        "mechanical_bottom_nodes": _to_torch_nodes(mechanical_bottom_nodes),
        "fixed_point_nodes": _to_torch_nodes(fixed_point_nodes),
        "thermal_dirichlet_nodes": _to_torch_nodes(thermal_dirichlet_nodes),
        "notch_face_nodes": _to_torch_nodes(notch_face_nodes),
        "mechanical_top_edges": _to_torch_edges(mechanical_top_edges),
        "mechanical_bottom_edges": _to_torch_edges(mechanical_bottom_edges),
        "thermal_dirichlet_edges": _to_torch_edges(thermal_dirichlet_edges),
        "thermal_insulated_edges": _to_torch_edges(thermal_insulated_edges),
        "notch_face_edges": _to_torch_edges(notch_face_edges),
        "boundary_source": boundary_source,
        "physical_groups_available": bool(boundary_data.get("has_physical_groups", False)),
        # Backward-compatible aliases
        "top_nodes": _to_torch_nodes(mechanical_top_nodes),
        "bottom_nodes": _to_torch_nodes(mechanical_bottom_nodes),
        "left_nodes": _to_torch_nodes(left_geom),
        "right_nodes": _to_torch_nodes(right_geom),
        "point1_node": _to_torch_nodes(np.asarray([fixed_point_nodes[0]], dtype=np.int64)),
        # Default report case uses constant T0 on the selected thermal boundaries.
        # time-dependent targets can be injected later via:
        # - thermal_dirichlet_fn(t)
        # - thermal_dirichlet_value
        # - thermal_dirichlet_values
        "thermal_dirichlet_value": None,
    }

    # In pre-cracked geometry mode, material-domain phase field starts from zero.
    hist_d = torch.zeros(inp.shape[0], dtype=inp.dtype, device=inp.device)

    return inp, T_conn, area_T, bc_dict, hist_d
