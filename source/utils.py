import torch
import torch.nn as nn
import numpy as np
import gmshparser
import re



class DistanceFunction:
    def __init__(self, x_init, y_init, theta, L, d0, order: int = 2):
        self.x_init = x_init
        self.y_init = y_init
        self.theta = theta
        self.L = L
        self.d0 = d0
        self.order = order

    def __call__(self, inp):
        '''
        This function computes distance function given a line with origin at (x_init, y_init),
        oriented at an angle theta from x-axis, and of length L. Value of the function is 1 at
        the line and goes to 0 at a distance of d0 from the line.

        '''

        L = torch.tensor([self.L], device=inp.device)
        d0 = torch.tensor([self.d0], device=inp.device)
        theta = torch.tensor([self.theta], device=inp.device)
        input_c = torch.clone(inp)

        # transform coordinate to shift origin to (x_init, y_init) and rotate axis by theta
        input_c[:, -2:] = input_c[:, -2:] - torch.tensor([self.x_init, self.y_init], device=inp.device)
        Rt = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]], device=inp.device)
        input_c[:, -2:] = torch.matmul(input_c[:, -2:], Rt)
        x = input_c[:, -2]
        y = input_c[:, -1]

        if self.order == 1:
            dist_fn_p1 = nn.ReLU()(x*(L-x))/(abs(x*(L-x))+np.finfo(float).eps)* \
                            nn.ReLU()(d0-abs(y))/(abs(d0-abs(y))+np.finfo(float).eps)* \
                            (1-abs(y)/d0)
            
            dist_fn_p2 = nn.ReLU()(x-L)/(abs(x-L)+np.finfo(float).eps)* \
                            nn.ReLU()(d0**2-((x-L)**2+y**2))/(abs(d0**2-((x-L)**2+y**2))+np.finfo(float).eps)* \
                            (1-torch.sqrt((x-L)**2+y**2)/d0)
            
            dist_fn_p3 = nn.ReLU()(-x)/(abs(x)+np.finfo(float).eps)* \
                            nn.ReLU()(d0**2-(x**2+y**2))/(abs(d0**2-(x**2+y**2))+np.finfo(float).eps)* \
                            (1-torch.sqrt(x**2 + y**2)/d0)
            
            dist_fn = dist_fn_p1 + dist_fn_p2 + dist_fn_p3

            
        if self.order == 2:
            dist_fn_p1 = nn.ReLU()(x*(L-x))/(abs(x*(L-x))+np.finfo(float).eps)* \
                            nn.ReLU()(d0-abs(y))/(abs(d0-abs(y))+np.finfo(float).eps)* \
                            (1-abs(y)/d0)**2
            
            dist_fn_p2 = nn.ReLU()(x-L)/(abs(x-L)+np.finfo(float).eps)* \
                            nn.ReLU()(d0**2-((x-L)**2+y**2))/(abs(d0**2-((x-L)**2+y**2))+np.finfo(float).eps)* \
                            (1-torch.sqrt((x-L)**2+y**2)/d0)**2
            
            dist_fn_p3 = nn.ReLU()(-x)/(abs(x)+np.finfo(float).eps)* \
                            nn.ReLU()(d0**2-(x**2+y**2))/(abs(d0**2-(x**2+y**2))+np.finfo(float).eps)* \
                            (1-torch.sqrt(x**2 + y**2)/d0)**2
            
            dist_fn = dist_fn_p1 + dist_fn_p2 + dist_fn_p3

        return dist_fn
    

    

def hist_alpha_init(inp, matprop, pffmodel, crack_dict):
    '''
    This function computes the initial phase field for a sample with a crack.
    See the paper "Phase-field modeling of fracture with physics-informed deep learning" for details.

    '''
    hist_alpha = torch.zeros((inp.shape[0], ), device = inp.device)
    
    if crack_dict["L_crack"][0] > 0:
        l0 = matprop.l0
        for j, L_crack in enumerate(crack_dict["L_crack"]):
            Lc = torch.tensor([L_crack], device=inp.device)
            theta = torch.tensor([crack_dict["angle_crack"][j]], device=inp.device)
            input_c = torch.clone(inp)

            # transform coordinate to shift origin to (x_init, y_init) and rotate axis by theta
            input_c[:, -2:] = input_c[:, -2:] - torch.tensor([crack_dict["x_init"][j], crack_dict["y_init"][j]], device=inp.device)
            Rt = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]], device=inp.device)
            input_c[:, -2:] = torch.matmul(input_c[:, -2:], Rt)
            x = input_c[:, -2]
            y = input_c[:, -1]

            if pffmodel.PFF_model == 'AT1':
                hist_alpha_p1 = nn.ReLU()(x*(Lc-x))/(abs(x*(Lc-x))+np.finfo(float).eps)* \
                                    nn.ReLU()(2*l0-abs(y))/(abs(2*l0-abs(y))+np.finfo(float).eps)* \
                                    (1-abs(y)/l0/2)**2

                hist_alpha_p2 = nn.ReLU()(x-Lc+np.finfo(float).eps)/(abs(x-Lc)+np.finfo(float).eps)* \
                                    nn.ReLU()(2*l0-torch.sqrt((x-Lc)**2+y**2)+np.finfo(float).eps)/(abs(2*l0-torch.sqrt((x-Lc)**2+y**2))+np.finfo(float).eps)* \
                                    (1-torch.sqrt((x-Lc)**2+y**2)/l0/2)**2

                hist_alpha_p3 = nn.ReLU()(-x+np.finfo(float).eps)/(abs(x)+np.finfo(float).eps)* \
                                    nn.ReLU()(2*l0-torch.sqrt(x**2+y**2)+np.finfo(float).eps)/(abs(2*l0-torch.sqrt(x**2+y**2))+np.finfo(float).eps)* \
                                    (1-torch.sqrt(x**2+y**2)/l0/2)**2
                
            elif pffmodel.PFF_model == 'AT2':
                hist_alpha_p1 = nn.ReLU()(x*(Lc-x))/(abs(x*(Lc-x))+np.finfo(float).eps)* \
                                    torch.exp(-abs(y)/l0)

                hist_alpha_p2 = nn.ReLU()(x-Lc+np.finfo(float).eps)/(abs(x-Lc)+np.finfo(float).eps)* \
                                    torch.exp(-torch.sqrt((x-Lc)**2+y**2)/l0)

                hist_alpha_p3 = nn.ReLU()(-x+np.finfo(float).eps)/(abs(x)+np.finfo(float).eps)* \
                                    torch.exp(-torch.sqrt(x**2+y**2)/l0)

            hist_alpha = hist_alpha + hist_alpha_p1 + hist_alpha_p2 + hist_alpha_p3

    return hist_alpha



def _parse_msh_physical_metadata(filename):
    """
    Lightweight parser for $PhysicalNames and $Entities sections in .msh (v4+ text).
    Returns:
    - physical_name_map: {(dim, physical_tag): name}
    - entity_phys_map: {dim: {entity_tag: [physical_tags]}}
    """
    physical_name_map = {}
    entity_phys_map = {0: {}, 1: {}, 2: {}, 3: {}}

    try:
        with open(filename, "r", encoding="utf-8", errors="ignore") as file:
            lines = file.read().splitlines()
    except OSError:
        return physical_name_map, entity_phys_map

    n_lines = len(lines)
    i = 0
    while i < n_lines:
        line = lines[i].strip()

        if line == "$PhysicalNames" and i + 1 < n_lines:
            i += 1
            try:
                n_phys = int(lines[i].strip())
            except ValueError:
                n_phys = 0
            i += 1
            for _ in range(n_phys):
                if i >= n_lines:
                    break
                rec = lines[i].strip()
                match = re.match(r"^\s*(\d+)\s+(\d+)\s+\"(.*)\"\s*$", rec)
                if match:
                    dim = int(match.group(1))
                    tag = int(match.group(2))
                    name = match.group(3)
                    physical_name_map[(dim, tag)] = name
                else:
                    parts = rec.split(maxsplit=2)
                    if len(parts) >= 2:
                        try:
                            dim = int(parts[0])
                            tag = int(parts[1])
                            name = parts[2].strip().strip("\"") if len(parts) == 3 else f"physical_{tag}"
                            physical_name_map[(dim, tag)] = name
                        except ValueError:
                            pass
                i += 1
            continue

        if line == "$Entities" and i + 1 < n_lines:
            i += 1
            try:
                n_points, n_curves, n_surfaces, n_volumes = [int(v) for v in lines[i].strip().split()]
            except ValueError:
                n_points = n_curves = n_surfaces = n_volumes = 0
            i += 1

            # points
            for _ in range(n_points):
                if i >= n_lines:
                    break
                tok = lines[i].strip().split()
                if len(tok) >= 5:
                    try:
                        tag = int(tok[0])
                        n_phys = int(tok[4])
                        phys = [int(v) for v in tok[5:5 + n_phys]]
                        entity_phys_map[0][tag] = phys
                    except ValueError:
                        pass
                i += 1

            # curves / surfaces / volumes share same prefix pattern in v4:
            # tag minX minY minZ maxX maxY maxZ numPhysicalTags ...
            for dim, n_ent in ((1, n_curves), (2, n_surfaces), (3, n_volumes)):
                for _ in range(n_ent):
                    if i >= n_lines:
                        break
                    tok = lines[i].strip().split()
                    if len(tok) >= 8:
                        try:
                            tag = int(tok[0])
                            n_phys = int(tok[7])
                            phys = [int(v) for v in tok[8:8 + n_phys]]
                            entity_phys_map[dim][tag] = phys
                        except ValueError:
                            pass
                    i += 1
            continue

        i += 1

    return physical_name_map, entity_phys_map


def _unique_edges(edges):
    if edges is None or len(edges) == 0:
        return np.empty((0, 2), dtype=np.int64)
    arr = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    arr = np.sort(arr, axis=1)
    arr = np.unique(arr, axis=0)
    return arr


def parse_mesh(filename="meshed_geom.msh", gradient_type='numerical', return_boundary_data=False):
    '''
    Parses .msh file to obtain nodal coordinates and triangular connectivity.
    This workflow assumes 2D triangular discretization only.
    If numr_dict["gradient_type"] = autodiff, then Gauss points of elements in a one point Gauss 
    quadrature are returned.

    '''

    mesh = gmshparser.parse(filename)
    physical_name_map, entity_phys_map = _parse_msh_physical_metadata(filename)

    # Gmsh 2D triangle element types used in this project:
    # - 2: first-order triangle (T3)
    # - 9: second-order triangle (T6)
    tri_2d_types = {2, 9}

    # Build node arrays from node tags for robust indexing.
    node_coords = {}
    point_nodes_by_entity = {}
    for node_entity in mesh.get_node_entities():
        ent_dim = int(node_entity.get_dimension())
        ent_tag = int(node_entity.get_tag())
        ent_node_ids = []
        for node in node_entity.get_nodes():
            xyz = node.get_coordinates()
            ntag = int(node.get_tag())
            node_coords[ntag] = (float(xyz[0]), float(xyz[1]))
            ent_node_ids.append(ntag)
        if ent_dim == 0:
            point_nodes_by_entity[ent_tag] = ent_node_ids

    assert len(node_coords) > 0, "Mesh has no nodes."
    node_tags_sorted = np.asarray(sorted(node_coords.keys()), dtype=np.int64)
    X = np.asarray([node_coords[int(tag)][0] for tag in node_tags_sorted], dtype=np.float64)
    Y = np.asarray([node_coords[int(tag)][1] for tag in node_tags_sorted], dtype=np.float64)
    tag_to_idx = {int(tag): idx for idx, tag in enumerate(node_tags_sorted)}

    tri_conn = []
    line_edges_by_entity = {}
    non_tri_2d = {}
    for elem_entity in mesh.get_element_entities():
        dim = int(elem_entity.get_dimension())
        etype = int(elem_entity.get_element_type())
        ent_tag = int(elem_entity.get_tag())
        elems = list(elem_entity.get_elements())

        if dim == 2:
            if etype in tri_2d_types:
                for elem in elems:
                    conn = list(elem.get_connectivity())
                    if etype == 2:
                        if len(conn) < 3:
                            continue
                        tri_conn.append(
                            [
                                tag_to_idx[int(conn[0])],
                                tag_to_idx[int(conn[1])],
                                tag_to_idx[int(conn[2])],
                            ]
                        )
                    elif etype == 9:
                        if len(conn) < 6:
                            continue
                        # T6 -> 4*T3 subdivision using midside nodes:
                        # [v1,v2,v3,m12,m23,m31] ->
                        # [v1,m12,m31], [m12,v2,m23], [m31,m23,v3], [m12,m23,m31]
                        v1 = tag_to_idx[int(conn[0])]
                        v2 = tag_to_idx[int(conn[1])]
                        v3 = tag_to_idx[int(conn[2])]
                        m12 = tag_to_idx[int(conn[3])]
                        m23 = tag_to_idx[int(conn[4])]
                        m31 = tag_to_idx[int(conn[5])]
                        tri_conn.extend(
                            [
                                [v1, m12, m31],
                                [m12, v2, m23],
                                [m31, m23, v3],
                                [m12, m23, m31],
                            ]
                        )
            else:
                non_tri_2d[etype] = non_tri_2d.get(etype, 0) + len(elems)
        elif dim == 1:
            edges = []
            for elem in elems:
                conn = list(elem.get_connectivity())
                if len(conn) < 2:
                    continue
                if len(conn) == 2:
                    a = tag_to_idx[int(conn[0])]
                    b = tag_to_idx[int(conn[1])]
                    if a != b:
                        edges.append([a, b])
                else:
                    # Split second-order line [n1,n2,nm] into two segments [n1,nm], [nm,n2].
                    # This preserves boundary edge resolution when T6 midside nodes are used.
                    a = tag_to_idx[int(conn[0])]
                    b = tag_to_idx[int(conn[1])]
                    m = tag_to_idx[int(conn[2])]
                    if a != m:
                        edges.append([a, m])
                    if m != b:
                        edges.append([m, b])
            if len(edges) > 0:
                line_edges_by_entity[ent_tag] = _unique_edges(edges)

    if len(tri_conn) == 0 or len(non_tri_2d) > 0:
        raise AssertionError(
            "Discretization must contain only 2D triangular elements. "
            f"Found non-triangular 2D element types: {non_tri_2d}"
        )

    T = np.asarray(tri_conn, dtype=np.int64)

    # Keep only nodes actually used by the linearized triangular connectivity.
    # This avoids orphan high-order midside nodes polluting nodal postprocessing
    # (e.g., HI/HII/He displayed as zeros at nodes not belonging to any triangle).
    used_nodes = np.unique(T.reshape(-1))
    old_to_new = -np.ones((X.shape[0],), dtype=np.int64)
    old_to_new[used_nodes] = np.arange(used_nodes.shape[0], dtype=np.int64)
    X = X[used_nodes]
    Y = Y[used_nodes]
    T = old_to_new[T]

    area = X[T[:, 0]]*(Y[T[:, 1]]-Y[T[:, 2]]) + X[T[:, 1]]*(Y[T[:, 2]]-Y[T[:, 0]]) + X[T[:, 2]]*(Y[T[:, 0]]-Y[T[:, 1]])
    area = 0.5*area

    if gradient_type == 'autodiff':
        X = (X[T[:, 0]] + X[T[:, 1]] + X[T[:, 2]])/3
        Y = (Y[T[:, 0]] + Y[T[:, 1]] + Y[T[:, 2]])/3

    if not return_boundary_data:
        return X, Y, T, area

    def _remap_nodes(nodes):
        nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
        if nodes.size == 0:
            return np.empty((0,), dtype=np.int64)
        valid = old_to_new[nodes] >= 0
        if not np.any(valid):
            return np.empty((0,), dtype=np.int64)
        out = old_to_new[nodes[valid]]
        return np.unique(out)

    def _remap_edges(edges):
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        if edges.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        a = old_to_new[edges[:, 0]]
        b = old_to_new[edges[:, 1]]
        valid = (a >= 0) & (b >= 0) & (a != b)
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.int64)
        e = np.column_stack((a[valid], b[valid])).astype(np.int64)
        e = np.sort(e, axis=1)
        return np.unique(e, axis=0)

    line_edges_by_entity = {
        ent_tag: _remap_edges(edges)
        for ent_tag, edges in line_edges_by_entity.items()
    }
    line_edges_by_entity = {
        ent_tag: edges for ent_tag, edges in line_edges_by_entity.items() if edges.size > 0
    }

    line_phys_tags_by_entity = {}
    physical_edges_by_tag = {}
    physical_nodes_by_tag = {}
    for ent_tag, edges in line_edges_by_entity.items():
        phys_tags = entity_phys_map.get(1, {}).get(ent_tag, [])
        line_phys_tags_by_entity[ent_tag] = list(phys_tags)
        for ptag in phys_tags:
            if ptag not in physical_edges_by_tag:
                physical_edges_by_tag[ptag] = []
            physical_edges_by_tag[ptag].append(edges)

    for ptag, edge_chunks in physical_edges_by_tag.items():
        edge_arr = _unique_edges(np.vstack(edge_chunks))
        edge_arr = _remap_edges(edge_arr)
        physical_edges_by_tag[ptag] = edge_arr
        physical_nodes_by_tag[ptag] = np.unique(edge_arr.reshape(-1)) if edge_arr.size > 0 else np.empty((0,), dtype=np.int64)

    physical_edges_by_tag = {k: v for k, v in physical_edges_by_tag.items() if v.size > 0}
    physical_nodes_by_tag = {k: _remap_nodes(v) for k, v in physical_nodes_by_tag.items()}
    physical_nodes_by_tag = {k: v for k, v in physical_nodes_by_tag.items() if v.size > 0}

    point_nodes_idx_by_entity = {}
    for ent_tag, node_ids in point_nodes_by_entity.items():
        idx_full = np.asarray([tag_to_idx[nid] for nid in node_ids if nid in tag_to_idx], dtype=np.int64)
        point_nodes_idx_by_entity[ent_tag] = _remap_nodes(idx_full)
    point_phys_tags_by_entity = {
        ent_tag: list(entity_phys_map.get(0, {}).get(ent_tag, []))
        for ent_tag in point_nodes_idx_by_entity.keys()
    }
    physical_points_by_tag = {}
    for ent_tag, nodes in point_nodes_idx_by_entity.items():
        for ptag in point_phys_tags_by_entity.get(ent_tag, []):
            if ptag not in physical_points_by_tag:
                physical_points_by_tag[ptag] = []
            if nodes.size > 0:
                physical_points_by_tag[ptag].append(nodes)
    for ptag, chunks in physical_points_by_tag.items():
        physical_points_by_tag[ptag] = _remap_nodes(np.unique(np.concatenate(chunks)))
    physical_points_by_tag = {k: v for k, v in physical_points_by_tag.items() if v.size > 0}

    line_edges_all = _unique_edges(np.vstack(list(line_edges_by_entity.values()))) if len(line_edges_by_entity) > 0 else np.empty((0, 2), dtype=np.int64)
    has_physical_groups = (
        any(len(tags) > 0 for dim_map in entity_phys_map.values() for tags in dim_map.values())
        or len(physical_name_map) > 0
    )

    boundary_data = {
        "line_edges_by_entity": line_edges_by_entity,
        "line_phys_tags_by_entity": line_phys_tags_by_entity,
        "line_edges_all": line_edges_all,
        "physical_edges_by_tag": physical_edges_by_tag,
        "physical_nodes_by_tag": physical_nodes_by_tag,
        "point_nodes_by_entity": point_nodes_idx_by_entity,
        "point_phys_tags_by_entity": point_phys_tags_by_entity,
        "physical_points_by_tag": physical_points_by_tag,
        "physical_name_map": physical_name_map,
        "entity_phys_map": entity_phys_map,
        "has_physical_groups": has_physical_groups,
    }

    return X, Y, T, area, boundary_data
