from runtime_env import configure_runtime_env

configure_runtime_env()

from config import *

PATH_SOURCE = Path(__file__).parents[2]
sys.path.insert(0, str(PATH_SOURCE / Path("source")))

from construct_model import construct_tm_model
from input_data_from_mesh import prep_input_data_tm
from postprocess_tm import postprocess_tm


if __name__ == "__main__":
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

    postprocess_tm(
        results_path=results_path,
        inp=inp,
        T_conn=T_conn,
        step_idx=-1,
        dpi=400,
        bc_dict=bc_dict,
    )
