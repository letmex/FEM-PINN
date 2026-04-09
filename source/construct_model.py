import torch
from pff_model import PFFModel
from material_properties import MaterialProperties
from thermal_properties import ThermalProperties
from thermo_mech_model import ThermoMechModel
from network import NeuralNet, init_xavier, MonolithicTMPhaseNet

def construct_model(PFF_model_dict, mat_prop_dict, network_dict, domain_extrema, device):
    # Phase field model
    pffmodel = PFFModel(PFF_model = PFF_model_dict["PFF_model"], 
                        se_split = PFF_model_dict["se_split"],
                        tol_ir = torch.tensor(PFF_model_dict["tol_ir"], device=device))

    # Material model
    matprop = MaterialProperties(mat_E = torch.tensor(mat_prop_dict["mat_E"], device=device), 
                                mat_nu = torch.tensor(mat_prop_dict["mat_nu"], device=device), 
                                w1 = torch.tensor(mat_prop_dict["w1"], device=device), 
                                l0 = torch.tensor(mat_prop_dict["l0"], device=device))

    # Neural network
    network = NeuralNet(input_dimension=domain_extrema.shape[0], 
                        output_dimension=domain_extrema.shape[0]+1,
                        n_hidden_layers=network_dict["hidden_layers"],
                        neurons=network_dict["neurons"],
                        activation=network_dict["activation"],
                        init_coeff=network_dict["init_coeff"])
    torch.manual_seed(network_dict["seed"])
    init_xavier(network)

    return pffmodel, matprop, network


def construct_tm_model(tm_model_dict, thermal_prop_dict, network_dict, domain_extrema, device):
    thermo_model = ThermoMechModel(
        E0=torch.tensor(tm_model_dict["E0"], device=device),
        v0=torch.tensor(tm_model_dict["v0"], device=device),
        GcI=torch.tensor(tm_model_dict["GcI"], device=device),
        GcII=torch.tensor(tm_model_dict["GcII"], device=device),
        kappa=torch.tensor(tm_model_dict["kappa"], device=device),
        l0=torch.tensor(tm_model_dict["l0"], device=device),
        etaPF=torch.tensor(tm_model_dict["etaPF"], device=device),
        eps_r=torch.tensor(tm_model_dict["eps_r"], device=device),
    )

    thermal_prop = ThermalProperties(
        alpha=torch.tensor(thermal_prop_dict["alpha"], device=device),
        rho=torch.tensor(thermal_prop_dict["rho"], device=device),
        k0=torch.tensor(thermal_prop_dict["k0"], device=device),
        c=torch.tensor(thermal_prop_dict["c"], device=device),
        T0=torch.tensor(thermal_prop_dict["T0"], device=device),
        TFinal=torch.tensor(thermal_prop_dict["TFinal"], device=device),
        Tref=torch.tensor(thermal_prop_dict["Tref"], device=device),
        thk=torch.tensor(thermal_prop_dict["thk"], device=device),
    ).to(device)

    monolithic_net = MonolithicTMPhaseNet(
        spatial_dimension=domain_extrema.shape[0],
        n_hidden_layers=network_dict["hidden_layers"],
        neurons=network_dict["neurons"],
        activation=network_dict["activation"],
        init_coeff=network_dict["init_coeff"],
        seed=network_dict.get("seed", 1),
        phase_output_bias_init=network_dict.get("phase_output_bias", -4.0),
        domain_extrema=domain_extrema.to(device),
        use_input_normalization=network_dict.get("use_input_normalization", True),
    )

    return thermo_model, thermal_prop, monolithic_net
