import torch.optim as optim
import numpy as np

def get_optimizer(params, optimizer_type: str="LBFGS"):
    if optimizer_type == "LBFGS":
        optimizer = optim.LBFGS(params, lr=float(0.5), max_iter=20000, max_eval=20000000, history_size=250,
                             line_search_fn="strong_wolfe",
                             tolerance_change=1.0*np.finfo(float).eps, tolerance_grad=1.0*np.finfo(float).eps)           
    elif optimizer_type == "ADAM":
        optimizer = optim.Adam(params, lr=5e-4, betas=(0.9, 0.999), eps=1.0*np.finfo(float).eps, weight_decay=0)
    elif optimizer_type == "RPROP":
        optimizer = optim.Rprop(params, lr=1e-5, step_sizes=(1e-10, 50))
    else:
        raise ValueError("Optimizer type not recognized. Please choose from LBFGS, ADAM, RPROP.")
    return optimizer


def get_optimizer_tm(params, optimizer_type: str = "LBFGS", settings: dict = None):
    if settings is None:
        settings = {}

    if optimizer_type == "LBFGS":
        lr = float(settings.get("lr_lbfgs", 0.5))
        max_iter = int(settings.get("max_iter_lbfgs", 500))
        history_size = int(settings.get("history_size_lbfgs", 100))
        optimizer = optim.LBFGS(
            params,
            lr=lr,
            max_iter=max_iter,
            max_eval=max_iter * 5,
            history_size=history_size,
            line_search_fn="strong_wolfe",
            tolerance_change=1.0 * np.finfo(float).eps,
            tolerance_grad=1.0 * np.finfo(float).eps,
        )
    elif optimizer_type == "RPROP":
        lr = float(settings.get("lr_rprop", 1e-5))
        step_lo = float(settings.get("rprop_step_lo", 1e-10))
        step_hi = float(settings.get("rprop_step_hi", 50.0))
        optimizer = optim.Rprop(params, lr=lr, step_sizes=(step_lo, step_hi))
    elif optimizer_type == "ADAM":
        lr = float(settings.get("lr_adam", 5e-4))
        optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1.0 * np.finfo(float).eps, weight_decay=0)
    else:
        raise ValueError("Optimizer type not recognized. Please choose from LBFGS, ADAM, RPROP.")
    return optimizer
