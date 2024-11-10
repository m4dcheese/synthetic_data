import torch
from config import config, make_dotdict_recursive
from models.cfm import build_cfm_from_config
from torch.nn.parallel import DistributedDataParallel
from utils import get_optimizer


def save_trained_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, save_path):
    path = f"{save_path}/model_weights.pth"

    module_ = model.module if isinstance(model, DistributedDataParallel) else model

    checkpoint = {
        "config": config.get_dict_recursive(),
        "model_state_dict": module_.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, path)


def load_trained_model(path: str, device: str = "cpu"):
    """Load model from checkpoint including config and weights."""
    data = torch.load(path, map_location=device)
    c = make_dotdict_recursive(data["config"])
    cfm_model = build_cfm_from_config(c)
    cfm_model.load_state_dict(data["model_state_dict"])
    optimizer = get_optimizer(c.optimizer.optimizer_str)(
        params=cfm_model.parameters(),
        lr=c.optimizer.lr,
        weight_decay=c.optimizer.weight_decay,
    )
    optimizer.load_state_dict(data["optimizer_state_dict"])
    return c, cfm_model, optimizer
