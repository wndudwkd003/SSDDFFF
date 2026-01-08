import torch
from config.config import Config
from core.projector.linear import LinearProjector
from core.projector.mlp import MLP


def get_head(head: str, input_dim: int, output_dim: int) -> torch.nn.Module:
    if head == "linear":
        head = LinearProjector(
            input_dim=input_dim,
            output_dim=output_dim,
        )
    elif head == "mlp":
        head = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
        )
    else:
        raise ValueError(f"Unknown head type: {head}")

    return head
