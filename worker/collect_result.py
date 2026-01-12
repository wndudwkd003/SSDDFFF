from utils.losses import ReconLoss, anomaly_score_l1
from utils.strategy import confident_strategy

from tqdm.auto import tqdm
from typing import Any
import torch
from torch import nn


@torch.no_grad()
def collect_meta_scores_from_ae(model: nn.Module, loader, device: torch.device):
    model.eval()

    score_list = []
    filename_list = []
    media_type_list = []

    for batch in tqdm(loader, desc="Collect meta scores (ae)"):
        x = batch["pixel_values"].to(device)
        out = model(x)
        recon = out["recon"]
        scores = anomaly_score_l1(recon.detach(), x.detach())
        score_list.extend(scores.detach().cpu().numpy().tolist())

        filename_list.extend(list(batch["filename"]))
        media_type_list.extend(list(batch["media_type"]))

    return score_list, filename_list, media_type_list


def aggregate_per_file(
    values, filenames, media_types, *, kind: str, t_video: float = 0.8
):
    per_file: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    key = "probs" if kind == "probs" else "scores"

    for fn, mt, v in zip(filenames, media_types, values):
        if fn not in per_file:
            per_file[fn] = {"media_type": mt, key: []}
            order.append(fn)
        per_file[fn][key].append(float(v))

    out_filenames: list[str] = []
    out_values: list[float] = []

    for fn in order:
        mt = per_file[fn]["media_type"]
        vs = per_file[fn][key]

        if mt == "image":
            final_v = float(vs[0])
        else:
            final_v = float(confident_strategy(vs, t=float(t_video)))

        out_filenames.append(fn)
        out_values.append(final_v)

    return out_filenames, out_values, {k: v[key] for k, v in per_file.items()}


@torch.no_grad()
def collect_meta_probs_from_model(model: nn.Module, loader, device: torch.device):
    model.eval()

    prob_list = []
    filename_list = []
    media_type_list = []

    for batch in tqdm(loader, desc="Collect meta probs (model)"):
        x = batch["pixel_values"].to(device)
        out = model(x)
        logits = out["logits"]
        probs = torch.softmax(logits, dim=1)[:, 1]
        prob_list.extend(probs.detach().cpu().numpy().tolist())

        filename_list.extend(list(batch["filename"]))
        media_type_list.extend(list(batch["media_type"]))

    return prob_list, filename_list, media_type_list
