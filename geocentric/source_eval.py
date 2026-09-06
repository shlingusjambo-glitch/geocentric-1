"""Bounded held-out loss by source, without crossing source token boundaries."""
from pathlib import Path
import json
import math

import numpy as np
import torch


@torch.no_grad()
def evaluate_sources(model, bin_path, meta_path, device, autocast, windows_per_source=2):
    if windows_per_source < 1:
        raise ValueError("windows_per_source must be positive")
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text())
    if not meta.get("sources"):
        return {}
    data = np.memmap(bin_path, dtype=meta["dtype"], mode="r")
    results = {}
    training, depth = model.training, model.active_layers
    model.eval()
    model.active_layers = None
    try:
        for entry in meta["sources"]:
            start, size = int(entry["start"]), int(entry["tokens"])
            if start < 0 or size < 0 or start + size > len(data):
                raise ValueError("Source metadata extends outside validation corpus")
            source = entry["source"]
            if size < 2:
                results[source] = {"loss": None, "tokens": 0, "reason": "Too few validation tokens"}
                continue
            context = min(model.config.block_size, size - 1)
            available = max(1, (size - 1) // context)
            indices = np.linspace(0, available - 1, min(windows_per_source, available), dtype=int)
            total, count = 0., 0
            for index in indices:
                offset = start + int(index) * context
                window = torch.tensor(np.array(data[offset:offset + context + 1], dtype=np.int64), device=device)
                with autocast:
                    _, loss = model(window[:-1][None], labels=window[1:][None],
                                    return_logits=False, loss_reduction="sum")
                total += float(loss)
                count += context
            value = total / count
            if not math.isfinite(value):
                raise ValueError(f"Non-finite validation loss for {source}")
            results[source] = {"loss": value, "perplexity": math.exp(min(value, 20)),
                               "tokens": count, "windows": len(indices), "context": context}
    finally:
        model.active_layers = depth
        model.train(training)
        del data
    return results
