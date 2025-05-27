# scripts/probe_mouse_stats.py
import json, random, torch, numpy as np
from pathlib import Path
from owl_idms.data.cod_datasets import get_cod_paths

N_FILES     = 318        # 1–2 min on an SSD is plenty
SAMPLE_PER  = 1_000      # samples per file
PCTS        = [0.5, 50, 95, 99.5]   # tweak as you like
rng         = np.random.default_rng(0)

root   = Path("/home/shared/cod_data")
paths  = get_cod_paths(root)

reservoir = []
for idx, (_, m_path, _) in enumerate(random.sample(paths, N_FILES)):
    tensor = torch.load(m_path, mmap=True, map_location="cpu")
    if tensor.ndim != 2 or tensor.shape[1] < 2:          # sanity check
        continue
    # uniform slice without loading everything
    if len(tensor) > SAMPLE_PER:
        offset = rng.integers(0, len(tensor) - SAMPLE_PER)
        reservoir.append(tensor[offset:offset + SAMPLE_PER])
    else:
        reservoir.append(tensor)

all_deltas = torch.cat(reservoir, 0).float()             # (N, 2)
mag        = torch.linalg.vector_norm(all_deltas, dim=1)

stats = {
    "clip_mag": float(torch.quantile(mag, 0.995)),       # ≈99.5 %
    "median":   float(torch.median(mag)),
    "iqr":      float(torch.quantile(mag, 0.75)
                      - torch.quantile(mag, 0.25)),
    "x_mean":   float(all_deltas[:,0].mean()),
    "x_std":    float(all_deltas[:,0].std()),             # optional
    # …add anything else you need
}

Path("./mouse_norm_stats.json").write_text(json.dumps(stats, indent=2))
print("Saved:", stats)
