"""Run the existing continuous-recording WPE and ordinary MVDR interfaces."""
import json
from pathlib import Path

import numpy as np

from dereverb import single_wpe
from spatial import apply_weights, mvdr, weighted_psd

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    with np.load(root / "fixtures/ordinary.npz") as data:
        y = data["segment_0"]
        frame_weights = data["noise_weight_0"]
        steering = data["steering"]
    x = single_wpe(y, taps=2, delay=2, iterations=3)
    r = weighted_psd([x], [frame_weights])
    w = np.stack([mvdr(rf, vf) for rf, vf in zip(r, steering)])
    enhanced = apply_weights(w, x)
    print(json.dumps({
        "residual_shape": list(x.shape), "enhanced_shape": list(enhanced.shape),
        "input_energy": float(np.sum(np.abs(y) ** 2)),
        "residual_energy": float(np.sum(np.abs(x) ** 2)),
        "response_residual": float(np.max(np.abs(np.sum(steering.conj()*w, axis=1)-1))),
    }, indent=2))
