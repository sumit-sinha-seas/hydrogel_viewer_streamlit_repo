# Hydrogel Viewer (Streamlit)

This is a **pure viewer** for your hydrogel inverse-design runs.

- No JAX
- No PDE solve
- No GPU required

You run the heavy simulation once, export a **viewer bundle**, then the experimentalist opens a URL and inspects gels interactively.

---

## 1) Export a viewer bundle (run once on your machine)

In your notebook, after you have:
- `P_np` (simulated RGB image), float in [0,1], shape H x W x 3
- `H_img` (target image), either uint8 or float in [0,1]
- `df` (instruction table as a pandas DataFrame) with at least columns:
  - `GelIndex`, `x_um`, `y_um`, and either `R_uM...Y_uM` or `t_uM` columns you can rename

Run this export cell:

```python
import os, json
import numpy as np
from PIL import Image

BUNDLE_DIR = "viewer_bundle"
os.makedirs(BUNDLE_DIR, exist_ok=True)

# 1) Save simulated image
np.save(os.path.join(BUNDLE_DIR, "P_np.npy"), np.asarray(P_np).astype(np.float32))

# 2) Save target image (as PNG for convenience)
H_arr = np.asarray(H_img)
if H_arr.dtype != np.uint8:
    H_u8 = np.clip(np.rint(H_arr * 255.0), 0, 255).astype(np.uint8)
else:
    H_u8 = H_arr
Image.fromarray(H_u8).save(os.path.join(BUNDLE_DIR, "H_u8.png"))

# 3) Save instruction table
# Make sure GelIndex is a column name
if "GelIndex" not in df.columns:
    raise RuntimeError("df must contain a GelIndex column")
df.to_csv(os.path.join(BUNDLE_DIR, "instructions_table.csv"), index=False)

# 4) Save metadata
meta = {
    "H": int(P_np.shape[0]),
    "W": int(P_np.shape[1]),
    "L_PHYS_UM": float(L_PHYS_UM),
    "RGB_THR": int(RGB_THR),
}
with open(os.path.join(BUNDLE_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("Wrote bundle ->", BUNDLE_DIR)
```

Copy the resulting `viewer_bundle/` folder into this repo under `viewer_bundle/`.

---

## 2) Run locally (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 3) Deploy on Hugging Face Spaces

1. On Hugging Face, click **Create new Space** and choose **Streamlit** as the SDK. (The repo YAML should include `sdk: streamlit`.)
2. Upload/push:
   - `app.py`
   - `requirements.txt`
   - `viewer_bundle/` (with the files you exported)
3. Wait for the Space to build. The app will serve at your Space URL.

To update a run later, replace the files inside `viewer_bundle/` and push again.
