import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# =========================================================
# Hydrogel Viewer (Streamlit)
# =========================================================

DEFAULT_BUNDLE_DIR = Path("viewer_bundle_iterated")

# Match Colab viewer
TOL_uM = 1e-6  # ON if uM > OFF_uM + TOL_uM


def _to_uint8(img01_or_u8):
    arr = np.asarray(img01_or_u8)
    if arr.dtype == np.uint8:
        return arr
    return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)


def pix_from_xy_um(x_um, y_um, L_PHYS_UM, W, H):
    ix = int(np.floor((float(x_um) / float(L_PHYS_UM)) * int(W)))
    iy = int(np.floor((float(y_um) / float(L_PHYS_UM)) * int(H)))
    ix = int(np.clip(ix, 0, int(W) - 1))
    iy = int(np.clip(iy, 0, int(H) - 1))
    return ix, iy


# --------- EXACT Colab viewer versions ---------
def uM_to_bits6(u6, off_uM, tol=TOL_uM):
    u6 = np.asarray(u6, dtype=float)
    return (u6 > (float(off_uM) + float(tol))).astype(int)  # [R,G,B,C,M,Y]


def threshold_rgb8(rgb8, thr):
    r, g, b = [int(x) for x in np.asarray(rgb8).reshape(3,)]
    rT = 0 if r < int(thr) else r
    gT = 0 if g < int(thr) else g
    bT = 0 if b < int(thr) else b
    return np.array([rT, gT, bT], dtype=int)


def rgb8_presence_bits(rgb8_thr):
    rT, gT, bT = [int(x) for x in np.asarray(rgb8_thr).reshape(3,)]
    return np.array([int(rT > 0), int(gT > 0), int(bT > 0)], dtype=int)


def rgb8_to_bits6_binary_cmy_v2(r8, g8, b8, thr=100, res_thr_abs=20, res_thr_rel=0.15):
    """
    EXACT Colab rule (two-stage with residual tests).
    bits = [R,G,B,C,M,Y]
    """
    # Stage 1: presence threshold (keep intensities)
    r = 0 if int(r8) < int(thr) else int(r8)
    g = 0 if int(g8) < int(thr) else int(g8)
    b = 0 if int(b8) < int(thr) else int(b8)

    bits = np.zeros(6, dtype=int)  # [R,G,B,C,M,Y]
    n = int(r > 0) + int(g > 0) + int(b > 0)

    if n == 0:
        return bits

    if n == 1:
        if r > 0:
            bits[0] = 1
        elif g > 0:
            bits[1] = 1
        else:
            bits[2] = 1
        return bits

    def keep_primary(residual, ref):
        return (residual >= res_thr_abs) or (residual >= res_thr_rel * max(ref, 1))

    # n == 2
    if n == 2:
        if (r > 0) and (g > 0):
            bits[5] = 1  # Y
            d = abs(r - g)
            ref = max(r, g)
            if keep_primary(d, ref):
                if r > g:
                    bits[0] = 1
                elif g > r:
                    bits[1] = 1
            return bits

        if (r > 0) and (b > 0):
            bits[4] = 1  # M
            d = abs(r - b)
            ref = max(r, b)
            if keep_primary(d, ref):
                if r > b:
                    bits[0] = 1
                elif b > r:
                    bits[2] = 1
            return bits

        if (g > 0) and (b > 0):
            bits[3] = 1  # C
            d = abs(g - b)
            ref = max(g, b)
            if keep_primary(d, ref):
                if g > b:
                    bits[1] = 1
                elif b > g:
                    bits[2] = 1
            return bits

    # n == 3: CMY always on
    mn = min(r, g, b)
    bits[3] = bits[4] = bits[5] = 1  # C,M,Y

    r_ex = r - mn
    g_ex = g - mn
    b_ex = b - mn
    ref = max(r, g, b)

    if keep_primary(r_ex, ref):
        bits[0] = 1
    if keep_primary(g_ex, ref):
        bits[1] = 1
    if keep_primary(b_ex, ref):
        bits[2] = 1

    return bits
# ----------------------------------------------


def load_bundle_from_dir(bundle_dir: Path):
    bundle_dir = Path(bundle_dir)

    meta_path = bundle_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {bundle_dir}")
    meta = json.loads(meta_path.read_text())

    P_path = bundle_dir / "P_np.npy"
    if not P_path.exists():
        raise FileNotFoundError(f"Missing P_np.npy in {bundle_dir}")
    P_np = np.load(P_path)

    H_npy = bundle_dir / "H_img.npy"
    H_png = bundle_dir / "H_u8.png"
    if H_npy.exists():
        H_img = np.load(H_npy)
    elif H_png.exists():
        H_img = np.asarray(Image.open(H_png).convert("RGB"), dtype=np.uint8)
    else:
        raise FileNotFoundError(f"Missing H_img.npy or H_u8.png in {bundle_dir}")

    table_path = bundle_dir / "instructions_table.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing instructions_table.csv in {bundle_dir}")
    df = pd.read_csv(table_path)

    return meta, P_np, H_img, df


def maybe_unpack_uploaded_zip(upload) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="hydrogel_bundle_"))
    zpath = tmp / "bundle.zip"
    zpath.write_bytes(upload.getbuffer())
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(tmp)

    # If zip contains a folder viewer_bundle/, use it, else use tmp root
    if (tmp / "viewer_bundle" / "meta.json").exists():
        return tmp / "viewer_bundle"
    return tmp


def discover_local_bundles(repo_root: Path) -> list[Path]:
    """
    Detect local bundle directories:
      - viewer_bundle_iterated/
      - viewer_bundle_non_iterated/
      - viewer_bundle/
    Only keep dirs that contain meta.json.
    """
    candidates = []
    for p in repo_root.glob("viewer_bundle*"):
        if p.is_dir() and (p / "meta.json").exists():
            candidates.append(p)

    # Sort with nice preference: iterated first, then non_iterated, then plain
    def key(p: Path):
        name = p.name.lower()
        if "iterated" in name:
            return (0, name)
        if "non_iterated" in name or "noniterated" in name:
            return (1, name)
        return (2, name)

    candidates = sorted(candidates, key=key)
    return candidates


def _col_exact(df: pd.DataFrame, name: str):
    name = name.strip()
    for c in df.columns:
        if str(c).strip() == name:
            return c
    return None


def _pick_coord_col(df: pd.DataFrame, which: str):
    assert which in ("x", "y")
    # exact first
    c = _col_exact(df, f"{which}_um")
    if c is not None:
        return c

    # case-insensitive exact, but reject any column containing 'uM' (capital M)
    target = f"{which}_um".lower()
    for c in df.columns:
        cs = str(c).strip()
        if cs.lower() == target and ("uM" not in cs):
            return c
    return None


def _pick_um_col(df: pd.DataFrame, ch: str):
    c = _col_exact(df, f"{ch}_uM")
    if c is not None:
        return c
    want = f"{ch.lower()}_um"
    for col in df.columns:
        if str(col).strip().lower() == want:
            return col
    return None


def get_target_rgb8_for_gid(gid: int, df: pd.DataFrame, H_img, gid_to_i, x_um, y_um, L_PHYS_UM, W, H):
    gid = int(gid)

    # prefer table targets
    if all(c in df.columns for c in ["target_R8", "target_G8", "target_B8"]):
        row = df[df["GelIndex"] == gid]
        if len(row) == 1:
            r8 = int(row.iloc[0]["target_R8"])
            g8 = int(row.iloc[0]["target_G8"])
            b8 = int(row.iloc[0]["target_B8"])
            return np.array([r8, g8, b8], dtype=int), "from table"

    # fallback: sample from target image
    if gid not in gid_to_i:
        raise KeyError(f"GelIndex {gid} not found")
    i = gid_to_i[gid]
    ix, iy = pix_from_xy_um(float(x_um[i]), float(y_um[i]), L_PHYS_UM, W, H)

    H_arr = np.asarray(H_img)
    if H_arr.dtype == np.uint8:
        rgb8 = H_arr[iy, ix, :3].astype(int)
        return rgb8, "sampled from H_img (uint8)"

    rgb01 = H_arr[iy, ix, :3]
    rgb8 = np.clip(np.rint(rgb01 * 255.0), 0, 255).astype(int)
    return rgb8, "sampled from H_img (float01)"


def main():
    st.set_page_config(page_title="Hydrogel Viewer", layout="wide")
    st.title("Hydrogel instruction viewer")

    repo_root = Path(".")

    with st.sidebar:
        st.header("Bundle source")

        use_uploaded = st.toggle("Upload bundle .zip", value=False)
        upload = None

        bundle_dir = None

        if use_uploaded:
            upload = st.file_uploader("Upload bundle.zip", type=["zip"], accept_multiple_files=False)
            if upload is None:
                st.info("Upload a bundle.zip above.")
                st.stop()

            # Unpack zip, then possibly let user choose inside it if multiple bundles exist
            base_dir = maybe_unpack_uploaded_zip(upload)
            found = discover_local_bundles(base_dir)
            if len(found) >= 2:
                bundle_name = st.selectbox(
                    "Choose bundle inside uploaded zip",
                    options=[p.name for p in found],
                    index=0
                )
                bundle_dir = next(p for p in found if p.name == bundle_name)
            else:
                bundle_dir = base_dir

        else:
            local = discover_local_bundles(repo_root)
            if len(local) >= 2:
                bundle_name = st.selectbox(
                    "Choose local bundle",
                    options=[p.name for p in local],
                    index=0  # iterated tends to come first due to sorting
                )
                bundle_dir = next(p for p in local if p.name == bundle_name)
            elif len(local) == 1:
                bundle_dir = local[0]
                st.caption(f"Using bundle: `{bundle_dir.name}`")
            else:
                # fallback
                bundle_dir = DEFAULT_BUNDLE_DIR
                st.caption(f"Using fallback: `{bundle_dir}`")

        st.divider()
        show_debug = st.toggle("Show debug", value=False)

    # Load selected bundle
    try:
        meta, P_np, H_img, df = load_bundle_from_dir(bundle_dir)
    except Exception as e:
        st.error(str(e))
        st.write("Expected bundle files in selected folder:")
        st.code(
            "meta.json\nP_np.npy\nH_img.npy (or H_u8.png)\ninstructions_table.csv\n"
        )
        st.stop()

    # Meta
    H = int(meta["H"])
    W = int(meta["W"])
    L_PHYS_UM = float(meta["L_PHYS_UM"])
    TEMPLATE_MIN_uM = float(meta.get("TEMPLATE_MIN_uM", 0.01))
    RGB_THR = int(meta.get("RGB_THR", 100))

    # Shapes
    if P_np.shape[:2] != (H, W) or P_np.shape[-1] != 3:
        st.error(f"P_np has shape {P_np.shape}, expected (H,W,3)=({H},{W},3)")
        st.stop()

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    gelcol = _col_exact(df, "GelIndex") or _col_exact(df, "gelindex") or "GelIndex"
    xcol = _pick_coord_col(df, "x")
    ycol = _pick_coord_col(df, "y")
    ucols = [_pick_um_col(df, ch) for ch in ["R", "G", "B", "C", "M", "Y"]]

    missing = []
    if gelcol not in df.columns:
        missing.append("GelIndex")
    if xcol is None:
        missing.append("x_um")
    if ycol is None:
        missing.append("y_um")
    for name, col in zip(["R_uM", "G_uM", "B_uM", "C_uM", "M_uM", "Y_uM"], ucols):
        if col is None:
            missing.append(name)

    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.write("Detected columns:", list(df.columns))
        st.stop()

    # Canonical GelIndex
    df["GelIndex"] = pd.to_numeric(df[gelcol], errors="coerce").astype("Int64")
    df = df.dropna(subset=["GelIndex"]).copy()
    df["GelIndex"] = df["GelIndex"].astype(int)

    gids = df["GelIndex"].to_numpy(dtype=int)
    x_um = df[xcol].astype(float).to_numpy()
    y_um = df[ycol].astype(float).to_numpy()
    u6 = df[ucols].astype(float).to_numpy()

    gid_to_i = {int(gids[i]): i for i in range(len(gids))}
    gid_list = sorted(gid_to_i.keys())

    with st.sidebar:
        st.header("Viewer controls")
        gid = st.selectbox("GelIndex", options=gid_list, index=0)
        zoom = st.slider("Zoom (pixels)", min_value=4, max_value=40, value=12, step=2)
        st.caption(f"Bundle: `{bundle_dir.name}`")
        st.caption(f"Viewer RGB_THR={RGB_THR}")
        st.caption(f"OFF_uM={TEMPLATE_MIN_uM}  (bits ON if uM > OFF+{TOL_uM:g})")

        if show_debug:
            st.write("bundle_dir:", str(bundle_dir))
            st.write("xcol:", xcol, "ycol:", ycol)
            st.write("y min/max:", float(np.min(y_um)), float(np.max(y_um)))

    gid = int(gid)
    i = gid_to_i[gid]
    x = float(x_um[i])
    y = float(y_um[i])
    ix, iy = pix_from_xy_um(x, y, L_PHYS_UM, W, H)

    bits6_from_uM = uM_to_bits6(u6[i, :], off_uM=TEMPLATE_MIN_uM, tol=TOL_uM)
    bits6_str = "".join(str(int(b)) for b in bits6_from_uM)

    rgb8, rgb_src = get_target_rgb8_for_gid(gid, df, H_img, gid_to_i, x_um, y_um, L_PHYS_UM, W, H)
    rgb8_thr = threshold_rgb8(rgb8, thr=RGB_THR)
    rgb_bits = rgb8_presence_bits(rgb8_thr)

    bits6_rule = rgb8_to_bits6_binary_cmy_v2(
        int(rgb8[0]), int(rgb8[1]), int(rgb8[2]),
        thr=RGB_THR, res_thr_abs=20, res_thr_rel=0.15
    )
    match = bool(np.all(bits6_rule == bits6_from_uM))

    sim_rgb01 = np.asarray(P_np)[iy, ix, :]
    sim_rgb8 = np.clip(np.rint(sim_rgb01 * 255.0), 0, 255).astype(int)

    z = int(zoom)
    y0, y1 = max(0, iy - z), min(H, iy + z + 1)
    x0, x1 = max(0, ix - z), min(W, ix + z + 1)

    patch_sim = np.asarray(P_np)[y0:y1, x0:x1, :]
    H_u8 = _to_uint8(H_img)
    H01 = H_u8.astype(np.float32) / 255.0
    patch_tgt = H01[y0:y1, x0:x1, :]

    colA, colB = st.columns([2.2, 1.2], gap="large")

    with colA:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        ax.imshow(P_np, origin="upper", aspect="equal")

        ix_all = np.clip(np.floor((x_um / L_PHYS_UM) * W).astype(int), 0, W - 1)
        iy_all = np.clip(np.floor((y_um / L_PHYS_UM) * H).astype(int), 0, H - 1)

        ax.scatter(ix_all, iy_all, s=14, marker="o", linewidths=0.0, alpha=0.55, zorder=3)
        ax.scatter([ix], [iy], s=180, marker="o", edgecolors="w", linewidths=2.0, zorder=4)

        ax.set_title("Forward-simulated RGB (precomputed)")
        ax.set_xlim(0, W - 1)
        ax.set_ylim(H - 1, 0)

        xt = np.linspace(0, W - 1, 5)
        yt = np.linspace(0, H - 1, 5)
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        ax.set_xticklabels([f"{t*(L_PHYS_UM/(W-1)):.0f}" for t in xt])
        ax.set_yticklabels([f"{t*(L_PHYS_UM/(H-1)):.0f}" for t in yt])
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")

        st.pyplot(fig, clear_figure=True)

    with colB:
        st.subheader("Zoom patches")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            figS, axS = plt.subplots(1, 1, figsize=(3.1, 3.1))
            axS.imshow(patch_sim, origin="upper")
            axS.set_title(f"SIM (+/-{z}px)")
            axS.axis("off")
            st.pyplot(figS, clear_figure=True)
        with c2:
            figT, axT = plt.subplots(1, 1, figsize=(3.1, 3.1))
            axT.imshow(patch_tgt, origin="upper")
            axT.set_title(f"TARGET (+/-{z}px)")
            axT.axis("off")
            st.pyplot(figT, clear_figure=True)

    st.markdown("---")
    st.subheader("Details")

    details_html = (
        f"**Bundle:** `{bundle_dir.name}`<br><br>"
        f"**GelIndex:** {gid}<br><br>"
        f"**Position (µm):** ({x:.2f}, {y:.2f}) &nbsp;&nbsp; "
        f"**Pixel:** (ix={ix}, iy={iy})<br><br>"
        f"**Instructions uM [R,G,B,C,M,Y]:** {np.round(u6[i,:],3).tolist()}<br>"
        f"**Instructions bits6:** {bits6_from_uM.tolist()} &nbsp;&nbsp; "
        f"**string:** {bits6_str}<br><br>"
        f"**Target RGB8 ({rgb_src}):** {rgb8.tolist()}<br>"
        f"**Thresholded RGB8 (thr={RGB_THR}):** {rgb8_thr.tolist()}<br>"
        f"**RGB presence bits [R,G,B]:** {rgb_bits.tolist()}<br>"
        f"**Rule-predicted bits6 [R,G,B,C,M,Y]:** {bits6_rule.tolist()}<br>"
        f"**Matches instructions?** {'YES' if match else 'NO'}<br><br>"
        f"**Simulated RGB at gel (0..1):** {np.round(sim_rgb01,3).tolist()}<br>"
        f"**Simulated RGB8 at gel:** {sim_rgb8.tolist()}"
    )
    st.markdown(details_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
