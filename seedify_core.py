"""Seedify inference pipeline (self-contained).

Raw photograph -> EXIF orientation -> coarse seed localisation (0.25x) -> 900 x 900 px crop ->
ISNet matting (rembg 'isnet-general-use') -> mask clean-up -> fixed-scale white circle
(R = 340 px at native resolution, 512 px RGBA, paper-ring white balance to 235) ->
ShuffleNetV2 1.0x with scale augmentation (0.6-1.1).

Input checks (no seed / several seeds / seed cut by the frame) and an optional
'unrecognised seed' check (frozen DINOv2 ViT-S/14 features, cosine distance to the
5th nearest neighbour in a bank of the 1200 training seeds; no retraining) are included.

This module reproduces the research pipeline used in the manuscript exactly
(LEGUMINOSE/ANALIZ_ONLY_SEED/scripts s01, s02, s09, s10).
"""
import json
import os
import tempfile

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps

# ---- constants (identical to the research scripts)
SCALE = 0.25            # coarse localisation scale
HALF = 450              # 900 x 900 px crop around the seed
R_CIRCLE = 340          # circle radius at native resolution (px)
OUT = 512               # circle image size (px)
PAPER_REF = np.array([235.0, 235.0, 235.0], np.float32)
MIN_AREA_PX = 3000      # smallest accepted seed mask (native px); smallest training seed ~17 000 px
SECOND_SEED_RATIO = 0.15
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

SPECIES = {"AKASYA": "Robinia pseudoacacia", "ERGUVAN": "Cercis siliquastrum",
           "GLADİÇYA": "Gleditsia triacanthos", "KEÇİBOYNUZU": "Ceratonia siliqua"}
SPECIES_ORDER = ["Cercis siliquastrum", "Ceratonia siliqua", "Gleditsia triacanthos", "Robinia pseudoacacia"]

MSG = {
    "no_seed": "No seed was detected. Photograph a single seed on a plain, light background.",
    "several": "More than one seed was detected. Please photograph a single seed.",
    "edge": "The seed is too close to the camera or extends beyond the frame. "
            "Hold the camera about 8 cm above the seed.",
    "unknown": "Unrecognised seed: this seed does not resemble the four species the application was trained on "
               "(Cercis siliquastrum, Ceratonia siliqua, Gleditsia triacanthos, Robinia pseudoacacia).",
}


# ---------------------------------------------------------------- image helpers
def load_bgr(src):
    """Path, file-like object or PIL image -> BGR uint8 array with EXIF orientation applied."""
    im = src if isinstance(src, Image.Image) else Image.open(src)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def _dark_components(bgr):
    small = cv2.resize(bgr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(g, 255).astype(np.float32) + 1
    m = ((g.astype(np.float32) / bg) < 0.6).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return n, st, m.shape


def coarse_center(bgr):
    n, st, (H, W) = _dark_components(bgr)
    best = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H or a < 30:
            continue
        if best is None or a > st[best][4]:
            best = i
    if best is None:
        return None
    x, y, w, h, _ = st[best]
    return int((x + w / 2) / SCALE), int((y + h / 2) / SCALE)


def count_seeds_coarse(bgr):
    """Areas of dark, non-elongated components not touching the photo border (largest first)."""
    n, st, (H, W) = _dark_components(bgr)
    areas = [int(st[i][4]) for i in range(1, n)
             if st[i][4] >= 30 and max(st[i][2], st[i][3]) <= 4 * min(st[i][2], st[i][3])
             and not (st[i][0] == 0 or st[i][1] == 0 or st[i][0] + st[i][2] >= W or st[i][1] + st[i][3] >= H)]
    return sorted(areas, reverse=True)


def fill_holes(m):
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 1, cv2.FILLED)
    return out


def compose_circle(bgr, alpha_crop, crop_x0, crop_y0, cx, cy, paper_bgr):
    """Seed in a fixed-scale white circle; returns BGRA uint8 (OUT x OUT).

    Works only on the 2R x 2R window around the seed centroid (memory-light); the result is
    identical to white-balancing the full photograph, padding it with white and cropping.
    """
    H, W = bgr.shape[:2]
    P = R_CIRCLE
    cx, cy = int(round(cx)), int(round(cy))
    X0, Y0 = cx - P, cy - P                      # window origin in photo coordinates
    gain = PAPER_REF / np.asarray(paper_bgr, np.float32)
    tile = np.full((2 * P, 2 * P, 3), 255.0, np.float32)
    at = np.zeros((2 * P, 2 * P), np.uint8)
    xs0, ys0, xs1, ys1 = max(X0, 0), max(Y0, 0), min(X0 + 2 * P, W), min(Y0 + 2 * P, H)
    if xs1 > xs0 and ys1 > ys0:
        tile[ys0 - Y0:ys1 - Y0, xs0 - X0:xs1 - X0] = np.clip(bgr[ys0:ys1, xs0:xs1].astype(np.float32) * gain, 0, 255)
    # alpha: crop placed in photo coordinates (clipped to the photo), then windowed
    x0, y0 = int(crop_x0), int(crop_y0)
    ax0, ay0 = max(x0, 0, X0), max(y0, 0, Y0)
    ax1, ay1 = min(x0 + 2 * HALF, W, X0 + 2 * P), min(y0 + 2 * HALF, H, Y0 + 2 * P)
    if ax1 > ax0 and ay1 > ay0:
        at[ay0 - Y0:ay1 - Y0, ax0 - X0:ax1 - X0] = alpha_crop[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
    at = at.astype(np.float32)[..., None] / 255.0
    rgb = tile * at + 255.0 * (1 - at)
    rgb = cv2.resize(rgb, (OUT, OUT), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[:OUT, :OUT]
    rr = np.sqrt((xx - (OUT - 1) / 2) ** 2 + (yy - (OUT - 1) / 2) ** 2)
    circ = np.clip((OUT / 2 - rr) + 0.5, 0, 1)
    rgb = rgb * circ[..., None] + 255.0 * (1 - circ[..., None])
    return np.dstack([rgb, circ * 255]).round().clip(0, 255).astype(np.uint8)


def on_white_224(rgba, scale=1.0):
    """PIL RGBA circle image -> 224 x 224 RGB uint8 on white (optionally shrunk, for the OOD bank)."""
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, rgba.convert("RGBA")).convert("RGB")
    if scale != 1.0:
        W = im.size[0]
        s = im.resize((int(round(W * scale)),) * 2, Image.BICUBIC)
        canvas = Image.new("RGB", im.size, (255, 255, 255))
        o = (W - s.size[0]) // 2
        canvas.paste(s, (o, o))
        im = canvas
    return np.asarray(im.resize((224, 224), Image.BICUBIC)).copy()


def _norm(x):
    return (x - MEAN) / STD


# ---------------------------------------------------------------- ISNet session
def make_isnet_session(threads=2):
    """ISNet ('isnet-general-use') rembg session with memory-light ONNX Runtime options.

    The session class is instantiated directly because rembg.new_session() changed its signature
    between versions (2.0.66-2.0.75 build their own SessionOptions and reject a `sess_opts` argument).
    The model file and its pre/post-processing are identical across these versions.
    """
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False   # ~0.7 GB lower peak memory, identical masks
    so.enable_mem_pattern = False
    try:
        from rembg.sessions import sessions_class
        cls = next(sc for sc in sessions_class if sc.name() == "isnet-general-use")
        return cls("isnet-general-use", so)
    except Exception:
        from rembg import new_session
        try:
            return new_session("isnet-general-use", sess_opts=so)
        except TypeError:
            os.environ.setdefault("OMP_NUM_THREADS", str(threads))
            return new_session("isnet-general-use")


# ---------------------------------------------------------------- predictor
class SeedifyPredictor:
    def __init__(self, weights, ood_dir=None, ood_level="default", dinov2_file=None, threads=2):
        torch.set_num_threads(threads)
        import torchvision
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        self.classes = ck["classes"]
        m = torchvision.models.shufflenet_v2_x1_0(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(self.classes))
        m.load_state_dict(ck["state_dict"])
        self.model = m.eval()

        self.session = make_isnet_session(threads)

        self.ood = None
        self.ood_error = None
        if ood_dir:
            info = json.load(open(os.path.join(ood_dir, "ood_threshold.json")))
            bank = np.load(os.path.join(ood_dir, "ood_bank.npz"))["feats"].astype(np.float32)
            thr = info["threshold_default"] if ood_level == "default" else info["threshold_lenient"]
            import timm
            kw = dict(pretrained=True, num_classes=0, img_size=int(info.get("img_size", 224)))
            if dinov2_file and os.path.exists(dinov2_file):
                kw["pretrained_cfg_overlay"] = dict(file=dinov2_file)
            try:
                backbone = timm.create_model(info["backbone_id"], **kw).eval()
                self.ood = dict(backbone=backbone, bank=bank, thr=float(thr), k=int(info["k"]))
            except Exception as e:  # e.g. no internet to download the DINOv2 weights
                self.ood_error = f"{type(e).__name__}: {e}"

    # -- segmentation
    def segment(self, bgr):
        from rembg import remove
        H, W = bgr.shape[:2]
        cx, cy = coarse_center(bgr) or (W // 2, H // 2)
        P = HALF
        pad = cv2.copyMakeBorder(bgr, P, P, P, P, cv2.BORDER_REPLICATE)
        roi = pad[cy:cy + 2 * P, cx:cx + 2 * P]
        a = np.array(remove(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)), session=self.session, only_mask=True))
        m = (a > 127).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        comps = (n, lab, st)
        if n < 2:
            return None, None, comps
        m = fill_holes((lab == 1 + int(np.argmax(st[1:, 4]))).astype(np.uint8))
        keep = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        alpha = (a.astype(np.float32) * keep).astype(np.uint8)
        alpha[m > 0] = np.maximum(alpha[m > 0], 255 * (cv2.erode(m, np.ones((5, 5), np.uint8))[m > 0]))
        ys, xs = np.nonzero(m)
        d1 = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (241, 241)))
        d0 = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))
        paper = np.median(roi[(d1 > 0) & (d0 == 0)].reshape(-1, 3), axis=0)
        geo = dict(crop_x0=cx - P, crop_y0=cy - P, cx=cx - P + xs.mean(), cy=cy - P + ys.mean(), paper_bgr=paper)
        return alpha, geo, comps

    @staticmethod
    def _check_components(comps):
        n, lab, st = comps
        areas = sorted(st[1:, 4].tolist(), reverse=True)
        if not areas or areas[0] < MIN_AREA_PX:
            return "no_seed"
        if len(areas) > 1 and areas[1] >= SECOND_SEED_RATIO * areas[0]:
            return "several"
        big = (lab == 1 + int(np.argmax(st[1:, 4]))).astype(np.uint8)
        if big[0].any() or big[-1].any() or big[:, 0].any() or big[:, -1].any():
            return "edge"
        return None

    # -- classification
    @torch.no_grad()
    def classify(self, rgba):
        x = torch.from_numpy(on_white_224(rgba)).permute(2, 0, 1)[None].float() / 255
        p = torch.softmax(self.model(_norm(x)), 1)[0].numpy()
        probs = {SPECIES.get(c, c): float(v) for c, v in zip(self.classes, p)}
        return max(probs, key=probs.get), probs

    @torch.no_grad()
    def ood_score(self, rgba):
        x = torch.from_numpy(on_white_224(rgba)).permute(2, 0, 1)[None].float() / 255
        f = self.ood["backbone"](_norm(x)).numpy()
        f = f / np.linalg.norm(f, axis=1, keepdims=True)
        sim = f @ self.ood["bank"].T
        kth = -np.partition(-sim, self.ood["k"] - 1, axis=1)[:, self.ood["k"] - 1]
        return float(1.0 - kth[0])

    # -- full analysis
    def analyze(self, src):
        """Returns dict(status in {'ok','unknown','invalid'}, message, label, probs, circle (PIL RGBA), ood_score)."""
        bgr = load_bgr(src)
        res = dict(status="invalid", message=None, label=None, probs=None, circle=None, ood_score=None)
        alpha, geo, comps = self.segment(bgr)
        if alpha is None:
            return dict(res, message=MSG["no_seed"])
        problem = self._check_components(comps)
        if problem:
            return dict(res, message=MSG[problem])
        coarse = count_seeds_coarse(bgr)
        if len(coarse) > 1 and coarse[1] >= SECOND_SEED_RATIO * coarse[0]:
            return dict(res, message=MSG["several"])
        out = compose_circle(bgr, alpha, **geo)
        rgba = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGRA2RGBA))
        label, probs = self.classify(rgba)
        res.update(status="ok", label=label, probs=probs, circle=rgba, message=f"{label} (probability {probs[label]:.2f})")
        if self.ood is not None:
            sc = self.ood_score(rgba)
            res.update(ood_score=sc, ood_threshold=self.ood["thr"])
            if sc > self.ood["thr"]:
                res.update(status="unknown", message=MSG["unknown"])
        return res
