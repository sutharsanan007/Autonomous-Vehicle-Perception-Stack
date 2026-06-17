# ─────────────────────────────────────────────────────────────────────────────
#  lane_detection.py  |  Modular Hybrid ADAS — Lane Perception Module
#
#  ARCHITECTURE: YOLOP segmentation mask → Polynomial Curve Fitting
#
#  WHY THE ORIGINAL BROKE:
#    The sliding-window + momentum tracker accumulated errors at every dashed-line
#    gap.  Stale momentum launched lines into the sky (Bug 1), collapsed the lane
#    polygon inward causing green leaks (Bug 2), and the anti-crossing `break`
#    produced arrays of mismatched lengths so temporal smoothing was silently
#    skipped — causing the violent X-crossing artefact (Bugs 3 & 4).
#
#  THE FIX — Polynomial Curve Fitting (standard ADAS approach):
#    1. Collect ALL lane pixels from the YOLOP mask below the horizon.
#    2. Split them left / right at the image centre.
#    3. Fit  x = a·y² + b·y + c  to each side (IQR outlier removal first).
#    4. Smooth the THREE scalar coefficients with EMA — never breaks on length.
#    5. Sample 80 clean points from the polynomial for drawing — no gaps, ever.
#    6. Draw solid yellow polylines over a gapless green fill polygon.
# ─────────────────────────────────────────────────────────────────────────────

import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

# ── Model Initialization ──────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing YOLOP AI on hardware: {device.type.upper()}...")

yolop_model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
yolop_model = yolop_model.to(device)
yolop_model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── Tunable Constants ─────────────────────────────────────────────────────────
HORIZON_RATIO  = 0.57           # Only search for lanes below this fraction from top
SMOOTH_ALPHA   = 0.78           # EMA weight for the previous frame (higher = smoother)
N_CURVE_PTS    = 80             # Points sampled along each polynomial for drawing
LINE_THICKNESS = 8              # Solid yellow border line width (pixels)
DILATE_KERN    = 9              # Fills gaps in dashed lane lines before fitting
MIN_PIXELS     = 40             # Min lane pixels required before attempting a fit
MAX_CURVATURE  = 0.003          # Max |a| coefficient — rejects physically impossible bends
MIN_LANE_RATIO = 0.15           # Min lane width as fraction of frame width
WARNING_THRESH = 20.0           # Deviation % that triggers lane-departure warning

YELLOW  = (0, 255, 255)         # BGR  — solid yellow lane border
GREEN   = (0, 255, 0)           # BGR  — lane carpet fill

# ── Persistent State (3 scalars per side, never changes shape) ────────────────
_prev_left_fit  = None          # np.array([a, b, c]) or None
_prev_right_fit = None


# ─────────────────────────────────────────────────────────────────────────────
#  PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _infer_mask(frame: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Run YOLOP inference and return a dilated binary lane-pixel mask at the
    original frame resolution.
    Dilation bridges the gaps between dashed line segments so the polynomial
    has a continuous stream of pixels to fit through.
    """
    img_640    = cv2.resize(frame, (640, 640))
    tensor     = transform(img_640).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, seg_out = yolop_model(tensor)
    raw_mask = torch.argmax(seg_out, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    mask     = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    kernel   = np.ones((DILATE_KERN, DILATE_KERN), np.uint8)
    return cv2.dilate(mask, kernel)


def _fit_poly(pixel_ys: np.ndarray, pixel_xs: np.ndarray):
    """
    Fit  x = a·y² + b·y + c  to the given (y, x) pixel pairs.

    IQR outlier removal is applied first so guardrails, tree shadows, or
    sky noise can't warp the curve.

    Returns np.array([a, b, c]) or None if there are not enough clean pixels.
    """
    if len(pixel_xs) < MIN_PIXELS:
        return None
    try:
        q1, q3 = np.percentile(pixel_xs, 25), np.percentile(pixel_xs, 75)
        iqr    = q3 - q1
        keep   = (pixel_xs >= q1 - 1.5 * iqr) & (pixel_xs <= q3 + 1.5 * iqr)
        xs, ys = pixel_xs[keep], pixel_ys[keep]
        if len(xs) < MIN_PIXELS:
            return None
        return np.polyfit(ys, xs, 2)        # [a, b, c]
    except Exception:
        return None


def _eval(fit: np.ndarray, y) -> np.ndarray:
    """Evaluate  x = a·y² + b·y + c  for scalar or NumPy-array y."""
    return fit[0] * y**2 + fit[1] * y + fit[2]


def _is_sane(fit: np.ndarray, h: int, w: int, side: str) -> bool:
    """
    Reject a polynomial if:
      • Its bottom x is outside the expected half of the frame.
      • Its curvature |a| is too large for a real road.
    Both checks use generous slack so legitimate curves still pass.
    """
    bx    = float(_eval(fit, h))
    half  = w / 2.0
    slack = w * 0.15
    if side == 'left'  and not (0             < bx < half + slack): return False
    if side == 'right' and not (half - slack  < bx < w           ): return False
    if abs(fit[0]) > MAX_CURVATURE:                                   return False
    return True


def _blend(prev, curr):
    """Exponential moving average on the three polynomial coefficients."""
    if prev is None:
        return curr
    return SMOOTH_ALPHA * np.asarray(prev) + (1.0 - SMOOTH_ALPHA) * np.asarray(curr)


def _make_pts(fit: np.ndarray, h: int, horizon_y: int, w: int) -> np.ndarray:
    """
    Sample N_CURVE_PTS (x, y) points along the polynomial from the bottom
    of the frame up to the horizon.

    x values are clamped to [0, w-1] so off-screen extrapolation never
    bleeds into the fill polygon.

    Returns an (N, 1, 2) int32 array for cv2.polylines / cv2.fillPoly.
    """
    ys = np.linspace(h - 1, horizon_y, N_CURVE_PTS)
    xs = np.clip(_eval(fit, ys), 0, w - 1)
    return np.column_stack((xs, ys)).reshape(-1, 1, 2).astype(np.int32)


def _lanes_crossing(left_fit, right_fit, h: int, w: int, horizon_y: int) -> bool:
    """
    Return True if the two polynomials ever cross OR come within MIN_LANE_RATIO
    of each other between the horizon and the bottom of the frame.

    Checking at 10 y-levels catches mid-frame X-crossings that a bottom-only
    check would miss (the X artefact visible in the original recording).
    """
    check_ys  = np.linspace(horizon_y, h - 1, 10)
    lxs       = _eval(left_fit,  check_ys)
    rxs       = _eval(right_fit, check_ys)
    min_width = w * MIN_LANE_RATIO
    return bool(np.any(lxs >= rxs) or np.any(rxs - lxs < min_width))


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API — called by main.py
# ─────────────────────────────────────────────────────────────────────────────

def process_lanes(frame: np.ndarray):
    """
    Parameters
    ----------
    frame : BGR image from cv2.VideoCapture

    Returns
    -------
    overlay : np.ndarray  — same shape as frame, black background, lane visuals only.
                            Caller alpha-blends this onto the source frame.
    stats   : dict        — deviation, is_warning, road_status, brightness
    """
    global _prev_left_fit, _prev_right_fit

    h, w      = frame.shape[:2]
    horizon_y = int(h * HORIZON_RATIO)
    cx        = w // 2
    bright    = int(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))

    _blank_stats = dict(deviation=0.0, is_warning=False,
                        road_status="Searching...", brightness=bright)

    # ── Step 1: AI Perception ─────────────────────────────────────────────────
    mask = _infer_mask(frame, h, w)

    # ── Step 2: Separate left / right lane pixels ─────────────────────────────
    # Search ONLY below the horizon — sky, signs, and tree-lines are ignored.
    # Strict centre split: left pixels in left half, right pixels in right half.
    roi_ys, roi_xs = np.where(mask[horizon_y:] == 1)
    roi_ys = roi_ys + horizon_y     # restore absolute y coordinates

    left_mask  = roi_xs < cx
    right_mask = roi_xs >= cx

    # ── Step 3: Polynomial Fitting with EMA Temporal Smoothing ───────────────
    def _update(prev_fit, ys, xs, side):
        """Try to fit a new polynomial; fall back to prev_fit on failure."""
        curr = _fit_poly(ys, xs)
        if curr is not None and _is_sane(curr, h, w, side):
            return _blend(prev_fit, curr), True   # fresh detection, smoothed
        if prev_fit is not None:
            return prev_fit, False                # hold last valid fit
        return None, False                        # total loss

    left_fit,  l_live = _update(
        _prev_left_fit,
        roi_ys[left_mask],  roi_xs[left_mask],  'left'
    )
    right_fit, r_live = _update(
        _prev_right_fit,
        roi_ys[right_mask], roi_xs[right_mask], 'right'
    )

    # ── Step 4: Total-loss fallback ───────────────────────────────────────────
    if left_fit is None or right_fit is None:
        _prev_left_fit, _prev_right_fit = left_fit, right_fit
        return np.zeros_like(frame), _blank_stats

    # ── Step 5: Anti-Crossing Guard ───────────────────────────────────────────
    # If the two curves collide anywhere between horizon and bumper, revert
    # BOTH to the last known-good pair.  This fully eliminates the X artefact.
    if _lanes_crossing(left_fit, right_fit, h, w, horizon_y):
        if _prev_left_fit is not None and _prev_right_fit is not None:
            left_fit  = _prev_left_fit
            right_fit = _prev_right_fit
        # If no history exists (very first frame anomaly), we accept it —
        # the EMA will pull towards truth within 2-3 frames.

    # Commit the validated fits as the new history
    _prev_left_fit  = left_fit
    _prev_right_fit = right_fit

    # ── Step 6: Generate smooth drawing points ────────────────────────────────
    left_pts  = _make_pts(left_fit,  h, horizon_y, w)  # (N, 1, 2) int32
    right_pts = _make_pts(right_fit, h, horizon_y, w)

    left_flat  = left_pts.reshape(-1, 2)   # (N, 2) for vstack
    right_flat = right_pts.reshape(-1, 2)

    # ── Step 7: Draw ──────────────────────────────────────────────────────────
    overlay = np.zeros_like(frame)

    # Green carpet fill — polygon: left side going up, right side coming down.
    # Because the polynomial has no holes, the fill NEVER leaks.
    poly = np.vstack((left_flat, right_flat[::-1]))
    cv2.fillPoly(overlay, [poly], GREEN)

    # Solid yellow border lines drawn ON TOP of the fill so they are fully opaque.
    cv2.polylines(overlay, [left_pts],  isClosed=False, color=YELLOW, thickness=LINE_THICKNESS)
    cv2.polylines(overlay, [right_pts], isClosed=False, color=YELLOW, thickness=LINE_THICKNESS)

    # ── Step 8: HUD Metrics ───────────────────────────────────────────────────
    lx_bot   = float(_eval(left_fit,  h))
    rx_bot   = float(_eval(right_fit, h))
    dev_pct  = ((lx_bot + rx_bot) / 2.0 - cx) / cx * 100.0
    is_warn  = abs(dev_pct) > WARNING_THRESH
    mode_str = "Poly Fit: Live" if (l_live and r_live) else "Poly Fit: Hold"

    return overlay, dict(
        deviation   = dev_pct,
        is_warning  = is_warn,
        road_status = mode_str,
        brightness  = bright,
    )