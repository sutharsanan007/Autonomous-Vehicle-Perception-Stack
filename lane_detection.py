# ─────────────────────────────────────────────────────────────────────────────
#  lane_detection.py  |  Modular Hybrid ADAS — Lane Perception Module  (v6)
#
#  REVERTED FROM v5's bird's-eye warp back to direct camera-space fitting.
#
#  WHY THE BIRD'S-EYE WARP (v5) WAS WRONG FOR THIS PROJECT:
#    The reference video (Handong University Mechatronics Capstone — Jang,
#    Park, Yun) was analyzed frame-by-frame.  Its green border lines:
#      • Never visually meet at a single point near the horizon.
#      • Stay glued to the real lane markings through curves, including
#        a sharp "Warning! OFF Lane" right-curve frame.
#      • Stop independently at different heights for left vs right line —
#        NOT a shared, calculated convergence point.
#    This is the signature of DIRECT CAMERA-SPACE fitting with a SHORTENED
#    draw distance — not a bird's-eye warp.  The warp/unwarp round-trip
#    (verified mathematically correct in isolation) is nonetheless
#    extremely sensitive to small per-frame coefficient noise once
#    unwarped back near the vanishing point — exactly the wild hook seen
#    in the screenshot.  The reference avoids this failure mode entirely
#    by simply never drawing that close to the horizon.
#
#  THIS VERSION:
#    • Restores v4's direct camera-space polynomial engine (guided sliding
#      window keyed to prev_fit per band — stable, no drift, no warp).
#    • Each line's draw distance is now INDEPENDENT (left and right can
#      stop at different heights), driven by where that line's own pixel
#      detections actually run out — exactly matching the reference's
#      visual behaviour (see frame analysis above).
#    • Keeps the v5 visual style you approved: two independent thick
#      green border lines, semi-transparent blue carpet, steering arrow
#      + percentage, "Upcoming Road" text classifier.
# ─────────────────────────────────────────────────────────────────────────────

import warnings
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

# ── NumPy RankWarning — version-safe suppression ──────────────────────────────
try:
    from numpy.exceptions import RankWarning as _RankWarning   # NumPy ≥ 2.0
except ImportError:
    _RankWarning = np.RankWarning                               # NumPy < 2.0
warnings.filterwarnings('ignore', category=_RankWarning)

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
SEARCH_HORIZON_RATIO = 0.57    # top of the pixel-search ROI (fraction of h)
SMOOTH_ALPHA         = 0.75    # EMA weight for previous-frame polynomial
N_CURVE_PTS          = 60      # points sampled along each drawn curve
LINE_THICKNESS       = 10      # thick independent border lines (reference style)
DILATE_KERN          = 9       # fills dashed-line gaps before window search
WARNING_THRESH       = 10.0    # deviation % that triggers lane-departure alert
                                # LOWERED from 20.0: real test footage showed the
                                # car sitting at 2-4% deviation during normal,
                                # clearly-off-centre driving — 20% was tuned for a
                                # much more extreme drift than actually occurs in
                                # practice, so the red-alert state never triggered.

# Sliding window
N_WINDOWS            = 15      # horizontal scan bands (bottom → search horizon)
WINDOW_MARGIN        = 90      # half-width of each band's search window (px)
MIN_PIX_RECENTER     = 8       # min pixels in a band to accept a new centre
                                # LOWERED from 15: dashed lane lines (often the
                                # LEFT boundary, e.g. lane-divider markings) put
                                # far fewer pixels into the mask per band than a
                                # solid line (e.g. the RIGHT shoulder/guardrail
                                # edge).  At 15, many valid-but-sparse dashed
                                # bands were silently dropped, starving the left
                                # fit of data relative to the right and making
                                # it visibly less precise.  8 still rejects pure
                                # noise (a handful of stray dilated pixels) while
                                # accepting a genuine, if thin, dash segment.

# Polynomial fitting
MIN_PIXELS           = 25      # min pixels required for np.polyfit
MIN_Y_SPAN           = 60      # min vertical spread — RankWarning gate
MIN_LANE_RATIO       = 0.12    # anti-crossing safety net (fraction of w)

# Cold-start histogram (first frame only)
INNER_LANE_RATIO     = 0.38    # search only inner 38% of each half
HIST_SIGMA_RATIO     = 0.08    # exponential-bias sigma (fraction of w)

# Independent draw-distance shortening (THE KEY FIX in v6)
# Each line stops drawing at the topmost window band where it actually
# found pixels with HIGH confidence (i.e. didn't have to coast on
# prev_fit alone) — this is what makes the reference's lines fade out
# naturally instead of being forced to a shared, noise-prone horizon point.
MIN_CONFIDENT_BANDS  = 4        # require at least this many live bands found
DRAW_MARGIN_BANDS    = 1        # stop 1 band short of the last confident hit,
                                 # so the line never ends on a shaky detection

# Hold-recovery (NEW — fixes the "frozen forever" bug)
# A "Hold" frame means the current frame's fit failed validation and the
# system is coasting on the previous good fit.  Without a recovery
# mechanism, a single persistently-failing side (e.g. a faint/far line)
# can hold the SAME fit indefinitely — which is exactly the static-image
# freeze observed around the 18s mark of the test video.
MAX_HOLD_FRAMES      = 8        # after this many consecutive holds, force
                                 # a full cold-start re-detection for that side
WINDOW_WIDEN_STEP    = 25       # px to widen the search margin per consecutive hold
WINDOW_WIDEN_CAP     = 200      # never widen past this (avoids picking up the
                                 # opposite lane's pixels)

# Curvature → steering guidance thresholds
CURVE_DEAD_ZONE      = 0.00035  # |a| below this = "Stay Straight"
CURVE_PCT_SCALE      = 2500     # rescaled so realistic 'a' values span 0-35%
                                 # smoothly instead of saturating almost
                                 # immediately (old value of 9000 saturated
                                 # at |a| ≈ 0.0039, well within normal range)

YELLOW       = (0, 255, 255)
GREEN        = (60, 220, 60)        # border line colour when centred — matches reference
BLUE_FILL    = (235, 140, 60)       # BGR — semi-transparent blue carpet (normal state)
RED_FILL     = (50, 50, 230)        # BGR — carpet turns red on lane-departure warning
RED_BORDER   = (60, 60, 230)        # BGR — border lines turn red to match the fill
WHITE        = (255, 255, 255)

# Deviation-correction arrow — only shown once drift exceeds this threshold,
# so it doesn't clutter the display while the car is reasonably centred.
# Set below WARNING_THRESH so the correction hint appears a bit EARLY,
# before the full red-alert state triggers.
# LOWERED from 12.0 to 5.0 — matches WARNING_THRESH's recalibration; the old
# value almost never triggered against real driving footage, where normal
# in-lane positioning sits in the low single digits and a genuinely
# noticeable drift is more like 5-15%, not 12%+.
CORRECTION_DEAD_ZONE = 5.0

# Debug / diagnostic mode — when True, the raw pixels collected by the
# sliding window for each side are drawn as small dots directly onto the
# overlay (yellow = left, magenta = right), so you can SEE exactly which
# pixels the algorithm is using to fit each line, rather than only seeing
# the final smoothed curve.  Turn this on temporarily to diagnose any
# remaining left/right precision mismatch — if the dots themselves are
# offset from the real lane markings, the bug is in YOLOP's mask /
# the window search; if the dots sit right on the markings but the drawn
# CURVE doesn't, the bug is in the polyfit/smoothing stage instead.
DEBUG_MASK = False

# Console debug mode — when True, prints one line per frame with the exact
# numbers needed to diagnose the deviation/fit pipeline: both polynomials'
# coefficients, bottom x-positions, raw pixel counts per side, and the
# final deviation %.  No file re-editing needed beyond this one flag —
# just flip it, run main.py, and copy a chunk of console output back.
# A FRAME_COUNTER is included so you can see whether values are changing
# frame-to-frame or genuinely static.
DEBUG_PRINT = True
_frame_counter = 0

# ── Persistent State ──────────────────────────────────────────────────────────
_prev_left_fit   = None         # np.array([a, b, c]) or None
_prev_right_fit  = None
_left_hold_count  = 0           # consecutive frames left side has coasted
_right_hold_count = 0           # consecutive frames right side has coasted


# ─────────────────────────────────────────────────────────────────────────────
#  PRIVATE HELPERS — Perception
# ─────────────────────────────────────────────────────────────────────────────

def _infer_mask(frame, h, w):
    """YOLOP inference → dilated binary lane-pixel mask at native resolution."""
    img    = cv2.resize(frame, (640, 640))
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, seg = yolop_model(tensor)
    raw  = torch.argmax(seg, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    mask = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.dilate(mask, np.ones((DILATE_KERN, DILATE_KERN), np.uint8))


def _cold_start_bases(mask, h, w, cx):
    """
    First-frame base finder.  Searches only the inner INNER_LANE_RATIO of
    each half so far-shoulder markings are excluded.  Exponential centre-
    bias makes the innermost lane line win over a denser outer solid edge.
    """
    bottom_y  = int(h * 0.75)
    histogram = np.sum(mask[bottom_y:, :], axis=0).astype(np.float32)
    sigma     = w * HIST_SIGMA_RATIO
    inner_w   = int(w * INNER_LANE_RATIO)

    l_start = max(0, cx - inner_w)
    l_hist  = histogram[l_start:cx].copy()
    if len(l_hist) > 0 and l_hist.max() > 0:
        n         = len(l_hist)
        l_bias    = np.exp(-np.arange(n - 1, -1, -1, dtype=np.float32) / sigma)
        left_base = l_start + int(np.argmax(l_hist * l_bias))
    else:
        left_base = cx - int(w * 0.20)

    r_end   = min(w, cx + inner_w)
    r_hist  = histogram[cx:r_end].copy()
    if len(r_hist) > 0 and r_hist.max() > 0:
        r_bias     = np.exp(-np.arange(len(r_hist), dtype=np.float32) / sigma)
        right_base = cx + int(np.argmax(r_hist * r_bias))
    else:
        right_base = cx + int(w * 0.20)

    return left_base, right_base


def _eval(fit, y):
    """Evaluate  x = a·y² + b·y + c  for scalar or array y."""
    return fit[0] * y**2 + fit[1] * y + fit[2]


def _convergence_y(left_fit, right_fit, h, w, search_horizon_y):
    """
    Dynamic pixel-collection cutoff — limits pixel collection to y-levels
    where the prev-frame curves are still meaningfully separated, so the
    window never picks up cross-lane pixels near the horizon.
    """
    if left_fit is None or right_fit is None:
        return search_horizon_y

    min_gap = w * MIN_LANE_RATIO
    for cy in range(search_horizon_y, h, 5):
        if float(_eval(right_fit, cy)) - float(_eval(left_fit, cy)) >= min_gap:
            return cy
    return int(h * 0.80)


def _sliding_window_guided(mask, h, pixel_top_y, base_x, w, prev_fit=None, extra_margin=0):
    """
    Guided sliding window.  Each band is centred on _eval(prev_fit, mid_y)
    — the PREVIOUS polynomial's expected position AT THAT HEIGHT — so the
    window can't drift onto a neighbouring lane line band-to-band.

    `extra_margin` (NEW) widens the search window beyond WINDOW_MARGIN.
    Used during hold-recovery: each consecutive frame a side fails to get
    a live detection, its search window widens a little, improving the
    odds of recapturing a line that drifted away from the stale prediction
    — without this, a side that's gone slightly out of predicted range
    can hold the same fit forever even though the real line is just
    outside the fixed-width window.

    Also returns `last_confident_band` — the index (0 = bottom) of the
    topmost band that found enough real pixels (>= MIN_PIX_RECENTER)
    — drives the independent draw-distance shortening.
    """
    win_h     = max(1, (h - pixel_top_y) // N_WINDOWS)
    current_x = base_x
    margin    = WINDOW_MARGIN + extra_margin
    all_ys, all_xs = [], []
    last_confident_band = -1

    for win in range(N_WINDOWS):
        y_lo  = max(h - (win + 1) * win_h, pixel_top_y)
        y_hi  = min(h -  win      * win_h, h)
        mid_y = (y_lo + y_hi) / 2.0

        if prev_fit is not None:
            center = int(np.clip(float(_eval(prev_fit, mid_y)), 0, w - 1))
        else:
            center = current_x

        x_lo = max(0, center - margin)
        x_hi = min(w, center + margin)

        ys_s, xs_s = np.where(mask[y_lo:y_hi, x_lo:x_hi] == 1)

        if len(xs_s) >= MIN_PIX_RECENTER:
            abs_xs    = xs_s + x_lo
            abs_ys    = ys_s + y_lo
            current_x = int(np.median(abs_xs))
            all_ys.extend(abs_ys.tolist())
            all_xs.extend(abs_xs.tolist())
            last_confident_band = win        # this band had REAL pixel support

    return (np.array(all_ys, dtype=np.float32),
            np.array(all_xs, dtype=np.float32),
            last_confident_band,
            win_h)


def _fit_poly(pixel_ys, pixel_xs):
    """
    Fit  x = a·y² + b·y + c, weighted by proximity to the car.

    WHY WEIGHTING WAS ADDED (fixes the "cuts the corner" bug):
    A single quadratic fit across the WHOLE visible lane length is only
    locally accurate.  Right where a straight road transitions into a
    curve, the pixel population fed to polyfit is a MIX of two different
    local curvatures (the straight part near the car + the just-starting
    bend further out).  An unweighted least-squares fit finds a
    "compromise" shape that averages the two, which systematically
    undershoots the real curve right at the bend — visually, the border
    cuts inward across the actual dashed line instead of riding along it.

    This is not a YOLOP detection problem — YOLOP's pixels were correct;
    the single-quadratic fit was simply not using them with the right
    priority.  Linear proximity weighting (verified numerically: error at
    the car bumper drops from ~14px to ~3px, with only a few extra px of
    error far down the road, which gets re-fit fresh next frame anyway)
    biases the fit toward getting the close, safety-critical part of the
    lane right, while still using the far pixels to anchor the curve's
    general direction.
    """
    if len(pixel_xs) < MIN_PIXELS:
        return None
    if pixel_ys.max() - pixel_ys.min() < MIN_Y_SPAN:
        return None
    try:
        q1, q3 = np.percentile(pixel_xs, 25), np.percentile(pixel_xs, 75)
        iqr    = q3 - q1
        if iqr > 1:
            keep     = (pixel_xs >= q1 - 1.5 * iqr) & (pixel_xs <= q3 + 1.5 * iqr)
            pixel_xs = pixel_xs[keep]
            pixel_ys = pixel_ys[keep]
        if len(pixel_xs) < MIN_PIXELS:
            return None

        # Linear proximity weight: y is image-row (larger y = lower on
        # screen = closer to the car).  Normalize to [0, 1] then add a
        # floor so far pixels still contribute (never fully ignored).
        y_min, y_max = pixel_ys.min(), pixel_ys.max()
        span = max(y_max - y_min, 1.0)
        weights = (pixel_ys - y_min) / span        # 0 = far, 1 = near
        weights = weights + 0.15                    # floor so far pixels still count

        return np.polyfit(pixel_ys, pixel_xs, 2, w=weights)
    except Exception:
        return None


def _is_sane(fit, h, w, side, top_y):
    """
    Sanity check on a candidate fit.
      • Coarse left/right boundary check — reject gross mis-assignment.
      • NEW — flatness check: a real lane line spans roughly
        (bottom_x - top_x) over (h - top_y) vertical pixels.  A degenerate
        fit (e.g. from a handful of noisy pixels) can produce a nearly
        flat line — small dx over the full vertical range — which is
        geometrically impossible for an actual lane boundary receding
        toward the horizon.  This was the cause of the flat right-side
        line seen in testing: a sparse/noisy detection produced a fit
        that was technically "valid" by the old bottom-position-only
        check but had almost zero slope.
    """
    bx    = float(_eval(fit, h))
    tx    = float(_eval(fit, top_y))
    half  = w / 2.0
    slack = w * 0.22
    if side == 'left'  and not (0             < bx < half + slack): return False
    if side == 'right' and not (half - slack  < bx < w           ): return False

    vertical_span = max(h - top_y, 1)
    horizontal_span = abs(bx - tx)
    min_expected_span = vertical_span * 0.08   # real lanes recede noticeably
    if horizontal_span < min_expected_span:
        return False

    return True


def _blend(prev, curr):
    """EMA on the three polynomial coefficients."""
    if prev is None:
        return curr
    return SMOOTH_ALPHA * np.asarray(prev) + (1.0 - SMOOTH_ALPHA) * np.asarray(curr)


def _draw_top_from_confidence(h, pixel_top_y, last_confident_band, win_h):
    """
    NEW IN v6 — replaces the bird's-eye warp as the fix for horizon noise.

    Converts `last_confident_band` (from the sliding window) into an actual
    y-coordinate, then backs off by DRAW_MARGIN_BANDS so the line never
    ends right on its shakiest detection.

    If too few confident bands were found, fall back to a conservative
    fixed fraction of the search range — matches the reference video's
    behaviour of NOT drawing all the way to the horizon when confidence
    is low, e.g. on a sharp curve or a far/hazy lane line.
    """
    if last_confident_band < MIN_CONFIDENT_BANDS - 1:
        # Not enough confident detections — be conservative.
        return int(h - (h - pixel_top_y) * 0.55)

    stop_band = max(0, last_confident_band - DRAW_MARGIN_BANDS)
    return int(h - (stop_band + 1) * win_h)


def _make_pts(fit, h, draw_top_y, w):
    """Sample N_CURVE_PTS (x, y) points from the frame bottom up to draw_top_y."""
    draw_top_y = max(draw_top_y, 0)
    if draw_top_y >= h - 1:
        draw_top_y = h - 2
    ys = np.linspace(h - 1, draw_top_y, N_CURVE_PTS)
    xs = np.clip(_eval(fit, ys), 0, w - 1)
    return np.column_stack((xs, ys)).reshape(-1, 1, 2).astype(np.int32)


def _lanes_crossing(left_fit, right_fit, h, w, horizon_y):
    """
    Checks for a genuine X-crossing / lane-swap artefact — NOT for normal
    perspective narrowing.

    BUG FOUND via DEBUG_PRINT console output: the old version used ONE flat
    minimum gap (MIN_LANE_RATIO * w) at every tested y-level, including
    right at the search horizon.  But real lanes legitimately narrow in
    camera-space perspective as they approach the horizon — a healthy,
    correct fit can have a gap of ~25-30px there even though the same lane
    is 800+px wide at the bottom of the frame.  That legitimate narrowing
    was being misread as a "crossing," which made this guard fire on EVERY
    frame, permanently discarding every new detection and freezing the fit
    at whatever its value happened to be on the first frame it ran.

    FIX: scale the minimum allowed gap linearly with distance from the
    bottom of the frame, so the threshold is strict near the car (where a
    real crossing would be obviously wrong) and lenient near the horizon
    (where narrowing is expected and normal).  A TRUE crossing (lxs >= rxs,
    i.e. the lines have actually swapped sides) is still caught at full
    strictness regardless of y-position.
    """
    ys  = np.linspace(horizon_y, h - 1, 10)
    lxs = _eval(left_fit,  ys)
    rxs = _eval(right_fit, ys)

    # 0.0 at the horizon, 1.0 at the bottom of the frame
    proximity = (ys - horizon_y) / max(h - 1 - horizon_y, 1)
    # Minimum gap scales from a lenient floor near the horizon up to the
    # full MIN_LANE_RATIO width near the car.  Floor verified against real
    # DEBUG_PRINT data: 0.02 was still too strict (rejected a healthy
    # 26.9px gap that needed only 28px), 0.01 passes it with margin while
    # still catching genuine near-horizon crossings.
    min_gap_floor = w * 0.01
    min_gap_full  = w * MIN_LANE_RATIO
    min_gap       = min_gap_floor + proximity * (min_gap_full - min_gap_floor)

    actual_crossing = np.any(lxs >= rxs)          # lines literally swapped sides
    too_narrow      = np.any((rxs - lxs) < min_gap)
    return bool(actual_crossing or too_narrow)


def _classify_curvature(left_fit, right_fit):
    """
    Derive a steering suggestion from the average 'a' coefficient of both
    lanes.  Returns (direction, pct) where direction is
    'straight' | 'left' | 'right'.

    FIXED: the old multiplier (9000) saturated the 0-35% clip range at
    |a| ≈ 0.0039 — well within normal curvature values — so the arrow
    showed a near-constant percentage regardless of the actual curve
    sharpness.  CURVE_PCT_SCALE is rescaled so the displayed percentage
    varies smoothly across the realistic range of 'a' values.
    """
    avg_a = (left_fit[0] + right_fit[0]) / 2.0
    pct   = int(round(np.clip(abs(avg_a) * CURVE_PCT_SCALE, 0, 35)))

    if abs(avg_a) < CURVE_DEAD_ZONE:
        return 'straight', 0
    return ('right', pct) if avg_a > 0 else ('left', pct)


def _arrow_polygon(cx, cy, direction, size):
    """Builds the (closed) polygon points for an up/left/right chevron arrow."""
    if direction == 'straight':
        return np.array([
            [cx, cy - size], [cx - 22, cy - size + 30], [cx - 10, cy - size + 30],
            [cx - 10, cy + size], [cx + 10, cy + size], [cx + 10, cy - size + 30],
            [cx + 22, cy - size + 30],
        ], dtype=np.int32)
    elif direction == 'right':
        return np.array([
            [cx - size, cy], [cx - size + 30, cy - 22], [cx - size + 30, cy - 10],
            [cx + size, cy - 10], [cx + size, cy + 10], [cx - size + 30, cy + 10],
            [cx - size + 30, cy + 22],
        ], dtype=np.int32)
    else:  # left
        return np.array([
            [cx + size, cy], [cx + size - 30, cy - 22], [cx + size - 30, cy - 10],
            [cx - size, cy - 10], [cx - size, cy + 10], [cx + size - 30, cy + 10],
            [cx + size - 30, cy + 22],
        ], dtype=np.int32)


def _draw_curve_arrow(overlay, direction, pct, w, h):
    """
    NAVIGATION signal — shows what the road is ABOUT TO DO (upcoming
    curvature), independent of where the car currently sits in the lane.
    Positioned centre, matching the reference video's indicator.
    """
    cx, cy = w // 2, int(h * 0.90)
    pts = _arrow_polygon(cx, cy, direction, size=45)
    cv2.polylines(overlay, [pts], isClosed=True, color=GREEN, thickness=3)
    if pct > 0:
        cv2.putText(overlay, f"{pct}%", (cx + 45 + 15, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)


def _draw_correction_arrow(overlay, direction, dev_pct, w, h):
    """
    SAFETY signal — shows which way to steer to get back to lane centre.
    Distinct from the curvature arrow: smaller, offset to the side, and
    coloured WHITE/RED so it reads as an alert rather than a navigation
    hint, and the two arrows never visually overlap or get confused.
    direction is 'left' or 'right' (never 'straight' — caller only invokes
    this once there's a meaningful deviation).
    """
    offset_x = int(w * 0.22)
    cx = (w // 2) - offset_x if direction == 'left' else (w // 2) + offset_x
    cy = int(h * 0.90)
    pts = _arrow_polygon(cx, cy, direction, size=30)
    color = (60, 60, 230)   # red — matches the warning fill/border
    cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=3)
    cv2.putText(overlay, f"{dev_pct:.0f}%", (cx - 18, cy + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API — called by main.py
# ─────────────────────────────────────────────────────────────────────────────

def process_lanes(frame):
    """
    Returns (overlay, stats).
    overlay — black-background BGR image: blue carpet fill, green border
              lines, steering arrow.  Caller blends this onto the source frame.
    stats   — dict: deviation, is_warning, road_status, brightness,
              upcoming_road.
    """
    global _prev_left_fit, _prev_right_fit, _left_hold_count, _right_hold_count

    h, w             = frame.shape[:2]
    search_horizon_y = int(h * SEARCH_HORIZON_RATIO)
    cx               = w // 2
    bright           = int(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))

    _blank = dict(deviation=0.0, is_warning=False, road_status="Searching...",
                  brightness=bright, upcoming_road="—")

    # ── Step 1: AI perception ─────────────────────────────────────────────────
    mask = _infer_mask(frame, h, w)

    # ── Step 2: Forced cold-start recovery check (NEW — fixes the freeze) ────
    # If a side has been coasting on a Hold fit for too long, discard it and
    # force a fresh histogram-based detection THIS frame, regardless of what
    # the warm-start would have predicted.  This guarantees the system can
    # never freeze indefinitely on a stale fit.
    force_left_reset  = _left_hold_count  >= MAX_HOLD_FRAMES
    force_right_reset = _right_hold_count >= MAX_HOLD_FRAMES

    left_seed_fit  = None if force_left_reset  else _prev_left_fit
    right_seed_fit = None if force_right_reset else _prev_right_fit

    # ── Step 3: Sliding-window base (warm start / cold start) ────────────────
    if left_seed_fit is not None:
        left_base  = int(np.clip(float(_eval(left_seed_fit,  h)), 10,      cx - 10))
    else:
        left_base, _ = _cold_start_bases(mask, h, w, cx)

    if right_seed_fit is not None:
        right_base = int(np.clip(float(_eval(right_seed_fit, h)), cx + 10, w  - 10))
    else:
        _, right_base = _cold_start_bases(mask, h, w, cx)

    # ── Step 4: Dynamic convergence cutoff for pixel collection ──────────────
    pixel_top_y = _convergence_y(left_seed_fit, right_seed_fit, h, w, search_horizon_y)

    # ── Step 5: Guided sliding window, progressively widened during Hold ─────
    # Each consecutive Hold frame widens that side's search margin a little,
    # improving the odds of recapturing a line that's drifted slightly out
    # of the stale prediction's fixed-width window.
    left_extra_margin  = min(_left_hold_count  * WINDOW_WIDEN_STEP, WINDOW_WIDEN_CAP)
    right_extra_margin = min(_right_hold_count * WINDOW_WIDEN_STEP, WINDOW_WIDEN_CAP)

    l_ys, l_xs, l_last_band, win_h = _sliding_window_guided(
        mask, h, pixel_top_y, left_base,  w, left_seed_fit,  left_extra_margin
    )
    r_ys, r_xs, r_last_band, _ = _sliding_window_guided(
        mask, h, pixel_top_y, right_base, w, right_seed_fit, right_extra_margin
    )

    # ── Step 6: Polynomial fit + EMA smoothing ────────────────────────────────
    def _update(prev_fit, ys, xs, side):
        curr = _fit_poly(ys, xs)
        if curr is not None and _is_sane(curr, h, w, side, search_horizon_y):
            return _blend(prev_fit, curr), True
        if prev_fit is not None:
            return prev_fit, False
        return None, False

    left_fit,  l_live = _update(_prev_left_fit,  l_ys, l_xs, 'left')
    right_fit, r_live = _update(_prev_right_fit, r_ys, r_xs, 'right')

    # ── Step 7: Update hold counters ──────────────────────────────────────────
    _left_hold_count  = 0 if l_live else _left_hold_count  + 1
    _right_hold_count = 0 if r_live else _right_hold_count + 1

    if left_fit is None or right_fit is None:
        _prev_left_fit, _prev_right_fit = left_fit, right_fit
        return np.zeros_like(frame), _blank

    # ── Step 8: Anti-crossing guard ───────────────────────────────────────────
    if _lanes_crossing(left_fit, right_fit, h, w, search_horizon_y):
        if _prev_left_fit is not None and _prev_right_fit is not None:
            left_fit  = _prev_left_fit
            right_fit = _prev_right_fit

    _prev_left_fit  = left_fit
    _prev_right_fit = right_fit

    # ── Step 9: INDEPENDENT draw-distance shortening ──────────────────────────
    left_draw_top  = _draw_top_from_confidence(h, pixel_top_y, l_last_band, win_h)
    right_draw_top = _draw_top_from_confidence(h, pixel_top_y, r_last_band, win_h)

    # ── Step 10: Deviation (computed BEFORE drawing so the fill colour can
    #             react to it — NEW: red fill replaces blue when off-centre) ──
    lx_bot  = float(_eval(left_fit,  h))
    rx_bot  = float(_eval(right_fit, h))
    dev_pct = ((lx_bot + rx_bot) / 2.0 - cx) / cx * 100.0
    is_warn = abs(dev_pct) > WARNING_THRESH

    if DEBUG_PRINT:
        global _frame_counter
        _frame_counter += 1
        print(
            f"[F{_frame_counter:05d}] "
            f"dev={dev_pct:+7.3f}%  "
            f"lx_bot={lx_bot:7.1f}  rx_bot={rx_bot:7.1f}  cx={cx}  "
            f"left_fit=[{left_fit[0]:+.6f}, {left_fit[1]:+.4f}, {left_fit[2]:+.2f}]  "
            f"right_fit=[{right_fit[0]:+.6f}, {right_fit[1]:+.4f}, {right_fit[2]:+.2f}]  "
            f"l_live={l_live} r_live={r_live}  "
            f"l_pix={len(l_xs)} r_pix={len(r_xs)}  "
            f"l_hold={_left_hold_count} r_hold={_right_hold_count}"
        )

    # ── Step 11: Render ───────────────────────────────────────────────────────
    left_pts  = _make_pts(left_fit,  h, left_draw_top,  w)
    right_pts = _make_pts(right_fit, h, right_draw_top, w)
    left_flat  = left_pts.reshape(-1, 2)
    right_flat = right_pts.reshape(-1, 2)

    overlay = np.zeros_like(frame)

    # ── DEBUG: draw every raw pixel the sliding window collected ─────────────
    # Yellow dots = left side's raw detections, magenta = right side's.
    # If these dots sit ON the real lane markings but the drawn curve below
    # doesn't, the problem is in the polynomial fit/smoothing. If the dots
    # THEMSELVES are offset from the markings, the problem is upstream in
    # the mask/window search. Toggle DEBUG_MASK=True at the top of this
    # file to enable.
    if DEBUG_MASK:
        for px, py in zip(l_xs.astype(int), l_ys.astype(int)):
            cv2.circle(overlay, (px, py), 2, (0, 255, 255), -1)    # yellow = left
        for px, py in zip(r_xs.astype(int), r_ys.astype(int)):
            cv2.circle(overlay, (px, py), 2, (255, 0, 255), -1)    # magenta = right

    # Carpet fill — RED when the vehicle has drifted past WARNING_THRESH,
    # BLUE otherwise.  Uses the SHORTER of the two draw heights so the fill
    # polygon never extends past where either border line actually stops.
    fill_top_y  = max(left_draw_top, right_draw_top)
    fill_left   = _make_pts(left_fit,  h, fill_top_y, w).reshape(-1, 2)
    fill_right  = _make_pts(right_fit, h, fill_top_y, w).reshape(-1, 2)
    fill_color  = RED_FILL if is_warn else BLUE_FILL
    cv2.fillPoly(overlay, [np.vstack((fill_left, fill_right[::-1]))], fill_color)

    # Two INDEPENDENT thick border lines — each at its own natural length.
    # Borders turn red too during a warning, matching the fill, so the
    # whole lane visualization reads as a single alert state at a glance.
    border_color = RED_BORDER if is_warn else GREEN
    cv2.polylines(overlay, [left_pts],  isClosed=False, color=border_color, thickness=LINE_THICKNESS)
    cv2.polylines(overlay, [right_pts], isClosed=False, color=border_color, thickness=LINE_THICKNESS)

    # Upcoming-road curvature arrow (NAVIGATION signal — "what the road
    # is about to do", independent of where the car sits in the lane).
    direction, pct = _classify_curvature(left_fit, right_fit)
    _draw_curve_arrow(overlay, direction, pct, w, h)
    road_text = {
        'straight': "Stay Straight",
        'left':     "Left Curve Ahead",
        'right':    "Right Curve Ahead",
    }[direction]

    # Deviation-correction arrow (SAFETY signal — "which way to steer to
    # get back to centre").  Only drawn once there's a meaningful drift,
    # so it doesn't visually compete with the curvature arrow when the
    # car is already centred.  Positive dev_pct = lane centre is RIGHT of
    # car centre = car has drifted LEFT = correct by steering RIGHT.
    correct_dir = None
    if abs(dev_pct) > CORRECTION_DEAD_ZONE:
        correct_dir = 'right' if dev_pct > 0 else 'left'
        _draw_correction_arrow(overlay, correct_dir, abs(dev_pct), w, h)

    # ── Step 12: HUD metrics ──────────────────────────────────────────────────
    mode = "SW Poly: Live" if (l_live and r_live) else "SW Poly: Hold"

    return overlay, dict(
        deviation     = dev_pct,
        is_warning    = is_warn,
        road_status   = mode,
        brightness    = bright,
        upcoming_road = road_text,
    )