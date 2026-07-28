# ─────────────────────────────────────────────────────────────────────────────
#  lane_detection.py  |  Modular Hybrid ADAS — Lane Perception Module  (v6.10)
# ─────────────────────────────────────────────────────────────────────────────

import warnings
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

try:
    from numpy.exceptions import RankWarning as _RankWarning
except ImportError:
    _RankWarning = np.RankWarning
warnings.filterwarnings('ignore', category=_RankWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing YOLOP AI on hardware: {device.type.upper()}...")

yolop_model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
yolop_model = yolop_model.to(device)
yolop_model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

SEARCH_HORIZON_RATIO = 0.57
SMOOTH_ALPHA         = 0.75
N_CURVE_PTS          = 60
LINE_THICKNESS       = 10
DILATE_KERN          = 9
WARNING_THRESH       = 6.0

N_WINDOWS            = 15
WINDOW_MARGIN        = 90
MIN_PIX_RECENTER     = 8

MIN_PIXELS           = 25
MIN_Y_SPAN           = 60
MIN_LANE_RATIO       = 0.12

INNER_LANE_RATIO     = 0.38
HIST_SIGMA_RATIO     = 0.08

MIN_CONFIDENT_BANDS  = 4
DRAW_MARGIN_BANDS    = 1

MAX_HOLD_FRAMES      = 8
WINDOW_WIDEN_STEP    = 25
WINDOW_WIDEN_CAP     = 200

YELLOW       = (0, 255, 255)
GREEN        = (60, 220, 60)
BLUE_FILL    = (235, 140, 60)
RED_FILL     = (50, 50, 230)
RED_BORDER   = (60, 60, 230)
WHITE        = (255, 255, 255)

CORRECTION_DEAD_ZONE = 3.0

DEBUG_MASK = False
DEBUG_PRINT = True
_frame_counter = 0

# ── PERSISTENT STATES ──
_prev_left_fit    = None         
_prev_right_fit   = None
_left_hold_count  = 0           
_right_hold_count = 0
_last_warn_state  = False           
_smoothed_bulge   = 0.0      # NEW: Temporal memory for the curve


def _infer_mask(frame, h, w, vehicle_boxes=None):
    img    = cv2.resize(frame, (640, 640))
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, seg = yolop_model(tensor)
    raw  = torch.argmax(seg, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    mask = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    
    if vehicle_boxes:
        for (x1, y1, x2, y2) in vehicle_boxes:
            bx1, by1 = max(0, int(x1) - 20), max(0, int(y1) - 20)
            bx2, by2 = min(w, int(x2) + 20), min(h, int(y2) + 20)
            cv2.rectangle(mask, (bx1, by1), (bx2, by2), 0, -1)
            
    return cv2.dilate(mask, np.ones((DILATE_KERN, DILATE_KERN), np.uint8))


def _cold_start_bases(mask, h, w, cx):
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
    return fit[0] * y**2 + fit[1] * y + fit[2]


def _convergence_y(left_fit, right_fit, h, w, search_horizon_y):
    if left_fit is None or right_fit is None:
        return search_horizon_y

    min_gap = w * MIN_LANE_RATIO
    for cy in range(search_horizon_y, h, 5):
        if float(_eval(right_fit, cy)) - float(_eval(left_fit, cy)) >= min_gap:
            return cy
    return int(h * 0.80)


def _sliding_window_guided(mask, h, pixel_top_y, base_x, w, prev_fit=None, extra_margin=0):
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
            last_confident_band = win        

    return (np.array(all_ys, dtype=np.float32),
            np.array(all_xs, dtype=np.float32),
            last_confident_band,
            win_h)


def _fit_poly(pixel_ys, pixel_xs):
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

        y_min, y_max = pixel_ys.min(), pixel_ys.max()
        span = max(y_max - y_min, 1.0)
        weights = (pixel_ys - y_min) / span        
        weights = weights + 0.15                    

        return np.polyfit(pixel_ys, pixel_xs, 2, w=weights)
    except Exception:
        return None


def _is_sane(fit, h, w, side, top_y):
    bx    = float(_eval(fit, h))
    tx    = float(_eval(fit, top_y))
    half  = w / 2.0
    slack = w * 0.22
    if side == 'left'  and not (0             < bx < half + slack): return False
    if side == 'right' and not (half - slack  < bx < w           ): return False

    vertical_span = max(h - top_y, 1)
    horizontal_span = abs(bx - tx)
    min_expected_span = vertical_span * 0.08   
    if horizontal_span < min_expected_span:
        return False

    return True


def _blend(prev, curr):
    if prev is None:
        return curr
    return SMOOTH_ALPHA * np.asarray(prev) + (1.0 - SMOOTH_ALPHA) * np.asarray(curr)


def _draw_top_from_confidence(h, pixel_top_y, last_confident_band, win_h):
    if last_confident_band < MIN_CONFIDENT_BANDS - 1:
        return int(h - (h - pixel_top_y) * 0.55)

    stop_band = max(0, last_confident_band - DRAW_MARGIN_BANDS)
    return int(h - (stop_band + 1) * win_h)


def _make_pts(fit, h, draw_top_y, w):
    draw_top_y = max(draw_top_y, 0)
    if draw_top_y >= h - 1:
        draw_top_y = h - 2
    ys = np.linspace(h - 1, draw_top_y, N_CURVE_PTS)
    xs = np.clip(_eval(fit, ys), 0, w - 1)
    return np.column_stack((xs, ys)).reshape(-1, 1, 2).astype(np.int32)


def _lanes_crossing(left_fit, right_fit, h, w, horizon_y):
    ys  = np.linspace(horizon_y, h - 1, 10)
    lxs = _eval(left_fit,  ys)
    rxs = _eval(right_fit, ys)

    proximity = (ys - horizon_y) / max(h - 1 - horizon_y, 1)
    min_gap_floor = w * 0.01
    min_gap_full  = w * MIN_LANE_RATIO
    min_gap       = min_gap_floor + proximity * (min_gap_full - min_gap_floor)

    actual_crossing = np.any(lxs >= rxs)          
    too_narrow      = np.any((rxs - lxs) < min_gap)
    return bool(actual_crossing or too_narrow)


# ── FINAL POLISH: Temporal Smoothing & Dynamic Sensitivity ──
def _classify_curvature(left_fit, right_fit, h, top_y):
    global _smoothed_bulge
    
    y_bot = float(h - 1)
    y_top = float(max(top_y, 0))
    carpet_height = y_bot - y_top
    
    # 1. HARD CUTOFF: If view is heavily blocked, smoothly decay memory to 0 (Straight).
    if carpet_height < (h * 0.25):
        _smoothed_bulge = _smoothed_bulge * 0.5
        return 'straight', 0
        
    y_mid = (y_bot + y_top) / 2.0
    
    cx_bot = (_eval(left_fit, y_bot) + _eval(right_fit, y_bot)) / 2.0
    cx_top = (_eval(left_fit, y_top) + _eval(right_fit, y_top)) / 2.0
    cx_mid = (_eval(left_fit, y_mid) + _eval(right_fit, y_mid)) / 2.0
    
    straight_mid_x = (cx_bot + cx_top) / 2.0
    raw_bulge = cx_mid - straight_mid_x
    
    # 2. TEMPORAL SMOOTHING: Absorbs frame-by-frame noise completely.
    _smoothed_bulge = (_smoothed_bulge * 0.85) + (raw_bulge * 0.15)
    
    # 3. VISIBILITY SCALING
    visibility_ratio = carpet_height / h 
    
    # Dead zone shrinks to just 2.0 pixels for perfect clear views (catches slight curves)
    dynamic_dead_zone = 2.0 + (1.0 - visibility_ratio) * 6.0 
    
    if abs(_smoothed_bulge) < dynamic_dead_zone:
        # If road is straight, gently pull memory back to 0 to prevent it floating near the edge
        _smoothed_bulge = _smoothed_bulge * 0.8
        return 'straight', 0

    # 4. SENSITIVITY CALIBRATION: The "Floor" prevents 100% jumps on short lines.
    expected_max_bulge = max(20.0, 55.0 * visibility_ratio) 
    
    pct = int(round(np.clip((abs(_smoothed_bulge) / expected_max_bulge) * 100.0, 0, 100)))
        
    return ('left', pct) if _smoothed_bulge > 0 else ('right', pct)


def _arrow_polygon(cx, cy, direction, size):
    if direction == 'straight':
        return np.array([
            [cx, cy - size], [cx - 22, cy - size + 30], [cx - 10, cy - size + 30],
            [cx - 10, cy + size], [cx + 10, cy + size], [cx + 10, cy - size + 30],
            [cx + 22, cy - size + 30],
        ], dtype=np.int32)
    elif direction == 'right':
        return np.array([
            [cx + size, cy], [cx + size - 30, cy - 22], [cx + size - 30, cy - 10],
            [cx - size, cy - 10], [cx - size, cy + 10], [cx + size - 30, cy + 10],
            [cx + size - 30, cy + 22],
        ], dtype=np.int32)
    else:  
        return np.array([
            [cx - size, cy], [cx - size + 30, cy - 22], [cx - size + 30, cy - 10],
            [cx + size, cy - 10], [cx + size, cy + 10], [cx - size + 30, cy + 10],
            [cx - size + 30, cy + 22],
        ], dtype=np.int32)


def _draw_curve_arrow(overlay, direction, pct, w, h):
    cx, cy = w // 2, int(h * 0.90)
    pts = _arrow_polygon(cx, cy, direction, size=45)
    cv2.polylines(overlay, [pts], isClosed=True, color=GREEN, thickness=3)
    if pct > 0:
        cv2.putText(overlay, f"{pct}%", (cx + 45 + 15, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)


def _draw_correction_arrow(overlay, direction, dev_pct, w, h):
    offset_x = int(w * 0.22)
    cx = (w // 2) - offset_x if direction == 'left' else (w // 2) + offset_x
    cy = int(h * 0.82)
    
    pts = _arrow_polygon(cx, cy, direction, size=30)
    color = (60, 60, 230)
    cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=3)
    
    text = f"{dev_pct:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.7, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = cx - tw // 2
    ty = cy - 40 
    
    cv2.putText(overlay, text, (tx, ty), font, scale, color, thick)


def process_lanes(frame, vehicle_boxes=None):
    global _prev_left_fit, _prev_right_fit, _left_hold_count, _right_hold_count, _frame_counter, _last_warn_state

    h, w             = frame.shape[:2]
    search_horizon_y = int(h * SEARCH_HORIZON_RATIO)
    cx               = w // 2
    bright           = int(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))

    _blank = dict(deviation=0.0, is_warning=False, road_status="Searching...",
                  brightness=bright, upcoming_road="—")

    mask = _infer_mask(frame, h, w, vehicle_boxes)

    force_left_reset  = _left_hold_count  >= MAX_HOLD_FRAMES
    force_right_reset = _right_hold_count >= MAX_HOLD_FRAMES

    left_seed_fit  = None if force_left_reset  else _prev_left_fit
    right_seed_fit = None if force_right_reset else _prev_right_fit

    if left_seed_fit is not None:
        left_base  = int(np.clip(float(_eval(left_seed_fit,  h)), 10,      cx - 10))
    else:
        left_base, _ = _cold_start_bases(mask, h, w, cx)

    if right_seed_fit is not None:
        right_base = int(np.clip(float(_eval(right_seed_fit, h)), cx + 10, w  - 10))
    else:
        _, right_base = _cold_start_bases(mask, h, w, cx)

    pixel_top_y = _convergence_y(left_seed_fit, right_seed_fit, h, w, search_horizon_y)

    left_extra_margin  = min(_left_hold_count  * WINDOW_WIDEN_STEP, WINDOW_WIDEN_CAP)
    right_extra_margin = min(_right_hold_count * WINDOW_WIDEN_STEP, WINDOW_WIDEN_CAP)

    l_ys, l_xs, l_last_band, win_h = _sliding_window_guided(
        mask, h, pixel_top_y, left_base,  w, left_seed_fit,  left_extra_margin
    )
    r_ys, r_xs, r_last_band, _ = _sliding_window_guided(
        mask, h, pixel_top_y, right_base, w, right_seed_fit, right_extra_margin
    )

    def _update(prev_fit, ys, xs, side):
        curr = _fit_poly(ys, xs)
        if curr is not None and _is_sane(curr, h, w, side, search_horizon_y):
            return _blend(prev_fit, curr), True
        if prev_fit is not None:
            return prev_fit, False
        return None, False

    left_fit,  l_live = _update(_prev_left_fit,  l_ys, l_xs, 'left')
    right_fit, r_live = _update(_prev_right_fit, r_ys, r_xs, 'right')

    _left_hold_count  = 0 if l_live else _left_hold_count  + 1
    _right_hold_count = 0 if r_live else _right_hold_count + 1

    if left_fit is None or right_fit is None:
        _prev_left_fit, _prev_right_fit = left_fit, right_fit
        return np.zeros_like(frame), _blank

    if _lanes_crossing(left_fit, right_fit, h, w, search_horizon_y):
        if _prev_left_fit is not None and _prev_right_fit is not None:
            left_fit  = _prev_left_fit
            right_fit = _prev_right_fit

    _prev_left_fit  = left_fit
    _prev_right_fit = right_fit

    left_draw_top  = _draw_top_from_confidence(h, pixel_top_y, l_last_band, win_h)
    right_draw_top = _draw_top_from_confidence(h, pixel_top_y, r_last_band, win_h)

    lx_bot  = float(_eval(left_fit,  h))
    rx_bot  = float(_eval(right_fit, h))
    dev_pct = ((lx_bot + rx_bot) / 2.0 - cx) / cx * 100.0
    is_warn = abs(dev_pct) > WARNING_THRESH

    if DEBUG_PRINT:
        _frame_counter += 1
        anomalies = []
        if force_left_reset: anomalies.append("L_FORCE_RESET")
        if force_right_reset: anomalies.append("R_FORCE_RESET")
        if not l_live: anomalies.append("L_HOLD")
        if not r_live: anomalies.append("R_HOLD")
        
        has_anomaly = len(anomalies) > 0
        warn_changed = (is_warn != _last_warn_state)

        if has_anomaly or warn_changed or _frame_counter % 30 == 0:
            dev_dir = "R" if dev_pct > 0 else "L"
            if is_warn: tag = "🚨 DEPARTURE"
            elif has_anomaly: tag = "⚠️ ANOMALY  "
            else: tag = "🟢 STABLE   "
            print(f"[{_frame_counter:05d}] {tag} | Dev: {abs(dev_pct):4.1f}% {dev_dir} | Sensors: [L:{'LIVE' if l_live else 'HOLD'} R:{'LIVE' if r_live else 'HOLD'}]")
            
        _last_warn_state = is_warn

    left_pts  = _make_pts(left_fit,  h, left_draw_top,  w)
    right_pts = _make_pts(right_fit, h, right_draw_top, w)
    
    overlay = np.zeros_like(frame)

    if DEBUG_MASK:
        for px, py in zip(l_xs.astype(int), l_ys.astype(int)):
            cv2.circle(overlay, (px, py), 2, (0, 255, 255), -1)    
        for px, py in zip(r_xs.astype(int), r_ys.astype(int)):
            cv2.circle(overlay, (px, py), 2, (255, 0, 255), -1)   

    fill_top_y  = max(left_draw_top, right_draw_top)
    fill_left   = _make_pts(left_fit,  h, fill_top_y, w).reshape(-1, 2)
    fill_right  = _make_pts(right_fit, h, fill_top_y, w).reshape(-1, 2)
    fill_color  = RED_FILL if is_warn else BLUE_FILL
    cv2.fillPoly(overlay, [np.vstack((fill_left, fill_right[::-1]))], fill_color)

    border_color = RED_BORDER if is_warn else GREEN
    cv2.polylines(overlay, [left_pts],  isClosed=False, color=border_color, thickness=LINE_THICKNESS)
    cv2.polylines(overlay, [right_pts], isClosed=False, color=border_color, thickness=LINE_THICKNESS)

    direction, pct = _classify_curvature(left_fit, right_fit, h, fill_top_y)
    _draw_curve_arrow(overlay, direction, pct, w, h)
    road_text = {
        'straight': "Stay Straight",
        'left':     "Left Curve Ahead",
        'right':    "Right Curve Ahead",
    }[direction]

    correct_dir = None
    if abs(dev_pct) > CORRECTION_DEAD_ZONE:
        correct_dir = 'right' if dev_pct > 0 else 'left'
        _draw_correction_arrow(overlay, correct_dir, abs(dev_pct), w, h)

    mode = "SW Poly: Live" if (l_live and r_live) else "SW Poly: Hold"

    return overlay, dict(
        deviation     = dev_pct,
        is_warning    = is_warn,
        road_status   = mode,
        brightness    = bright,
        upcoming_road = road_text,
    )