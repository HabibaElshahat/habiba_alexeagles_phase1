import cv2
import numpy as np
import argparse
import os
from math import atan2, pi


def imread_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

def binarize_gear(img):
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.sum(th == 255) > np.sum(th == 0):
        th = cv2.bitwise_not(th)

    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8), iterations=1)
    return th

def largest_contour(bin_img):
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)

def inner_hole_contour(bin_img, outer_cnt):
    if outer_cnt is None:
        return None
    mask = np.zeros_like(bin_img)
    cv2.drawContours(mask, [outer_cnt], -1, 255, -1)
    inner = cv2.bitwise_and(mask, cv2.bitwise_not(bin_img))
    cnts, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)

def contour_center(cnt, fallback_shape=None):
    if cnt is None:
        if fallback_shape is not None:
            h, w = fallback_shape[:2]
            return (w/2.0, h/2.0)
        return (0.0, 0.0)
    M = cv2.moments(cnt)
    if M["m00"] > 1e-9:
        return (M["m10"]/M["m00"], M["m01"]/M["m00"])
    (x, y), _ = cv2.minEnclosingCircle(cnt)
    return (x, y)

def robust_radius(cnt, center, q=0.95):
    if cnt is None:
        return 0.0
    cx, cy = center
    pts = cnt.reshape(-1, 2).astype(np.float32)
    r = np.sqrt((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2)
    q = np.clip(q, 0.5, 0.99)
    return float(np.quantile(r, q))

def robust_inner_radius(cnt, center, q=0.50):
    if cnt is None:
        return 0.0
    cx, cy = center
    pts = cnt.reshape(-1, 2).astype(np.float32)
    r = np.sqrt((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2)
    q = np.clip(q, 0.3, 0.9)
    return float(np.quantile(r, q))

def measure_diameters(bin_img, diameter_quantile=0.95):
    outer = largest_contour(bin_img)
    if outer is None:
        return 0.0, 0.0, (bin_img.shape[1]/2.0, bin_img.shape[0]/2.0), None, None
    ctr = contour_center(outer, fallback_shape=bin_img.shape)
    r_out = robust_radius(outer, ctr, q=diameter_quantile)
    hole = inner_hole_contour(bin_img, outer)
    r_in = robust_inner_radius(hole, ctr, q=0.5) if hole is not None else 0.0
    return 2.0*r_out, 2.0*r_in, ctr, outer, hole

def fit_angle(cnt):
    if cnt is None or len(cnt) < 5:
        return 0.0
    try:
        _, _, angle = cv2.fitEllipse(cnt)
        return float(angle)
    except:
        return 0.0

def align_sample_to_ideal(sample_bin, s_ctr, s_outer_r, ideal_shape, i_ctr, i_angle=0.0, s_angle=0.0, auto_rotate=False):
    H, W = ideal_shape[:2]

    def warp_scale(img, s, center):
        M1 = np.array([[1,0,-center[0]],[0,1,-center[1]]], dtype=np.float32)
        M2 = np.array([[s,0,0],[0,s,0]], dtype=np.float32)
        M3 = np.array([[1,0, center[0]],[0,1, center[1]]], dtype=np.float32)
        M = M3 @ M2 @ M1
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)

    def warp_rotate(img, angle_deg, center):
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)

    sb = sample_bin.copy()
    scaled = sb

    rotated = scaled
    if auto_rotate:
        dtheta = (i_angle - s_angle)
        rotated = warp_rotate(scaled, dtheta, s_ctr)

    dx = i_ctr[0] - s_ctr[0]
    dy = i_ctr[1] - s_ctr[1]
    M = np.float32([[1,0,dx],[0,1,dy]])
    aligned = cv2.warpAffine(rotated, M, (sample_bin.shape[1], sample_bin.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)

    return aligned

def rescale_to_match(sample_bin, s_outer_r, i_outer_r):
    if s_outer_r <= 1 or i_outer_r <= 1:
        return sample_bin
    scale = i_outer_r / s_outer_r
    H, W = sample_bin.shape[:2]
    newW = max(1, int(round(W * scale)))
    newH = max(1, int(round(H * scale)))
    return cv2.resize(sample_bin, (newW, newH), interpolation=cv2.INTER_NEAREST), scale

def paste_centered(img, target_shape, center_from, center_to):
    Ht, Wt = target_shape[:2]
    canvas = np.zeros(target_shape, dtype=img.dtype)
    dx = int(round(center_to[0] - center_from[0]))
    dy = int(round(center_to[1] - center_from[1]))
    # place with translation
    M = np.float32([[1,0,dx],[0,1,dy]])
    aligned = cv2.warpAffine(img, M, (Wt, Ht), flags=cv2.INTER_NEAREST, borderValue=0)
    return aligned


def outer_ring_mask(bin_ideal, outer_r, ring_frac=0.06):
    ring_w = max(2, int(round(ring_frac * outer_r)))
    r = ring_w
    k = 2*r + 1
    Y, X = np.ogrid[-r:r+1, -r:r+1]
    ker = np.zeros((k, k), np.uint8)
    ker[(X*X + Y*Y) <= r*r] = 1
    eroded = cv2.erode(bin_ideal, ker, iterations=1)
    ring = cv2.bitwise_and(bin_ideal, cv2.bitwise_not(eroded))
    return ring, ring_w

def angle_of_point(cx, cy, px, py):
    a = atan2(py - cy, px - cx)
    if a < 0:
        a += 2*pi
    return a

def detect_teeth_damage(bin_ideal, bin_sample, center, outer_r,
                        teeth=30, ring_frac=0.06, missing_factor=0.80, worn_factor=0.10,
                        min_component=300, debug_info=None):

    missing = cv2.bitwise_and(bin_ideal, cv2.bitwise_not(bin_sample))

    ring_mask, ring_w = outer_ring_mask(bin_ideal, outer_r, ring_frac=ring_frac)
    missing_zone = cv2.bitwise_and(missing, ring_mask)

    kernel = np.ones((3, 3), np.uint8)
    missing_zone = cv2.morphologyEx(missing_zone, cv2.MORPH_OPEN, kernel, iterations=2)
    missing_zone = cv2.morphologyEx(missing_zone, cv2.MORPH_CLOSE, kernel, iterations=1)

    sector_angle = 2 * pi / max(1, teeth)
    sector_len = (2 * pi * outer_r) / max(1, teeth)
    sector_area = sector_len * max(1, ring_w)  # approx pixels per sector

    MISSING_THRESH = missing_factor * sector_area
    WORN_THRESH = worn_factor * sector_area

    if debug_info:
        total_missing_pixels = int(np.count_nonzero(missing_zone))
        print(f"  Sector area (approx): {sector_area:.1f} px")
        print(f"  Missing threshold: {MISSING_THRESH:.1f} px")
        print(f"  Worn threshold: {WORN_THRESH:.1f} px")
        print(f"  Missing pixels total: {total_missing_pixels}")

    if np.count_nonzero(missing_zone) == 0:
        debug_enhanced = cv2.cvtColor(bin_ideal, cv2.COLOR_GRAY2BGR)
        for i in range(teeth):
            angle = i * sector_angle
            x = int(center[0] + outer_r * 0.7 * np.cos(angle))
            y = int(center[1] + outer_r * 0.7 * np.sin(angle))
            cv2.putText(debug_enhanced, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 255), 1)
        return [], [], 0, 0, debug_enhanced

    ys, xs = np.nonzero(missing_zone)
    if len(xs) == 0:
        debug_enhanced = cv2.cvtColor(bin_ideal, cv2.COLOR_GRAY2BGR)
        return [], [], 0, 0, debug_enhanced

    angs = np.arctan2(ys - center[1], xs - center[0])
    angs[angs < 0] += 2 * pi
    idxs = np.floor(angs / sector_angle).astype(int) % teeth

    per_tooth_area = np.bincount(idxs, minlength=teeth).astype(np.float32)  # pixel counts per tooth

    valid_mask = per_tooth_area >= max(1, min_component)

    missing_idx = []
    worn_idx = []

    for i in range(teeth):
        a = per_tooth_area[i]
        if not valid_mask[i]:
            # too small - ignore as noise
            if debug_info and a > 0:
                print(f"    Tooth {i+1}: Ignored (small) area={a:.1f}")
            continue

        if a >= max(MISSING_THRESH, 0.20 * sector_area):
            missing_idx.append(i)
            if debug_info:
                print(f"    Tooth {i+1}: BROKEN - {a:.1f} px")
        elif a >= max(WORN_THRESH, 0.10 * sector_area):
            worn_idx.append(i)
            if debug_info:
                print(f"    Tooth {i+1}: WORN - {a:.1f} px")
        elif debug_info and a > 0:
            print(f"    Tooth {i+1}: Minor damage - {a:.1f} px (ignored)")

    debug_enhanced = cv2.cvtColor(bin_ideal, cv2.COLOR_GRAY2BGR)

    H, W = bin_ideal.shape[:2]
    Y, X = np.indices((H, W))
    ang_map = np.arctan2(Y - center[1], X - center[0])
    ang_map[ang_map < 0] += 2 * pi

    for i in range(teeth):
        start_ang = i * sector_angle
        end_ang = (i + 1) * sector_angle
        sector_mask = (ang_map >= start_ang) & (ang_map < end_ang) & (ring_mask > 0)
        damage_mask = sector_mask & (missing_zone > 0)

        if i in missing_idx:
            debug_enhanced[damage_mask] = (0, 0, 255)
        elif i in worn_idx:
            debug_enhanced[damage_mask] = (0, 165, 255)
        elif np.any(damage_mask):
            debug_enhanced[damage_mask] = (0, 255, 0)

        angle = i * sector_angle
        x = int(center[0] + outer_r * 0.7 * np.cos(angle))
        y = int(center[1] + outer_r * 0.7 * np.sin(angle))
        cv2.putText(debug_enhanced, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 255), 1)

    return missing_idx, worn_idx, len(missing_idx), len(worn_idx), debug_enhanced

def classify_inner_opening(i_hole_r, s_hole_r, inner_large_factor=1.12, inner_missing_factor=0.60):
    if i_hole_r <= 1:
        return None
    if s_hole_r <= 1:
        return "Missing inner opening"
    if s_hole_r > i_hole_r * inner_large_factor:
        return "Large inner opening"
    if s_hole_r < i_hole_r * inner_missing_factor:
        return "Missing inner opening"
    return None


def inspect(ideal_path, sample_paths, teeth=30,
            ring_frac=0.06, missing_factor=0.80, worn_factor=0.10,
            diameter_quantile=0.95, diameter_tol_px=6, diameter_tol_frac=0.02,
            inner_large_factor=1.12, inner_missing_factor=0.60,
            min_component=300, one_based=False, save_debug=False, auto_rotate=False):

    ideal_gray = imread_gray(ideal_path)
    ideal_bin  = binarize_gear(ideal_gray)
    i_outer_d, i_inner_d, i_ctr, i_outer_cnt, i_hole_cnt = measure_diameters(ideal_bin, diameter_quantile)
    i_outer_r = i_outer_d/2.0
    i_hole_r  = i_inner_d/2.0

    print(f"ideal: OuterD={i_outer_d:.1f}px  InnerD={i_inner_d:.1f}px")

    if save_debug:
        cv2.imwrite("ideal_binary.png", ideal_bin)

    for sp in sample_paths:
        print(f"\n--- Inspecting: {sp} ---")
        try:
            s_gray = imread_gray(sp)
        except FileNotFoundError:
            print("Error: file not found.")
            continue
        s_bin = binarize_gear(s_gray)
        s_outer_d, s_inner_d, s_ctr, s_outer_cnt, s_hole_cnt = measure_diameters(s_bin, diameter_quantile)
        s_outer_r = s_outer_d/2.0
        s_hole_r  = s_inner_d/2.0

        px_diff = abs(s_outer_d - i_outer_d)
        frac_diff = px_diff / max(i_outer_d, 1.0)
        diam_flag = (px_diff > diameter_tol_px) and (frac_diff > diameter_tol_frac)

        sample_scaled, scale = rescale_to_match(s_bin, s_outer_r if s_outer_r>0 else 1.0, i_outer_r if i_outer_r>0 else 1.0)

        s_outer_cnt_scaled = largest_contour(sample_scaled)
        s_ctr_scaled = contour_center(s_outer_cnt_scaled, fallback_shape=sample_scaled.shape)

        sample_aligned = paste_centered(sample_scaled, ideal_bin.shape, s_ctr_scaled, i_ctr)

        if auto_rotate:
            i_angle = fit_angle(i_outer_cnt)
            s_angle = fit_angle(s_outer_cnt_scaled)
            Mrot = cv2.getRotationMatrix2D(i_ctr, i_angle - s_angle, 1.0)
            sample_aligned = cv2.warpAffine(sample_aligned, Mrot, (ideal_bin.shape[1], ideal_bin.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)

        miss_idx, worn_idx, miss_cnt, worn_cnt, miss_zone = detect_teeth_damage(
            ideal_bin, sample_aligned, i_ctr, i_outer_r,
            teeth=teeth, ring_frac=ring_frac, missing_factor=missing_factor, worn_factor=worn_factor,
            min_component=min_component
        )

        inner_issue = classify_inner_opening(i_hole_r, s_hole_r, inner_large_factor, inner_missing_factor)

        if one_based:
            miss_idx_out = [i+1 for i in miss_idx]
            worn_idx_out = [i+1 for i in worn_idx]
        else:
            miss_idx_out = miss_idx
            worn_idx_out = worn_idx


        print(f"Inner Diameter: {s_inner_d:.1f}px (Ideal {i_inner_d:.1f}px)")
        if inner_issue:
            print(f"Inner Opening: {inner_issue}")

        if miss_cnt > 0 and worn_cnt > 0:
            print(f"Teeth Status: {miss_cnt} broken and {worn_cnt} worn teeth detected")
        elif miss_cnt > 0:
            print(f"Teeth Status: {miss_cnt} broken teeth detected")
        elif worn_cnt > 0:
            print(f"Teeth Status: {worn_cnt} worn teeth detected")
        else:
            print("Teeth Status: No visible damage")

        if save_debug:
            base = os.path.splitext(os.path.basename(sp))[0]
            cv2.imwrite(f"{base}_bin.png", s_bin)
            cv2.imwrite(f"{base}_aligned.png", sample_aligned)
            cv2.imwrite(f"{base}_missing_zone.png", miss_zone)


def main():
    p = argparse.ArgumentParser(description="Gear inspection: diameter + broken/worn teeth via aligned comparison.")
    p.add_argument("--ideal", required=True, help="Path to ideal/reference gear image")
    p.add_argument("--samples", required=True, nargs="+", help="Paths to sample gear images")
    p.add_argument("--teeth", type=int, default=30, help="Number of teeth on the gear")
    p.add_argument("--ring-frac", type=float, default=0.06, help="Teeth ring thickness as fraction of outer radius")
    p.add_argument("--missing-factor", type=float, default=0.60, help="Area factor for broken tooth threshold")
    p.add_argument("--worn-factor", type=float, default=0.15, help="Area factor for worn tooth threshold")
    p.add_argument("--min-component", type=int, default=120, help="Min diff area (px) to consider a component")
    p.add_argument("--diameter-quantile", type=float, default=0.95, help="Quantile for robust outer diameter")
    p.add_argument("--diameter-tol-px", type=float, default=6, help="Absolute px tolerance for diameter diff flag")
    p.add_argument("--diameter-tol-frac", type=float, default=0.02, help="Relative tolerance (fraction of ideal)")
    p.add_argument("--inner-large-factor", type=float, default=1.12, help="Factor to call inner opening large")
    p.add_argument("--inner-missing-factor", type=float, default=0.60, help="Factor to call inner opening missing")
    p.add_argument("--one-based", action="store_true", help="Report tooth indices as 1-based")
    p.add_argument("--save-debug", action="store_true", help="Save debug images")
    p.add_argument("--auto-rotate", action="store_true", help="Try to align sample rotation to ideal")
    args = p.parse_args()

    inspect(args.ideal, args.samples, teeth=args.teeth,
            ring_frac=args.ring_frac, missing_factor=args.missing_factor, worn_factor=args.worn_factor,
            diameter_quantile=args.diameter_quantile, diameter_tol_px=args.diameter_tol_px, diameter_tol_frac=args.diameter_tol_frac,
            inner_large_factor=args.inner_large_factor, inner_missing_factor=args.inner_missing_factor,
            min_component=args.min_component, one_based=args.one_based, save_debug=args.save_debug, auto_rotate=args.auto_rotate)

if __name__ == "__main__":
    main()
