"""Quick test: run v8 ONNX height model on video, collect predictions."""
import sys, os, numpy as np, cv2, json, time
sys.path.insert(0, os.path.dirname(__file__))
from service.height_estimator import HeightEstimator

VIDEO = os.path.join(os.path.dirname(__file__), "..", "rgb_cam_20260415_021613_970_first5min.mp4")
MODELS = os.path.join(os.path.dirname(__file__), "models", "height")

he = HeightEstimator(MODELS)
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"FAIL: cannot open {VIDEO}")
    sys.exit(1)

driver_heights, passenger_heights = [], []
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {total} frames @ {fps:.1f} fps")

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1
    if frame_idx % 30 != 0:
        continue

    try:
        result = he.predict(frame)
        d = result.get("driver", {}) or {}
        p = result.get("passenger", {}) or {}
        if d.get("ema"):
            driver_heights.append(d["ema"])
        if p.get("ema"):
            passenger_heights.append(p["ema"])
    except Exception:
        pass

    if frame_idx % 300 == 0:
        print(f"  frame {frame_idx}/{total} | driver samples={len(driver_heights)} passenger={len(passenger_heights)}")

cap.release()

def summarize(name, vals):
    if not vals:
        print(f"\n  {name}: no predictions")
        return
    a = np.array(vals)
    print(f"\n  {name}:")
    print(f"    samples: {len(a)}")
    print(f"    mean±std: {a.mean():.1f}±{a.std():.1f} cm")
    print(f"    median:   {np.median(a):.1f} cm")
    print(f"    range:    {a.min():.1f} – {a.max():.1f} cm")
    print(f"    last 10:  {[round(v,1) for v in a[-10:]]}")

print("\n=== Results ===")
summarize("Driver", driver_heights)
summarize("Passenger", passenger_heights)
