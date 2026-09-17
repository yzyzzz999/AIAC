import sys, os, gc, warnings, csv, time, datetime
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import cv2
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
warnings.filterwarnings("ignore")

SEP = "=" * 60
CLASS_NAMES = ["C0-偏瘦", "C1-正常", "C2-超重", "C3-肥胖"]

STABLE_CONSECUTIVE = 15
MIN_FRAMES = 20
MAX_FRAMES = 180
CONFIDENCE_THRESHOLD = 0.4
FRAME_SKIP = 1  # Process every frame for real-time

def init_face_detector():
    try:
        import insightface
        from insightface.app import FaceAnalysis
        model = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        model.prepare(ctx_id=0, det_size=(640, 640))
        return model
    except Exception as e:
        print(f"  InsightFace init failed: {e}")
        return None

def detect_faces(image, fd=None, margin=0.3):
    if fd is None: return []
    try:
        arr = np.array(image)
        faces = fd.get(arr)
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x1 = max(0, bbox[0] - int(w * margin))
            y1 = max(0, bbox[1] - int(h * margin))
            x2 = min(arr.shape[1], bbox[2] + int(w * margin))
            y2 = min(arr.shape[0], bbox[3] + int(h * margin))
            crop = Image.fromarray(arr[y1:y2, x1:x2]).convert("RGB")
            results.append((crop, cx, cy, bbox))
        return results
    except:
        return []

class OfficialYOLOv8Neck(nn.Module):
    """官方YOLOv8-pose Neck（冻结权重）"""

    def __init__(self):
        super().__init__()

        from ultralytics import YOLO
        official_model = YOLO(str(Path(__file__).resolve().parent.parent / "models" / "bmi" / "yolov8n-pose.pt"))

        self.neck = nn.Sequential()
        for i, m in enumerate(official_model.model.model[10:22]):
            self.neck.add_module(str(i), m)

        for p in self.neck.parameters():
            p.requires_grad = False

    def forward(self, features):
        p3, p4, p5 = features

        y = {4: p3, 6: p4, 9: p5}
        x = p5
        layers = list(self.neck)

        # Layer 10: Upsample
        x = layers[0](x);
        y[10] = x
        # Layer 11: Concat([-1, 6])
        x = layers[1]([x, y[6]]);
        y[11] = x
        # Layer 12: C2f
        x = layers[2](x);
        y[12] = x
        # Layer 13: Upsample
        x = layers[3](x);
        y[13] = x
        # Layer 14: Concat([-1, 4])
        x = layers[4]([x, y[4]]);
        y[14] = x
        # Layer 15: C2f → P3 (64ch)
        x = layers[5](x);
        p3_out = x
        # Layer 16: Conv (下采样)
        x = layers[6](x)
        # Layer 17: Concat([-1, 12])
        x = layers[7]([x, y[12]])
        # Layer 18: C2f → P4 (128ch)
        x = layers[8](x);
        p4_out = x
        # Layer 19: Conv (下采样)
        x = layers[9](x)
        # Layer 20: Concat([-1, 9])
        x = layers[10]([x, y[9]])
        # Layer 21: C2f → P5 (256ch)
        x = layers[11](x);
        p5_out = x

        return [p3_out, p4_out, p5_out]




class BMIDualHead(nn.Module):
    """双任务Head：共享特征提取，仅在最后层分支（分类/回归）"""

    def __init__(self):
        super().__init__()
        neck_channels = [64, 128, 256]

        self.per_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
            )
            for in_ch in neck_channels
        ])

        self.shared_fc = nn.Sequential(
            nn.Linear(64 * 3 + 512, 64),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(64, 4)
        self.regressor = nn.Linear(64, 1)

    def forward(self, features, flip_features):
        pooled_features = []
        for feat, conv in zip(features, self.per_scale_convs):
            out = conv(feat)
            pooled = F.adaptive_avg_pool2d(out, (1, 1)).squeeze(-1).squeeze(-1)
            pooled_features.append(pooled)

        concatenated = torch.cat(pooled_features + [flip_features], dim=1)
        shared = self.shared_fc(concatenated)

        class_logits = self.classifier(shared)
        bmi = torch.sigmoid(self.regressor(shared).squeeze(-1)) * 40 + 10

        return bmi, class_logits, shared




class CachedYOLOv8FLIPModel(nn.Module):
    """YOLOv8 Neck + FLIP特征 + 双任务Head"""

    def __init__(self):
        super().__init__()
        self.neck = OfficialYOLOv8Neck()
        self.head = BMIDualHead()

    def forward(self, yolo_p3, yolo_p4, yolo_p5, flip_features):
        neck_output = self.neck([yolo_p3, yolo_p4, yolo_p5])
        bmi_pred, class_logits, shared_features = self.head(neck_output, flip_features)
        return bmi_pred, class_logits, shared_features

    def get_trainable_parameters(self):
        params = []
        params.extend(self.neck.parameters())
        params.extend(self.head.parameters())
        return params

class RealtimeFeatureExtractor:
    def __init__(self, yolo_path, flip_path, device=None):
        self.device = device or torch.device("cpu")
        print("Loading models...")
        gc.collect(); self._load_yolo(yolo_path); gc.collect(); self._load_flip(flip_path); print("OK")
    def _load_yolo(self, path):
        self.yolo_net = YOLO(path, verbose=False).model; self.yolo_net.eval().to("cpu")
    def _load_flip(self, path):
        from transformers import CLIPProcessor, CLIPModel
        self.flip_model = CLIPModel.from_pretrained(path); self.flip_processor = CLIPProcessor.from_pretrained(path); self.flip_model.eval().to(self.device)
    @torch.no_grad()
    def extract_yolo(self, imgs):
        feats, y, x = [], [], imgs
        for i, m in enumerate(self.yolo_net.model):
            if hasattr(m, "f"):
                if m.f == -1: x_input = x
                elif isinstance(m.f, int): x_input = y[m.f]
                else: x_input = [x if j == -1 else y[j] for j in m.f]
            else: x_input = x
            x = m(x_input); y.append(x)
            if i in [4, 6, 9]: feats.append(x.cpu())
        return feats
    @torch.no_grad()
    def extract_flip(self, imgs):
        outputs = self.flip_model.get_image_features(pixel_values=imgs.to(self.device))
        if hasattr(outputs, "image_embeds"): features = outputs.image_embeds
        elif hasattr(outputs, "last_hidden_state"):
            vp = getattr(self.flip_model, "visual_projection", None)
            if vp is None and hasattr(self.flip_model, "vision_model"):
                vp = getattr(self.flip_model.vision_model, "visual_projection", None)
            features = vp(outputs.last_hidden_state[:, 0, :]) if vp else outputs.last_hidden_state[:, 0, :]
        elif isinstance(outputs, torch.Tensor): features = outputs
        else: raise RuntimeError(f"Unknown CLIP output: {type(outputs)}")
        return features.cpu()

class SeatBuffer:
    def __init__(self, seat_label):
        self.seat = seat_label; self.class_ids = []; self.confidences = []
    def add(self, cls_id, confidence):
        self.class_ids.append(cls_id); self.confidences.append(confidence)
    def add_miss(self):
        self.class_ids.append(-1); self.confidences.append(0.0)
    @property
    def total_frames(self): return len(self.class_ids)
    @property
    def detected_frames(self): return sum(1 for c in self.class_ids if c >= 0)
    def is_stable(self, window=None, min_conf=None, majority_pct=0.6):
        if window is None: window = STABLE_CONSECUTIVE
        if min_conf is None: min_conf = CONFIDENCE_THRESHOLD
        if self.total_frames < window: return False, -1
        recent = [c for c in self.class_ids[-window:] if c >= 0]
        if len(recent) < window * 0.3: return False, -1
        from collections import Counter
        top_id, top_count = Counter(recent).most_common(1)[0]
        if top_count / len(recent) < majority_pct: return False, -1
        min_conf_val = [self.confidences[i] for i, c in enumerate(self.class_ids[-window:]) if c == top_id]
        if not min_conf_val: return False, -1
        if sum(min_conf_val) / len(min_conf_val) < min_conf: return False, -1
        return True, top_id

class_names = ["C0", "C1", "C2", "C3"]

def draw_status(frame, left_cls, right_cls, left_conf, right_conf, left_stable, right_stable, fps, frame_count):
    h, w = frame.shape[:2]
    mid_x = w // 2
    # Draw center line
    cv2.line(frame, (mid_x, 0), (mid_x, h), (200, 200, 200), 1)
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f} | Frame: {frame_count} | Q=quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Left side info (passenger)
    left_color = (0, 255, 0) if left_cls >= 0 else (0, 0, 255)
    lines_l = [
        "PASSENGER (LEFT)",
        f"Face: {'OK' if left_cls >= 0 else 'NONE'}",
        f"Class: {class_names[left_cls] if left_cls >= 0 else '-'}",
        f"Conf: {left_conf:.3f}" if left_conf > 0 else "Conf: -",
        f"Stable: {'YES' if left_stable else 'NO'}" if left_cls >= 0 else "",
    ]
    y_pos = 55
    for line in lines_l:
        if line:
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, left_color, 1)
            y_pos += 20
    # Right side info (driver)
    right_color = (0, 255, 0) if right_cls >= 0 else (0, 0, 255)
    lines_r = [
        "DRIVER (RIGHT)",
        f"Face: {'OK' if right_cls >= 0 else 'NONE'}",
        f"Class: {class_names[right_cls] if right_cls >= 0 else '-'}",
        f"Conf: {right_conf:.3f}" if right_conf > 0 else "Conf: -",
        f"Stable: {'YES' if right_stable else 'NO'}" if right_cls >= 0 else "",
    ]
    y_pos = 55
    for line in lines_r:
        if line:
            cv2.putText(frame, line, (mid_x + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, right_color, 1)
            y_pos += 20
    return frame

@torch.no_grad()
def process_side(half_img, face_img, ext, model, device):
    """Process one side, return (cls_id, cls_conf) or (-1, 0) if no face."""
    if face_img is None:
        return -1, 0.0
    try:
        body = half_img.resize((640, 640), Image.BILINEAR)
        body_t = torch.from_numpy(np.array(body).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        face_inputs = ext.flip_processor(images=face_img, return_tensors="pt")
        face_t = face_inputs["pixel_values"]
        yf = ext.extract_yolo(body_t)
        ff = ext.extract_flip(face_t)
        _, cls_v, _ = model(yf[0].to(device), yf[1].to(device), yf[2].to(device), ff)
        cls_id = int(cls_v[0].argmax().item())
        cls_conf = float(torch.softmax(cls_v[0], dim=-1).max().item())
        return cls_id, cls_conf
    except:
        return -1, 0.0

def main():
    import argparse
    p = argparse.ArgumentParser(description="Real-time dual BMI prediction (passenger + driver) from webcam")
    p.add_argument("--camera-id", type=int, default=0, help="Camera device index")
    p.add_argument("--model-path")
    p.add_argument("--device")
    a = p.parse_args()
    
    device = torch.device(a.device) if a.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.model_path is None: a.model_path = str(Path(__file__).parent / "best_yolov8_flip_bmi_cached2.pth")
    yolo_path = str(Path(__file__).parent / "models" / "yolov8n.pt")
    flip_path = str(Path(__file__).parent / "models" / "FLIP-base-16")
    
    print(SEP); print("Real-time Dual BMI Prediction"); print(SEP)
    
    print("\n1. Init face detector"); fd = init_face_detector(); print()
    if fd is None: print("  WARNING: Face detector failed"); return
    
    print("2. Init feature extractor"); ext = RealtimeFeatureExtractor(yolo_path, flip_path, device); print()
    
    print("3. Load model")
    if not os.path.exists(a.model_path): print(f"  Not found: {a.model_path}"); return
    model = CachedYOLOv8FLIPModel()
    ckpt = torch.load(a.model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "state_dict"]:
            if k in ckpt: model.load_state_dict(ckpt[k], strict=False); break
        else: model.load_state_dict(ckpt, strict=False)
    else: model = ckpt
    model.eval().to(device); print("  Model loaded\n")
    
    print("4. Opening camera...")
    cap = cv2.VideoCapture(a.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened(): print(f"  Cannot open camera {a.camera_id}"); return
    print(f"  Camera {a.camera_id} OK\n")
    
    left_buf = SeatBuffer("副驾"); right_buf = SeatBuffer("主驾")
    frame_count = 0; start_time = time.time()
    
    print("5. Running (press Q or ESC to quit)...")
    print(SEP)
    
    while frame_count < MAX_FRAMES:
        ret, frame_bgr = cap.read()
        if not ret: break
        frame_count += 1
        fps = frame_count / (time.time() - start_time)
        
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        left_pil = pil_img.crop((0, 0, w // 2, h))
        right_pil = pil_img.crop((w // 2, 0, w, h))
        
        left_faces = detect_faces(left_pil, fd)
        right_faces = detect_faces(right_pil, fd)
        left_face = left_faces[0][0] if left_faces else None
        right_face = right_faces[0][0] if right_faces else None
        
        # Process left
        l_cls, l_conf = process_side(left_pil, left_face, ext, model, device)
        if l_cls >= 0: left_buf.add(l_cls, l_conf)
        else: left_buf.add_miss()
        
        # Process right
        r_cls, r_conf = process_side(right_pil, right_face, ext, model, device)
        if r_cls >= 0: right_buf.add(r_cls, r_conf)
        else: right_buf.add_miss()
        
        l_stable, l_final = left_buf.is_stable()
        r_stable, r_final = right_buf.is_stable()
        
        # Draw overlay
        display = frame_bgr.copy()
        display = draw_status(display, l_final if l_stable else l_cls, r_final if r_stable else r_cls,
                              l_conf, r_conf, l_stable, r_stable, fps, frame_count)
        
        cv2.imshow("Dual BMI Prediction (Q=quit)", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(); print(SEP); print("6. Final Results"); print(SEP)
    
    def final_cls(buf):
        s, c = buf.is_stable()
        return c if s else buf.class_ids[-1] if buf.class_ids else -1
    
    l_c = final_cls(left_buf); r_c = final_cls(right_buf)
    l_str = class_names[l_c] if l_c >= 0 else "no_face"
    r_str = class_names[r_c] if r_c >= 0 else "no_face"
    
    print(f"  Total frames: {frame_count}")
    print(f"  Left (passenger):  {left_buf.detected_frames}/{left_buf.total_frames} faces, class={l_str}")
    print(f"  Right (driver):    {right_buf.detected_frames}/{right_buf.total_frames} faces, class={r_str}")
    print()
    
    out_dir = Path(__file__).parent
    csv_path = str(out_dir / f"dual_bmi_cam{a.camera_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Camera", "LeftClassID", "RightClassID", "LeftFaces", "RightFaces"])
        w.writerow([a.camera_id, l_str, r_str, left_buf.detected_frames, right_buf.detected_frames])
    print(f"  CSV: {csv_path}")
    print("Done!")

if __name__ == "__main__":
    main()
