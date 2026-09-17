"""
api/server.py
============
FastAPI REST API 服务：提供 HTTP 接口访问人脸识别能力。

Endpoints:
  POST /recognize/image     - 识别单张图片
  POST /recognize/batch     - 批量识别图片
  POST /recognize/video     - 处理视频文件（异步）
  GET  /recognize/result/{job_id}  - 获取视频处理结果

  GET  /gallery             - 获取 Gallery 中所有身份
  POST /gallery/register    - 手动注册身份
  POST /gallery/unregister  - 删除身份
  POST /gallery/clear       - 清空 Gallery
  POST /gallery/swap        - 主驾/副驾互换（兼容旧接口）

  GET  /health              - 健康检查
  GET  /stats               - 运行统计

启动方式：
  uvicorn api.server:app --host 0.0.0.0 --port 7860 --reload
  或
  python -m api.server

生产部署（注意：Gallery 基于内存+pickle 持久化，仅支持单 worker）：
  uvicorn api.server:app --host 0.0.0.0 --port 7860 --workers 1 --timeout-keep-alive 60
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import asyncio
import tempfile
import base64
import traceback
import html as html_lib
import queue
import logging
import threading
import webbrowser
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

# 确保项目根目录在 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent

# 加载 .env（不依赖 shell；shell 已设置的同名变量优先）
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=False)

# ── 设备主开关 DEVICE=cuda|cpu ──
# 一个变量统一控制所有推理后端。各个变量仍可独立覆盖。
_DEVICE = os.getenv("DEVICE", "cuda").lower()
_ORT_DEFAULT = "CUDAExecutionProvider" if _DEVICE == "cuda" else "CPUExecutionProvider"
_DET_CTX_DEFAULT = "0" if _DEVICE == "cuda" else "-1"
_GA_DEVICE_DEFAULT = "gpu" if _DEVICE == "cuda" else "cpu"

os.environ.setdefault("ORT_PROVIDERS", _ORT_DEFAULT)
os.environ.setdefault("DETECTOR_CTX_ID", _DET_CTX_DEFAULT)
os.environ.setdefault("GENDER_AGE_DEVICE", _GA_DEVICE_DEFAULT)
os.environ.setdefault("GENDER_AGE_ORT_PROVIDERS", _ORT_DEFAULT)


def configure_logging():
    """统一配置日志。LOG_LEVEL 环境变量控制 stdout 级别（默认 INFO），文件始终 DEBUG。"""
    _LEVEL_MAP = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}
    _log_level = _LEVEL_MAP.get(os.getenv("LOG_LEVEL", "info").lower(), logging.INFO)
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "server.log", encoding="utf-8"),
        ],
    )
    # StreamHandler 输出级别受 LOG_LEVEL 控制，文件始终 DEBUG+
    logging.getLogger().handlers[0].setLevel(_log_level)
    # 抑制第三方库的 DEBUG
    for lib in ("uvicorn", "insightface", "onnxruntime", "PIL", "matplotlib"):
        logging.getLogger(lib).setLevel(logging.WARNING)
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
import cv2

from core.gallery import Gallery, DEFAULT_SIM_THRESHOLD
from service.detector import FaceDetector, DetectedFace
from service.recognizer import VehicleRecognizer
from service.ipc import IPCClient
try:
    from service.height_estimator import HeightEstimator
except ImportError:
    HeightEstimator = None
try:
    from service.bmi_predictor import BMIPredictor
except ImportError:
    BMIPredictor = None

log = logging.getLogger("api")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))
GALLERY_FILE = os.getenv("GALLERY_FILE", str(PROJECT_ROOT / "data" / "gallery.pkl"))
HEIGHT_MODELS_DIR = os.getenv("HEIGHT_MODELS_DIR", str(PROJECT_ROOT / "models" / "height"))
CAMERA_ID_RAW = os.getenv("CAMERA_ID", "0")
try:
    CAMERA_ID = int(CAMERA_ID_RAW)
except ValueError:
    CAMERA_ID = CAMERA_ID_RAW  # 视频文件路径
UVICORN_WORKERS = int(os.getenv("UVICORN_WORKERS", "1"))
AUTO_OPEN_BROWSER = os.getenv("AUTO_OPEN_BROWSER", "1").lower() not in ("0", "false", "no", "off")

# 识别参数（环境变量可覆盖默认值）
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", str(DEFAULT_SIM_THRESHOLD)))
SMOOTH_WINDOW = int(os.getenv("SMOOTH_WINDOW", "50"))
POSITIONLESS = os.getenv("POSITIONLESS", "true").lower() not in ("0", "false", "no", "off")
AUTO_ENROLL = os.getenv("AUTO_ENROLL", "true").lower() not in ("0", "false", "no", "off")

# IPC 客户端（连接 GPU 推理进程）
_ipc: Optional[IPCClient] = None

# 视频处理任务队列
_video_jobs: dict = {}
VIDEO_JOB_TTL = int(os.getenv("VIDEO_JOB_TTL", "3600"))  # 默认1小时后清理


def _cleanup_expired_video_jobs():
    """清理过期的视频处理任务及其临时文件。"""
    now = time.time()
    expired = []
    for job_id, job in list(_video_jobs.items()):
        try:
            submitted = datetime.fromisoformat(job["submitted_at"])
            age = now - submitted.timestamp()
        except Exception:
            age = float("inf")
        if age > VIDEO_JOB_TTL:
            expired.append(job_id)
    for job_id in expired:
        job = _video_jobs.pop(job_id, None)
        if job and job.get("output"):
            try:
                os.unlink(job["output"])
            except OSError:
                pass
_bmi_predictor = None
def _open_browser_after_startup():
    """服务启动完成后自动打开本机浏览器。"""
    if not AUTO_OPEN_BROWSER:
        return
    if UVICORN_WORKERS > 1:
        return
    url = f"http://127.0.0.1:{PORT}"

    def _worker():
        try:
            webbrowser.open(url, new=2)
            log.info("已自动打开浏览器: %s", url)
        except Exception:
            log.debug("自动打开浏览器失败", exc_info=True)

    threading.Timer(1.0, _worker).start()

def shutdown_ipc():
    """关闭 IPC 连接。"""
    global _ipc
    if _ipc:
        try:
            _ipc.close()
        except Exception:
            pass
        _ipc = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时连接/断开 GPU 推理进程。"""
    global _ipc

    configure_logging()
    log.info("启动 API 服务...")

    _ipc = IPCClient()
    if not _ipc.connect(timeout=30.0):
        log.error("无法连接到 GPU 推理进程（共享内存未就绪），请先启动 inference_worker")
        raise RuntimeError("IPC connection failed — is inference_worker running?")

    log.info("已连接到 GPU 推理进程")
    _open_browser_after_startup()

    yield

    log.info("关闭 API 服务...")
    shutdown_ipc()


# ============================================================================
# FastAPI 应用
# ============================================================================

app = FastAPI(
    title="Vehicle Face Recognition API",
    description="位置无关车载人脸识别 REST API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RegisterRequest(BaseModel):
    identity_id: str = Field(..., description="身份唯一标识")
    name: Optional[str] = Field(None, description="姓名/工号")
    metadata: Optional[dict] = Field(default_factory=dict, description="额外元数据")
    image_base64: Optional[str] = Field(None, description="base64编码的人脸图片")


class RecognizeImageResponse(BaseModel):
    faces: List[dict]
    total_detected: int
    total_known: int
    process_time_ms: float
    timestamp: str


def get_ipc() -> IPCClient:
    if _ipc is None or not _ipc.connected:
        raise HTTPException(503, "GPU 推理进程未连接，请稍后重试")
    return _ipc


def get_recognizer() -> VehicleRecognizer:
    raise HTTPException(503, "Direct recognizer access not available in IPC mode")


def get_gallery() -> Gallery:
    raise HTTPException(503, "Direct gallery access not available in IPC mode")

# 部署模式：仅开放 /health /stats /recognition/stop /recognition/resume
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "").lower() in ("1", "true", "yes", "on")
_DEPLOYMENT_MSG = "此接口在部署模式下已关闭"

def require_full_api():
    """部署模式下拒绝非关键 API 调用。"""
    if DEPLOYMENT_MODE:
        raise HTTPException(503, _DEPLOYMENT_MSG)


def base64_to_image(b64: str) -> np.ndarray:
    """Base64 字符串转 cv2 图片。"""
    buf = base64.b64decode(b64)
    nparr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _json_float(value, default: float = 0.0) -> float:
    """把 numpy / Python 数值统一转成可 JSON 序列化的 float。"""
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _json_bool(value) -> bool:
    try:
        return bool(value)
    except Exception:
        return False


def _face_public_payload(face: dict) -> dict:
    """把当前帧 face 结果整理成外部调用字段。未匹配时 identity_id/name 返回空字符串。"""
    identity_id = face.get("identity_id") or ""
    name = face.get("name") or identity_id or ""
    return {
        "name": str(name),
        "identity_id": str(identity_id),
        "position": str(face.get("position") or ""),
        "similarity": round(_json_float(face.get("similarity")), 4),
        "confidence": round(_json_float(face.get("confidence")), 4),
        "is_known": _json_bool(face.get("is_known")),
        "gender": face.get("gender"),
        "gender_score": round(_json_float(face.get("gender_score")), 4),
        "gender_stable": _json_bool(face.get("gender_stable") or face.get("gender_fixed")),
        "gender_stable_source": face.get("gender_stable_source") or face.get("gender_fixed_source"),
        "gender_stable_samples": int(_json_float(face.get("gender_stable_samples") or face.get("gender_fixed_samples"), 0.0)),
        "age_group": face.get("age_group"),
        "age_score": round(_json_float(face.get("age_score")), 4),
        "age_stable": _json_bool(face.get("age_stable") or face.get("age_fixed")),
        "age_stable_source": face.get("age_stable_source") or face.get("age_fixed_source"),
        "age_stable_samples": int(_json_float(face.get("age_stable_samples") or face.get("age_fixed_samples"), 0.0)),
        "attribute_enabled": _json_bool(face.get("attribute_enabled")),
        "attribute_backend": face.get("attribute_backend"),
        "attribute_error": face.get("attribute_error"),
    }


# ============================================================================
# 识别接口
# ============================================================================

@app.post("/recognize/image", response_model=RecognizeImageResponse)
async def recognize_image(
    file: UploadFile = File(...),
    return_image: bool = Query(False, description="是否返回带标注图片"),
):
    """识别单张图片中的所有人脸（通过 IPC 发送到 GPU 进程处理）。"""
    require_full_api()
    try:
        contents = await file.read()
        # 保存到临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tf.write(contents)
            tmp_path = tf.name
        try:
            ipc = get_ipc()
            result = ipc.send_cmd({"cmd": "recognize_image", "image_path": tmp_path})
            if result.get("error"):
                raise HTTPException(500, result["error"])
            result.pop("frame_count", None)
            return result
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"识别失败: {str(e)}")


@app.post("/recognize/image_base64")
async def recognize_image_base64(body: dict):
    """识别 Base64 编码的图片（用于小程序/APP）。"""
    require_full_api()
    b64 = body.get("image", "")
    return_image = body.get("return_image", False)

    if not b64:
        raise HTTPException(400, "image 字段不能为空")

    try:
        img = base64_to_image(b64)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            cv2.imwrite(tf.name, img)
            tmp_path = tf.name
        try:
            ipc = get_ipc()
            result = ipc.send_cmd({"cmd": "recognize_image", "image_path": tmp_path})
            if result.get("error"):
                raise HTTPException(500, result["error"])
            result.pop("frame_count", None)
            return result
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        raise HTTPException(500, f"识别失败: {str(e)}")


@app.post("/recognize/batch")
async def recognize_batch(
    files: List[UploadFile] = File(...),
):
    """批量识别多张图片。"""
    require_full_api()
    if len(files) > 50:
        raise HTTPException(400, "最多支持 50 张图片批量识别")

    results = []
    for f in files:
        try:
            contents = await f.read()
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                tf.write(contents)
                tmp_path = tf.name
            try:
                ipc = get_ipc()
                r = ipc.send_cmd({"cmd": "recognize_image", "image_path": tmp_path})
                r.pop("frame_count", None)
                results.append({"filename": f.filename, "result": r})
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    return {"total": len(files), "results": results}


@app.post("/recognize/video")
async def recognize_video(
    file: UploadFile = File(...),
    skip_frames: int = Query(3, ge=1, le=30, description="每隔N帧处理一帧"),
):
    require_full_api()
    """
    上传视频文件并异步处理。

    返回 job_id，用于后续查询结果。
    """
    if not file.filename:
        raise HTTPException(400, "请上传视频文件")

    job_id = str(uuid.uuid4())[:8]

    # 保存到临时文件
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    output_path = tempfile.mktemp(suffix=".mp4")
    _video_jobs[job_id] = {
        "status": "processing",
        "input": tmp_path,
        "output": output_path,
        "skip_frames": skip_frames,
        "submitted_at": datetime.now().isoformat(),
        "result": None,
        "error": None,
    }

    async def process_video_async():
        try:
            ipc = get_ipc()
            r = ipc.send_cmd({
                "cmd": "process_video",
                "video_path": tmp_path,
                "output_path": output_path,
                "skip_frames": skip_frames,
            })
            if r.get("error"):
                _video_jobs[job_id]["status"] = "failed"
                _video_jobs[job_id]["error"] = r["error"]
            else:
                _video_jobs[job_id]["status"] = "completed"
                _video_jobs[job_id]["result"] = r
        except Exception as e:
            _video_jobs[job_id]["status"] = "failed"
            _video_jobs[job_id]["error"] = str(e)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    asyncio.create_task(process_video_async())
    return {"job_id": job_id, "status": "processing", "message": "视频已提交处理"}


@app.get("/recognize/result/{job_id}")
async def get_video_result(job_id: str):
    """查询视频处理结果。"""
    _cleanup_expired_video_jobs()
    job = _video_jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} 不存在")

    resp = {
        "job_id": job_id,
        "status": job["status"],
        "submitted_at": job["submitted_at"],
    }

    if job["status"] == "completed":
        resp["result"] = job["result"]
        resp["download_url"] = f"/recognize/download/{job_id}"
    elif job["status"] == "failed":
        resp["error"] = job["error"]

    return resp


@app.get("/recognize/download/{job_id}")
async def download_video(job_id: str):
    """下载处理后的视频。"""
    _cleanup_expired_video_jobs()
    job = _video_jobs.get(job_id)
    if not job or job["status"] != "completed":
        raise HTTPException(404, "视频不存在或未处理完成")

    output_path = job["output"]
    if not os.path.exists(output_path):
        raise HTTPException(404, "输出视频文件不存在")

    def iterfile():
        with open(output_path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk

    return StreamingResponse(
        iterfile(),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename=result_{job_id}.mp4"},
    )


# ============================================================================
# Gallery 管理接口
# ============================================================================

@app.get("/gallery")
async def get_gallery_info():
    """获取 Gallery 中所有身份。"""
    require_full_api()
    ipc = get_ipc()
    resp = ipc.send_cmd({"cmd": "gallery_list"})
    if resp.get("error"):
        raise HTTPException(503, resp["error"])
    gallery_dict = resp.get("gallery", {})
    identities = [
        {"identity_id": kid, "name": v.get("name", ""), "is_registered": v.get("is_registered", False)}
        for kid, v in gallery_dict.items()
    ]
    return {"count": len(identities), "identities": identities}


@app.post("/gallery/register")
async def register_identity(req: RegisterRequest):
    """手动注册身份（需提供 base64 编码的人脸图片）。"""
    require_full_api()
    ipc = get_ipc()
    embedding = None

    if req.image_base64:
        try:
            img = base64_to_image(req.image_base64)
            # 通过 IPC 命令让 GPU 进程检测人脸并返回 embedding
            import tempfile as _tmp
            with _tmp.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                cv2.imwrite(f.name, img)
                tmp_path = f.name
            try:
                resp = ipc.send_cmd({"cmd": "recognize_image", "image_path": tmp_path})
                faces = resp.get("faces", [])
                if faces:
                    embedding = faces[0].get("embedding")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except Exception as e:
            log.warning("gallery register image decode failed: %s", e)

    if embedding is None:
        raise HTTPException(400, "请在请求中提供有效的人脸图片用于注册")

    resp = ipc.send_cmd({
        "cmd": "register",
        "identity_id": req.identity_id,
        "name": req.name,
        "metadata": req.metadata,
        "embedding": embedding if isinstance(embedding, list) else embedding.tolist(),
    })
    if resp.get("error"):
        raise HTTPException(400, resp["error"])

    return {
        "status": "registered",
        "identity_id": req.identity_id,
        "name": req.name,
        "gallery_count": len(ipc.send_cmd({"cmd": "gallery_list"}).get("gallery", {})),
    }


@app.post("/gallery/register_from_frames")
async def register_from_frames(
    identity_id: str = Query(...),
    name: Optional[str] = Query(None),
    embeddings_b64: List[str] = Query(..., description="base64编码的embedding列表"),
):
    """从多帧 embedding 注册身份。"""
    require_full_api()
    ipc = get_ipc()
    embs = []
    for b64 in embeddings_b64:
        buf = base64.b64decode(b64)
        arr = np.frombuffer(buf, dtype=np.float32)
        embs.append(arr.tolist())
    if not embs:
        raise HTTPException(400, "embeddings 不能为空")
    # 使用所有 embedding 的平均值注册
    avg_emb = np.mean(np.array(embs), axis=0).tolist()
    resp = ipc.send_cmd({
        "cmd": "register", "identity_id": identity_id,
        "name": name, "embedding": avg_emb,
    })
    if resp.get("error"):
        raise HTTPException(400, resp["error"])
    return {"status": "registered", "identity_id": identity_id, "n_samples": len(embs)}


@app.post("/gallery/unregister")
async def unregister_identity(identity_id: str = Query(...)):
    """删除指定身份。"""
    require_full_api()
    ipc = get_ipc()
    resp = ipc.send_cmd({"cmd": "unregister", "identity_id": identity_id})
    return {"status": "deleted" if resp.get("ok") else "not_found", "identity_id": identity_id}


@app.post("/gallery/clear")
async def clear_gallery():
    """清空 Gallery。"""
    require_full_api()
    ipc = get_ipc()
    resp = ipc.send_cmd({"cmd": "clear_gallery"})
    return {"status": "cleared", "gallery_count": 0}


@app.post("/gallery/lock")
async def lock_gallery():
    require_full_api()
    return {"status": "locked", "message": "Gallery lock managed by GPU process"}


@app.post("/gallery/unlock")
async def unlock_gallery():
    require_full_api()
    return {"status": "unlocked", "message": "Gallery lock managed by GPU process"}


@app.get("/gallery/lock-status")
async def gallery_lock_status():
    require_full_api()
    return {"locked": False}


@app.get("/gallery/stats")
async def get_gallery_stats():
    require_full_api()
    ipc = get_ipc()
    resp = ipc.send_cmd({"cmd": "gallery_list"})
    return {"status": "ok", "stats": {"identity_count": len(resp.get("gallery", {}))}}


@app.post("/gallery/cleanup")
async def cleanup_gallery(
    min_confidence: float = Query(0.5),
    max_age_hours: float = Query(24.0),
    keep_registered: bool = Query(True),
    dry_run: bool = Query(False),
):
    require_full_api()
    return {"status": "ok", "message": "Gallery cleanup managed by GPU process"}


@app.post("/gallery/swap")
async def swap_identities(
    id_a: str = Query(..., description="身份A的ID"),
    id_b: str = Query(..., description="身份B的ID"),
):
    require_full_api()
    """互换两个身份的 embedding。"""
    ipc = get_ipc()
    # Simple swap: send command to GPU process
    resp = ipc.send_cmd({"cmd": "swap"})
    if resp.get("error"):
        raise HTTPException(503, resp["error"])
    return {"status": "swapped", "id_a": id_a, "id_b": id_b}


@app.get("/gallery/export")
async def export_gallery():
    """导出 Gallery（用于备份或迁移）。"""
    require_full_api()
    ipc = get_ipc()
    resp = ipc.send_cmd({"cmd": "gallery_list"})
    gallery_dict = resp.get("gallery", {})
    return {
        "exported_at": datetime.now().isoformat(),
        "count": len(gallery_dict),
        "identities": gallery_dict,
    }


# ============================================================================
# 健康检查与统计
# ============================================================================

# ── 枚举映射表 ─────────────────────────────────────────────────────────

_ENUM_MAP = {
    "gender": {"0": "female", "1": "male"},
    "age": {
        "0": "12-17", "1": "18-40", "2": "41-59", "3": "60-74", "4": "75+",
    },
    "cloth": {
        "0": "西装外套", "1": "薄夹克", "2": "长款大衣", "3": "羽绒服",
        "4": "长袖针织毛衣", "5": "长袖衬衫", "6": "短袖", "7": "背心",
        "8": "长袖", "9": "连帽卫衣",
    },
}

# 年龄字符串 → 数字
_AGE_TO_ID = {v: int(k) for k, v in _ENUM_MAP["age"].items()}

# 性别字符串 → 数字
_GENDER_TO_ID = {"male": 1, "female": 0}

# 衣着字符串 → 数字
_CLOTH_TO_ID = {v: int(k) for k, v in _ENUM_MAP["cloth"].items()}


def _digitize_person(face: dict, height_data: dict, cloth_data: dict, bmi_data: dict,
                     role_key: str, pos_key: str) -> Optional[dict]:
    """将人脸识别结果转为数字枚举。

    返回 None 表示该位置无人。
    有人但未识别 → identity_id=""，其余字段尽力填充。
    """
    matched = None
    for f in face.get("faces", []):
        if f.get("position") == pos_key:
            matched = f
            break
    if matched is None:
        return None  # 真的没人

    identity_id = str(matched.get("identity_id") or "")

    # 有 ID → 全套使用匹配到的数据；没 ID → 全部返回 null，不给半吊子数据
    if identity_id:
        gender_str = str(matched.get("gender") or "").strip().lower()
        gender = _GENDER_TO_ID.get(gender_str)

        age_str = str(matched.get("age_group") or "").strip()
        age = _AGE_TO_ID.get(age_str)

        cloth_str = str(cloth_data.get(role_key) or "").strip()
        cloth = _CLOTH_TO_ID.get(cloth_str)

        h = (height_data or {}).get(role_key)
        height_val = round(float(h.get("ema", 0) or 0), 1) if h else None

        bmi_obj = bmi_data.get(role_key) if bmi_data else {}
        bmi = round(float(bmi_obj.get("bmi", 0) or 0), 1) if (bmi_obj and bmi_obj.get("face_detected")) else None

        can_recv = bool(height_data.get("can_received")) if height_data else False
        return {
            "identity_id": identity_id,
            "gender": gender,
            "age": age,
            "cloth": cloth,
            "height": height_val,
            "bmi": bmi,
            "can_received": can_recv,
        }
    else:
        # 有人但未识别 → 只告诉下游有人，不给假数据
        can_recv = bool(height_data.get("can_received")) if height_data else False
        return {
            "identity_id": "",
            "gender": None,
            "age": None,
            "cloth": None,
            "height": None,
            "bmi": None,
            "can_received": can_recv,
        }


@app.get("/health")
async def health_check():
    """健康检查接口。"""
    ipc = get_ipc()
    cur = ipc.read_result()
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "service": "vehicle-face-recognition",
        "gpu_worker": "connected" if ipc.connected else "disconnected",
    }


@app.get("/stats")
async def get_stats():
    """运行时状态。数字枚举输出，无人返回 null，附带 enum_map 对照表。"""
    ipc = get_ipc()
    cur = ipc.read_result()
    height_data = cur.get("height", {})
    cloth_data = cur.get("cloth", {})
    bmi_data = cur.get("bmi_cache", {})
    seat_states = cur.get("seat_states", {"left": "empty", "right": "empty"})
    face_lost = cur.get("face_lost", False)

    def _digitize_from_held(face: dict, role_key: str) -> Optional[dict]:
        """从 held face 数据中提取数字枚举。"""
        gender_str = str(face.get("gender") or "").strip().lower()
        gender = _GENDER_TO_ID.get(gender_str)
        age_str = str(face.get("age_group") or "").strip()
        age = _AGE_TO_ID.get(age_str)
        cloth_str = str(cloth_data.get(role_key) or "").strip()
        cloth = _CLOTH_TO_ID.get(cloth_str)
        h = (height_data or {}).get(role_key)
        height_val = round(float(h.get("ema", 0) or 0), 1) if h else None
        bmi_obj = bmi_data.get(role_key) if bmi_data else {}
        bmi = round(float(bmi_obj.get("bmi", 0) or 0), 1) if (bmi_obj and bmi_obj.get("face_detected")) else None
        can_recv = bool(height_data.get("can_received")) if height_data else False
        return {
            "identity_id": str(face.get("identity_id") or ""),
            "gender": gender,
            "age": age,
            "cloth": cloth,
            "height": height_val,
            "bmi": bmi,
            "can_received": can_recv,
        }

    def _build_person(role_key: str, pos_key: str) -> Optional[dict]:
        state = seat_states.get(pos_key, "empty")
        if state == "empty":
            return None
        # 尝试正常 digitize（当前帧有新数据）
        person = _digitize_person(cur, height_data, cloth_data, bmi_data, role_key, pos_key)
        if person is not None:
            return person
        # face_lost 状态: 从 IPC faces 中取 held 数据
        for f in cur.get("faces", []):
            if f.get("position") == pos_key and f.get("identity_id"):
                return _digitize_from_held(f, role_key)
        return None

    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "running": cur.get("running", False),
        "paused": cur.get("paused", False),
        "frame_count": cur.get("frame_count", 0),
        "total_detected": cur.get("total_detected", 0),
        "total_known": cur.get("total_known", 0),
        "gallery_count": cur.get("gallery_count", 0),
        "driver": _build_person("driver", "right"),
        "passenger": _build_person("passenger", "left"),
        "face_lost": face_lost,
        "seat_states": seat_states,
        "enum_map": _ENUM_MAP,
    }


# ──── 身高估计 API ────
class SeatPersonConfig(BaseModel):
    seat_x: float = Field(0.30, ge=0.0, le=1.0)
    seat_y: float = Field(0.30, ge=0.0, le=1.0)
    gender: Optional[int] = Field(None, ge=-1, le=1)

class HeightConfig(BaseModel):
    driver: SeatPersonConfig = Field(default_factory=lambda: SeatPersonConfig(seat_x=0.30, seat_y=0.30))
    passenger: SeatPersonConfig = Field(default_factory=lambda: SeatPersonConfig(seat_x=0.30, seat_y=0.30))
    stable_mode: str = Field("reduce", pattern="^(continue|reduce|stop)$")


@app.get("/height/status")
async def get_height_status():
    require_full_api()
    ipc = get_ipc()
    cur = ipc.read_result()
    height_data = cur.get("height", {})
    return {"ready": bool(height_data), **height_data}


@app.post("/height/config")
async def set_height_config(cfg: HeightConfig):
    require_full_api()
    ipc = get_ipc()
    d_g = cfg.driver.gender if cfg.driver.gender is not None else 1
    p_g = cfg.passenger.gender if cfg.passenger.gender is not None else 1
    resp = ipc.send_cmd({
        "cmd": "set_seat_params",
        "dx": cfg.driver.seat_x, "dy": cfg.driver.seat_y,
        "px": cfg.passenger.seat_x, "py": cfg.passenger.seat_y,
        "d_gender": d_g, "p_gender": p_g,
    })
    if resp.get("error"):
        raise HTTPException(503, resp["error"])
    return {"status": "ok", "message": "配置已更新", "config": cfg.model_dump()}


@app.post("/height/stable_mode")
async def set_stable_mode(data: dict):
    require_full_api()
    # stable_mode 通过 height/config 的 stable_mode 字段一并设置
    mode = data.get("mode", "continue")
    return {"status": "ok", "message": f"stable_mode={mode} (set via /height/config)"}


@app.get("/height/health")
async def height_health():
    require_full_api()
    ipc = get_ipc()
    cur = ipc.read_result()
    height_data = cur.get("height", {})
    return {"loaded": bool(height_data), "ready": bool(height_data.get("driver") or height_data.get("passenger"))}


@app.get("/status-widget")
async def status_widget():
    """实时状态卡片。iframe 内部轮询 /stats，不需要手动刷新页面。"""
    require_full_api()
    body = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        * { box-sizing: border-box; }
        body {
          margin: 0;
          background: #1a1f26;
          color: #e7e9ea;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .wrap { padding: 12px 14px 14px; }
        .top {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr;
          gap: 8px;
          margin-bottom: 10px;
        }
        .metric {
          background: #252b33;
          border-radius: 8px;
          padding: 10px 12px;
        }
        .metric .k { font-size: 11px; color: #8899a6; margin-bottom: 4px; }
        .metric .v { font-size: 22px; color: #fff; font-weight: 800; }
        .row {
          display: grid;
          grid-template-columns: 64px 1fr;
          gap: 10px;
          align-items: stretch;
          padding: 8px 10px;
          background: #252b33;
          border-radius: 8px;
          margin-top: 8px;
        }
        .label {
          color: #a9b8c5;
          font-weight: 800;
          font-size: 13px;
          display: flex;
          flex-direction: column;
          justify-content: center;
          text-align: center;
        }
        .label span { font-size: 10px; font-weight: 500; color: #718395; }
        .empty {
          background: #30363d;
          border-radius: 7px;
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 58px;
          color: #607080;
          font-weight: 800;
        }
        .person {
          border: 1px solid rgba(46,204,113,0.35);
          background: rgba(46,204,113,0.13);
          color: #2ecc71;
          border-radius: 7px;
          padding: 8px 10px;
          min-width: 0;
        }
        .name { color: #fff; font-size: 13px; font-weight: 800; }
        .id { font-size: 11px; overflow-wrap: anywhere; margin-top: 2px; }
        .attr { color: #f1c40f; font-size: 12px; margin-top: 3px; font-weight: 700; }
        .small { color: #c3cad1; font-size: 10px; margin-top: 3px; }
        .stamp { color: #536471; font-size: 10px; margin-top: 8px; }
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="top">
          <div class="metric"><div class="k">Gallery</div><div class="v" id="gallery">--</div></div>
          <div class="metric"><div class="k">帧</div><div class="v" id="frame">--</div></div>
          <div class="metric"><div class="k">人脸</div><div class="v" id="faces">--</div></div>
        </div>
        <div id="driver"></div>
        <div id="passenger"></div>
        <div class="stamp" id="stamp">等待数据...</div>
      </div>
      <script>
        function esc(value) {
          return String(value ?? "").replace(/[&<>"']/g, ch => ({
            "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
          }[ch]));
        }
        const _EM = {gender:{0:"female",1:"male"},age:{0:"12-17",1:"18-40",2:"41-59",3:"60-74",4:"75+"},cloth:{0:"西装外套",1:"薄夹克",2:"长款大衣",3:"羽绒服",4:"长袖针织毛衣",5:"长袖衬衫",6:"短袖",7:"背心",8:"长袖",9:"连帽卫衣"}};
        function faceBlock(title, person) {
          if (!person) {
            return `<div class="row">
              <div class="label">${title}</div>
              <div class="empty">无人</div>
            </div>`;
          }
          const gender = _EM.gender[person.gender] || "...";
          const age = _EM.age[person.age] || "...";
          const cloth = _EM.cloth[person.cloth] || "...";
          const h_can = person.can_received ? "" : " ⚠️座椅参数未传递，已使用默认值";
          const h_text = person.height ? ` · 身高 ${(person.height - 5).toFixed(0)}-${(person.height + 5).toFixed(0)}cm${h_can}` : "";
          const bmi_text = person.bmi ? ` · BMI ${person.bmi.toFixed(1)}` : "";
          const id = esc(person.identity_id || "");
          return `<div class="row">
            <div class="label">${title}</div>
            <div class="person">
              <div class="name">ID: ${id}</div>
              <div class="id"></div>
              <div class="attr">性别: ${esc(gender)} · 年龄: ${esc(age)} · 衣物: ${esc(cloth)}${h_text}${bmi_text}</div>
            </div>
          </div>`;
        }
        async function refreshWidget() {
          try {
            const res = await fetch("/stats?_=" + Date.now(), { cache: "no-store" });
            const stats = await res.json();
            document.getElementById("gallery").textContent = stats.running ? "运行中" : "已停止";
            document.getElementById("frame").textContent = stats.paused ? "已暂停" : "活跃";
            document.getElementById("faces").textContent = new Date(stats.timestamp).toLocaleTimeString();
            document.getElementById("driver").innerHTML = faceBlock("主驾", stats.driver);
            document.getElementById("passenger").innerHTML = faceBlock("副驾", stats.passenger);
          } catch (e) {
            document.getElementById("stamp").textContent = "刷新失败 · " + new Date().toLocaleTimeString();
          }
        }
        refreshWidget();
        setInterval(refreshWidget, 1000);
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=body)


# ============================================================================
# 视频流预览
# ============================================================================

def _mjpeg_stream(read_fn):
    """生成 MJPEG 流的通用 helper。read_fn 返回 np.ndarray 或 None。
    跳过未变化的帧以减少 CPU/带宽开销。"""
    async def generate():
        boundary = "frame"
        last_hash = 0
        while True:
            try:
                frame = await asyncio.to_thread(read_fn)
                if frame is None or frame.size == 0:
                    await asyncio.sleep(0.03)
                    continue
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.1)
                continue

            # 快速帧变化检测：采样像素 CRC，跳过重复帧
            h = frame.shape[0] // 2
            w = frame.shape[1] // 2
            sample = frame[::h, ::w, :]  # 2x2 网格采样
            cur_hash = hash(sample.tobytes())
            if cur_hash == last_hash:
                await asyncio.sleep(0.02)
                continue
            last_hash = cur_hash

            try:
                _, buf = await asyncio.to_thread(
                    cv2.imencode,
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 65],
                )
                frame_bytes = buf.tobytes()
                yield (f"--{boundary}\r\n"
                       f"Content-Type: image/jpeg\r\n"
                       f"Content-Length: {len(frame_bytes)}\r\n\r\n").encode()
                yield frame_bytes + b"\r\n"
                await asyncio.sleep(0.02)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.02)
                continue

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/preview")
async def camera_preview():
    """实时视频流预览页面（带人脸框标注）。"""
    require_full_api()
    if CAMERA_ID is None:
        raise HTTPException(503, "摄像头未配置，请设置 CAMERA_ID 环境变量")

    ipc = get_ipc()
    return _mjpeg_stream(ipc.read_frame)


@app.get("/raw")
async def camera_raw():
    """实时原始摄像头视频流（800x600，无标注）。部署/非部署模式均可用。"""
    if CAMERA_ID is None:
        raise HTTPException(503, "摄像头未配置，请设置 CAMERA_ID 环境变量")

    ipc = get_ipc()
    return _mjpeg_stream(ipc.read_raw)


@app.get("/")
async def root():
    """车载人脸识别监控面板。"""
    body = """
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>车载人脸识别监控系统</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: #0f1419;
                color: #e7e9ea;
                min-height: 100vh;
            }
            /* ---- 顶部导航 ---- */
            .navbar {
                background: #1a1f26;
                border-bottom: 1px solid #2f3338;
                padding: 12px 24px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                position: sticky;
                top: 0;
                z-index: 100;
            }
            .navbar-title {
                font-size: 18px;
                font-weight: 700;
                color: #fff;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .navbar-title .dot {
                width: 10px; height: 10px; border-radius: 50%;
                background: #e74c3c;
            }
            .navbar-title .dot.active { background: #2ecc71; animation: pulse 2s infinite; }
            @keyframes pulse {
                0%,100% { opacity: 1; }
                50% { opacity: 0.4; }
            }
            .navbar-actions { display: flex; gap: 8px; }
            .btn {
                border: none; border-radius: 6px; padding: 8px 16px;
                font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s;
            }
            .btn-start { background: #2ecc71; color: #fff; }
            .btn-start:hover { background: #27ae60; }
            .btn-stop { background: #e74c3c; color: #fff; }
            .btn-stop:hover { background: #c0392b; }
            .btn-clear { background: #34495e; color: #ccc; }
            .btn-clear:hover { background: #4a6278; }
            .btn:disabled { opacity: 0.5; cursor: not-allowed; }

            /* ---- 主布局 ---- */
            .container {
                display: grid;
                grid-template-columns: 1fr 360px;
                gap: 16px;
                padding: 16px;
                max-width: 1400px;
                margin: 0 auto;
            }
            /* ---- 视频区 ---- */
            .video-section { background: #1a1f26; border-radius: 12px; overflow: hidden; }
            .video-header {
                padding: 12px 16px;
                border-bottom: 1px solid #2f3338;
                font-size: 13px;
                color: #8899a6;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .video-wrap { position: relative; background: #000; }
            .video-wrap img {
                width: 100%; display: block;
                min-height: 300px;
                max-height: 70vh;
                object-fit: contain;
            }
            .video-no-signal {
                width: 100%; min-height: 300px;
                display: flex; align-items: center; justify-content: center;
                color: #4a6278; font-size: 14px; flex-direction: column; gap: 12px;
            }
            .video-overlay {
                position: absolute; top: 8px; left: 8px;
                background: rgba(0,0,0,0.65); border-radius: 6px;
                padding: 6px 10px; font-size: 12px;
                display: flex; gap: 12px;
            }
            .overlay-item { display: flex; align-items: center; gap: 4px; }
            .overlay-item .val { color: #2ecc71; font-weight: 700; }
            .debug-line { font-size: 10px; color: #536471; padding: 6px 14px 0; }
            .debug-line.ok { color: #2ecc71; }
            .debug-line.err { color: #e74c3c; }
            .status-widget-frame {
                width: 100%;
                height: 310px;
                border: 0;
                display: block;
                background: #1a1f26;
            }
            /* ---- 右侧面板 ---- */
            .sidebar { display: flex; flex-direction: column; gap: 12px; }
            /* ---- 信息卡片 ---- */
            .card {
                background: #1a1f26; border-radius: 12px;
                border: 1px solid #2f3338; overflow: hidden;
            }
            .card-header {
                padding: 10px 14px;
                border-bottom: 1px solid #2f3338;
                font-size: 12px; font-weight: 700;
                color: #8899a6;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                display: flex; align-items: center; gap: 6px;
            }
            .card-header .icon { font-size: 14px; }
            .card-body { padding: 12px 14px; }
            /* ---- 位置状态 ---- */
            .positions { display: flex; flex-direction: column; gap: 8px; }
            .position-row {
                display: flex; align-items: center; gap: 10px;
                padding: 8px 10px; background: #252b33; border-radius: 8px;
            }
            .position-label {
                font-size: 11px; font-weight: 700; color: #8899a6;
                width: 44px; text-align: center;
            }
            .position-badge {
                flex: 1; padding: 4px 10px; border-radius: 6px; font-size: 12px;
                text-align: center; font-weight: 600;
            }
            .position-badge.empty { background: #2f3338; color: #536471; }
            .position-badge.driver { background: rgba(46,204,113,0.15); color: #2ecc71; border: 1px solid rgba(46,204,113,0.3); }
            .position-badge.passenger { background: rgba(52,152,219,0.15); color: #3498db; border: 1px solid rgba(52,152,219,0.3); }
            .position-badge .name { font-size: 12px; color: #fff; margin-bottom: 3px; }
            .position-badge .attr { font-size: 11px; color: #f1c40f; margin-top: 2px; }
            .position-badge .conf { font-size: 10px; opacity: 0.6; }
            /* ---- 位置状态 ---- */
            .bmi-value { font-size:11px; font-weight:600; border-radius:4px; padding:2px 6px; display:inline-block; }
            .bmi-has { color:#2ecc71; background:rgba(46,204,113,0.15); }
        </style>
    </head>
    <body>
        <nav class="navbar">
            <div class="navbar-title">
                <div class="dot" id="statusDot"></div>
                <span id="statusText">车载人脸识别监控系统</span>
            </div>
            <div class="navbar-actions">
                <button class="btn btn-start" id="btnStart" onclick="doStart()">开始</button>
                <button class="btn btn-stop" id="btnPause" onclick="doPause()">暂停</button>
                <button class="btn btn-clear" id="btnLock" onclick="doLock()">锁定</button>
                <button class="btn btn-clear" onclick="doClear()">清空</button>
                <button class="btn btn-start" id="btnRecStart" onclick="doRecStart()">🔴 开始录制</button>
                <button class="btn btn-stop" id="btnRecStop" onclick="doRecStop()" disabled>⏹ 停止录制</button>
                <span id="recStatus" style="color:#e74c3c;font-size:13px;font-weight:600;margin-left:8px;display:none"></span>
            </div>
        </nav>

        <div class="container">
            <!-- 左侧：视频流 -->
            <div class="video-section">
                <div class="video-header">
                    <span>实时视频流 · 带人脸标注</span>
                    <span id="frameInfo">--</span>
                </div>
                <div class="video-wrap">
                    <img id="videoStream" src="/preview" alt="video" onerror="showNoSignal()" onload="hideNoSignal()" />
                    <div class="video-overlay" id="videoOverlay">
                        <div class="overlay-item">帧 <span class="val" id="frameNum">--</span></div>
                        <div class="overlay-item">人脸 <span class="val" id="faceCount">--</span></div>
                        <div class="overlay-item">识别 <span class="val" id="knownCount">--</span></div>
                    </div>
                </div>
            </div>

            <!-- 右侧：数据面板 -->
            <div class="sidebar">

                <!-- 位置状态 -->
                <div class="card">
                    <div class="card-header"><span class="icon">🚗</span> 位置状态 <span id="pausedBadge" style="display:none;background:#e74c3c;color:#fff;font-size:10px;padding:1px 6px;border-radius:4px;margin-left:6px">已暂停</span></div>
                    <div class="card-body">
                        <div class="positions">
                            <div class="position-row">
                                <div class="position-label">主驾<br><span style="font-size:9px;font-weight:400">right</span></div>
                                <div class="position-badge empty" id="posRight">无人</div>
                                <div class="bmi-label" id="bmiRight" style="font-size:11px;color:#8899a6;margin-top:4px;text-align:center">BMI: --</div>
                            </div>
                            <div class="position-row">
                                <div class="position-label">副驾<br><span style="font-size:9px;font-weight:400">left</span></div>
                                <div class="position-badge empty" id="posLeft">无人</div>
                                <div class="bmi-label" id="bmiLeft" style="font-size:11px;color:#8899a6;margin-top:4px;text-align:center">BMI: --</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Body ROI 调参面板 -->
                <div class="card" id="roiPanel">
                    <div class="card-header"><span class="icon">🎯</span> Body ROI 调参</div>
                    <div class="card-body">
                        <div style="margin-bottom:10px;font-weight:700;color:#3498db">Left (副驾侧)</div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:12px">
                            <div><label style="font-size:10px;color:#8899a6">x1</label><input id="roi_lx1" type="number" step="0.01" min="0" max="1" style="width:100%;background:#252b33;border:1px solid #2f3338;color:#fff;border-radius:4px;padding:4px"></div>
                            <div><label style="font-size:10px;color:#8899a6">y1</label><input id="roi_ly1" type="number" step="0.01" min="0" max="1" style="width:100%;background:#252b33;border:1px solid #2f3338;color:#fff;border-radius:4px;padding:4px"></div>
                            <div><label style="font-size:10px;color:#8899a6">x2</label><input id="roi_lx2" type="number" step="0.01" min="0" max="1" style="width:100%;background:#252b33;border:1px solid #2f3338;color:#fff;border-radius:4px;padding:4px"></div>
                            <div><label style="font-size:10px;color:#8899a6">y2</label><input id="roi_ly2" type="number" step="0.01" min="0" max="1" style="width:100%;background:#252b33;border:1px solid #2f3338;color:#fff;border-radius:4px;padding:4px"></div>
                        </div>
                        <div style="margin-bottom:10px;font-weight:700;color:#2ecc71">Right (主驾侧)</div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:12px">
                            <div><label style="font-size:10px;color:#8899a6">x1</label><input id="roi_rx1" type="number" step="0.01" min="0" max="1" style="width:100%;background:#252b33;border:1px solid #2f3338;color:#fff;border-radius:4px;padding:4px"></div>
                            <div><label style="font-size:10px;color:#8899a6">y1</label><input id="roi_ry1" type="number" step="0.01" min="0" max="1" style="width:100%;background:#252b33;border:1px solid #2f3338;color:#fff;border-radius:4px;padding:4px"></div>
                            <div><label style="font-size:10px;color:#8899a6">x2</label><input id="roi_rx2" type="number" step="0.01" min="0" max="1" style="width:100%;background:#252b33;border:1px solid #2f3338;color:#fff;border-radius:4px;padding:4px"></div>
                            <div><label style="font-size:10px;color:#8899a6">y2</label><input id="roi_ry2" type="number" step="0.01" min="0" max="1" style="width:100%;background:#252b33;border:1px solid #2f3338;color:#fff;border-radius:4px;padding:4px"></div>
                        </div>
                        <button class="btn btn-start" onclick="saveRoi()" style="width:100%">保存 ROI</button>
                        <div id="roiMsg" style="font-size:11px;color:#2ecc71;margin-top:6px;display:none"></div>
                    </div>
                </div>

            </div>
        </div>

        <script>
            // ---- 工具 ----
            function showNoSignal() {
                const wrap = document.querySelector(".video-wrap");
                let noSig = wrap.querySelector(".video-no-signal");
                if (!noSig) {
                    noSig = document.createElement("div");
                    noSig.className = "video-no-signal";
                    noSig.innerHTML = '<div style="font-size:32px">📷</div><div>无视频信号</div>';
                    wrap.appendChild(noSig);
                }
                noSig.style.display = "flex";
            }
            function hideNoSignal() {
                const noSig = document.querySelector(".video-no-signal");
                if (noSig) noSig.style.display = "none";
            }

            // ---- API 操作 ----
            async function fetchJSON(url, method, body) {
                try {
                    const finalUrl = method && method !== "GET"
                        ? url
                        : url + (url.includes("?") ? "&" : "?") + "_=" + Date.now();
                    const res = await fetch(finalUrl, {
                        method: method || "GET",
                        cache: "no-store",
                        headers: body ? {"Content-Type": "application/json"} : {},
                        body: body ? JSON.stringify(body) : undefined,
                    });
                    if (!res.ok) return null;
                    return await res.json();
                } catch (e) {
                    return null;
                }
            }
            async function doStart() {
                const r = await fetchJSON("/recognition/start", "POST");
                if (r) syncFromResponse(r);
            }
            async function doPause() {
                const isPaused = document.getElementById("pausedBadge").style.display !== "none";
                if (isPaused) {
                    const r = await fetchJSON("/recognition/resume", "POST");
                    if (r) syncFromResponse(r);
                } else {
                    const r = await fetchJSON("/recognition/stop", "POST");
                    if (r) syncFromResponse(r);
                }
            }
            async function doClear() {
                if (!confirm("确定要清空 Gallery 吗？")) return;
                await fetchJSON("/gallery/clear", "POST");
            }
            async function doLock() {
                const btn = document.getElementById("btnLock");
                if (btn.textContent === "锁定") {
                    if (!confirm("锁定后 Gallery 将固定，不会自动注册新人或更新已有身份。确定锁定？")) return;
                    const r = await fetchJSON("/gallery/lock", "POST");
                    if (r) syncLockStatus(true);
                } else {
                    const r = await fetchJSON("/gallery/unlock", "POST");
                    if (r) syncLockStatus(false);
                }
            }
            function syncLockStatus(locked) {
                const btn = document.getElementById("btnLock");
                if (locked) {
                    btn.textContent = "解锁";
                    btn.style.background = "#e67e22";
                    btn.style.color = "#fff";
                } else {
                    btn.textContent = "锁定";
                    btn.style.background = "";
                    btn.style.color = "";
                }
            }

            // ---- 渲染 ----
            const _EM = {gender:{"0":"female","1":"male"},age:{"0":"12-17","1":"18-40","2":"41-59","3":"60-74","4":"75+"},cloth:{"0":"西装外套","1":"薄夹克","2":"长款大衣","3":"羽绒服","4":"长袖针织毛衣","5":"长袖衬衫","6":"短袖","7":"背心","8":"长袖","9":"连帽卫衣"}};
            function renderPosition(pos, data) {
                const el = document.getElementById("pos" + pos.charAt(0).toUpperCase() + pos.slice(1));
                if (!data) {
                    el.className = "position-badge empty";
                    el.textContent = "无人";
                } else {
                    const isDriver = pos === "right";
                    const gender = _EM.gender[data.gender] || "...";
                    const ageLabel = _EM.age[data.age] || "...";
                    const clothLabel = _EM.cloth[data.cloth] || "...";
                    const id = data.identity_id || "未知";
                    const h_can = data.can_received ? "" : " ⚠️座椅参数未传递，已使用默认值";
                    const h_text = data.height ? `身高 ${(data.height - 5).toFixed(0)}-${(data.height + 5).toFixed(0)}cm${h_can}` : "";
                    const b_text = data.bmi ? `BMI ${data.bmi.toFixed(1)}` : "";
                    el.className = "position-badge " + (isDriver ? "driver" : "passenger");
                    el.innerHTML = `<div class="name">${id}</div>
                        <div class="attr">${gender} · ${ageLabel} · ${clothLabel}</div>
                        <div class="conf">${h_text}${h_text && b_text ? " · " : ""}${b_text}</div>`;
                }
            }

            // ---- 状态同步 ----
            function syncFromResponse(r) {
                const isRunning = r.running !== undefined ? r.running : true;
                const isPaused = r.paused !== undefined ? r.paused : false;
                syncStatus(isRunning, isPaused);
            }
            function syncStatus(isRunning, isPaused) {
                const dot = document.getElementById("statusDot");
                const text = document.getElementById("statusText");
                const badge = document.getElementById("pausedBadge");
                const btn = document.getElementById("btnPause");
                const btnStart = document.getElementById("btnStart");

                if (!isRunning) {
                    dot.className = "dot";
                    text.textContent = "识别已停止";
                    btn.textContent = "开始";
                    btnStart.disabled = false;
                    btn.disabled = true;
                    badge.style.display = "none";
                } else if (isPaused) {
                    dot.className = "dot";
                    text.textContent = "识别已暂停";
                    btn.textContent = "继续";
                    btnStart.disabled = false;
                    btn.disabled = false;
                    badge.style.display = "inline";
                } else {
                    dot.className = "dot active";
                    text.textContent = "识别运行中";
                    btn.textContent = "暂停";
                    btnStart.disabled = true;
                    btn.disabled = false;
                    badge.style.display = "none";
                }
            }

            // ---- 刷新 ----
            async function doRecStart() {
                const r = await fetchJSON("/recording/start", "POST");
                if (r) syncRecStatus(true, 0, 7500);
            }
            async function doRecStop() {
                const r = await fetchJSON("/recording/stop", "POST");
                if (r) syncRecStatus(false, 0, 7500);
            }
            function syncRecStatus(recording, frames, maxFrames) {
                document.getElementById("btnRecStart").disabled = recording;
                document.getElementById("btnRecStop").disabled = !recording;
                const el = document.getElementById("recStatus");
                if (recording) {
                    el.style.display = "inline";
                    el.textContent = "● 录制中 " + frames + "/" + maxFrames;
                } else {
                    el.style.display = "none";
                }
            }

            async function refresh() {
                const statsR = await fetchJSON("/stats");
                if (!statsR) {
                    renderPosition("left", null);
                    renderPosition("right", null);
                    return;
                }
                syncStatus(statsR.running, statsR.paused);
                renderPosition("left", statsR.passenger);
                renderPosition("right", statsR.driver);

                // 视频叠加层
                document.getElementById("frameNum").textContent = statsR.frame_count ?? "--";
                document.getElementById("faceCount").textContent = statsR.total_detected ?? "--";
                document.getElementById("knownCount").textContent = statsR.total_known ?? "--";
                document.getElementById("frameInfo").textContent = "帧 " + (statsR.frame_count ?? "--");

                // BMI 显示：无人脸或变化 < 0.3 不更新，避免数值跳动
                const BMI_MIN_DELTA = 0.3;
                const bmiLeft = document.getElementById("bmiLeft");
                const bmiRight = document.getElementById("bmiRight");
                if (bmiLeft) {
                    const pb = statsR.passenger;
                    if (pb && typeof pb.bmi === "number") {
                        if (lastValidBMI.left == null || Math.abs(pb.bmi - lastValidBMI.left) >= BMI_MIN_DELTA) {
                            lastValidBMI.left = pb.bmi;
                        }
                    } else {
                        // 无人脸时清除缓存，不残留旧值
                        lastValidBMI.left = null;
                    }
                    bmiLeft.textContent = lastValidBMI.left != null ? "BMI: " + lastValidBMI.left.toFixed(1) : "BMI: --";
                }
                if (bmiRight) {
                    const db = statsR.driver;
                    if (db && typeof db.bmi === "number") {
                        if (lastValidBMI.right == null || Math.abs(db.bmi - lastValidBMI.right) >= BMI_MIN_DELTA) {
                            lastValidBMI.right = db.bmi;
                        }
                    } else {
                        // 无人脸时清除缓存，不残留旧值
                        lastValidBMI.right = null;
                    }
                    bmiRight.textContent = lastValidBMI.right != null ? "BMI: " + lastValidBMI.right.toFixed(1) : "BMI: --";
                }

                const lockR = await fetchJSON("/gallery/lock-status");
                if (lockR) syncLockStatus(lockR.locked);
            }


            // ── ROI 调参 ──
            async function loadRoi() {
                const r = await fetchJSON("/body-detection/roi");
                if (!r || !r.rois) return;
                const setVal = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
                setVal("roi_lx1", r.rois.left[0]); setVal("roi_ly1", r.rois.left[1]);
                setVal("roi_lx2", r.rois.left[2]); setVal("roi_ly2", r.rois.left[3]);
                setVal("roi_rx1", r.rois.right[0]); setVal("roi_ry1", r.rois.right[1]);
                setVal("roi_rx2", r.rois.right[2]); setVal("roi_ry2", r.rois.right[3]);
            }
            async function saveRoi() {
                const getVal = (id) => parseFloat(document.getElementById(id).value) || 0;
                const body = {
                    rois: {
                        left: [getVal("roi_lx1"), getVal("roi_ly1"), getVal("roi_lx2"), getVal("roi_ly2")],
                        right: [getVal("roi_rx1"), getVal("roi_ry1"), getVal("roi_rx2"), getVal("roi_ry2")],
                    }
                };
                const r = await fetchJSON("/body-detection/roi", "POST", body);
                const msg = document.getElementById("roiMsg");
                if (r) {
                    msg.style.display = "block";
                    msg.textContent = "ROI 已更新";
                    setTimeout(() => { msg.style.display = "none"; }, 2000);
                }
            }
            loadRoi();

            // ---- 启动 ----
            refresh();
            setInterval(refresh, 1000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=body)



@app.post("/recognition/start")
async def recognition_start():
    require_full_api()
    ipc = get_ipc()
    ipc.send_cmd({"cmd": "resume"})
    cur = ipc.read_result()
    return {"status": "started", "message": "Recognition managed by GPU process", "running": cur.get("running", True), "paused": False}


@app.post("/recognition/stop")
async def recognition_stop():
    """暂停识别。"""
    ipc = get_ipc()
    resp = ipc.send_cmd({"cmd": "pause"})
    if resp.get("error"):
        raise HTTPException(503, resp["error"])
    return {"status": "paused", "message": "识别已暂停"}


@app.post("/recognition/resume")
async def recognition_resume():
    """继续已暂停的识别。"""
    ipc = get_ipc()
    resp = ipc.send_cmd({"cmd": "resume"})
    if resp.get("error"):
        raise HTTPException(503, resp["error"])
    return {"status": "resumed", "message": "识别已恢复"}


@app.post("/recording/start")
async def recording_start():
    ipc = get_ipc()
    resp = ipc.send_cmd({"cmd": "start_recording"})
    cur = ipc.read_result()
    return {"recording": cur.get("recording", False), "frames": cur.get("recording_frames", 0)}

@app.post("/recording/stop")
async def recording_stop():
    ipc = get_ipc()
    resp = ipc.send_cmd({"cmd": "stop_recording"})
    cur = ipc.read_result()
    return {"recording": cur.get("recording", False), "frames": cur.get("recording_frames", 0)}

@app.get("/recording/status")
async def recording_status():
    ipc = get_ipc()
    resp = ipc.send_cmd({"cmd": "recording_status"})
    if resp.get("ok"):
        return {"recording": resp.get("recording", False), "frames": resp.get("frames", 0), "max_frames": resp.get("max_frames", 7500)}
    cur = ipc.read_result()
    return {"recording": cur.get("recording", False), "frames": cur.get("recording_frames", 0), "max_frames": 7500}

@app.get("/recognition/status")
async def recognition_status():
    """查询识别线程状态。"""
    require_full_api()
    ipc = get_ipc()
    cur = ipc.read_result()
    return {
        "running": cur.get("running", False),
        "paused": cur.get("paused", False),
        "current_frame": cur.get("frame_count", 0),
        "current_faces": cur.get("total_detected", 0),
        "current_known": cur.get("total_known", 0),
        "current_cloth": cur.get("cloth", {}),
    }


@app.get("/seat-detection/config")
async def seat_detection_config():
    require_full_api()
    ipc = get_ipc()
    cur = ipc.read_result()
    return {
        "enabled": True,
        "mode": "auto",
        "seat_positions": cur.get("seat_positions", {"left": False, "right": False}),
        "description": "占位检测由 GPU 进程自适应帧率调度管理",
    }


@app.post("/seat-detection/enable")
async def seat_detection_enable():
    return {"status": "ok", "message": "Seat detection is always enabled in IPC mode"}


@app.post("/seat-detection/disable")
async def seat_detection_disable():
    return {"status": "ok", "message": "Seat detection is managed by GPU process"}


@app.get("/body-detection/roi")
async def get_body_roi():
    """获取当前 body detection ROI 配置。"""
    from utils import get_body_rois
    return {"status": "ok", "rois": get_body_rois()}


@app.post("/body-detection/roi")
async def set_body_roi(data: dict):
    """动态设置 body detection ROI。

    body: {"rois": {"left": [x1, y1, x2, y2], "right": [x1, y1, x2, y2]}}
    """
    rois = data.get("rois", {})
    left = rois.get("left", [0.0, 0.05, 0.40, 0.80])
    right = rois.get("right", [0.52, 0.05, 1.0, 0.80])
    # 更新 API 进程自身的 utils
    from utils import set_body_rois, get_body_rois
    set_body_rois(left, right)
    # 同步到 GPU worker 进程
    ipc = get_ipc()
    ipc.send_cmd({"cmd": "set_body_roi", "left": left, "right": right})
    log.info("Body ROI updated: %s", get_body_rois())
    return {"status": "ok", "rois": get_body_rois()}


# ============================================================================
# 主入口
# ============================================================================

def main():
    print(f"""
============================================================
  Vehicle Face Recognition API Server
  ============================================================
  Host  : {HOST}
  Port  : {PORT}
  Gallery: {GALLERY_FILE}
  Workers: {UVICORN_WORKERS}
============================================================
    """)
    uvicorn.run(
        "api.server:app",
        host=HOST,
        port=PORT,
        workers=UVICORN_WORKERS if UVICORN_WORKERS > 1 else 1,
        timeout_keep_alive=60,
        log_level="info",
    )


if __name__ == "__main__":
    main()
