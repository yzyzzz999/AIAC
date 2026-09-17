"""
core/gallery.py
==============
位置无关人脸识别 Gallery 核心模块。

Gallery 结构：
  {
    "identity_id_1": PersonEntry(embedding, name, n_samples, ...),
    "identity_id_2": PersonEntry(embedding, name, n_samples, ...),
    ...
  }

身份 ID 由调用方指定（如 "driver_001", "passenger_001", 或任意字符串）。
不再与物理座位位置绑定。
"""

from __future__ import annotations

import pickle
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


# ============================================================================
# 数学工具函数（与 gallery_manager.py 保持一致）
# ============================================================================

def normalize_emb(emb: np.ndarray) -> np.ndarray:
    """L2 归一化 embedding。"""
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的 cosine similarity（假设已归一化）。"""
    return float(np.dot(a, b))

log = logging.getLogger(__name__)


# ============================================================================
# 配置
# ============================================================================

DEFAULT_SIM_THRESHOLD = 0.40    # 识别阈值（提高以减少误识）
DEFAULT_MIN_ENROLL = 10         # 自动注册最少帧数
DEFAULT_AUTO_ENROLL_SIM = 0.30  # 自动注册相似度上限（降低以减少误注册）
DEFAULT_UPDATE_SIM = 0.50       # 增量更新阈值（提高以防止陌生人特征混入）


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class PersonEntry:
    """
    单个身份记录。

    embedding  : 归一化后的 centroid 向量（512维）
    name       : 可读名称（可选）
    n_samples  : 用于计算 centroid 的样本帧数
    is_registered: 是否正式注册（True=已确认身份，False=临时）
    confidence : 置信度 = n_samples / (n_samples + MIN_ENROLL_FRAMES)
    created_at : 创建时间戳
    updated_at : 更新时间戳
    metadata   : 额外元数据（如性别、年龄、ID卡号等）
    """
    embedding: np.ndarray
    name: Optional[str] = None
    n_samples: int = 1
    is_registered: bool = True
    confidence: float = 0.0
    created_at: float = field(default_factory=lambda: __import__("time").time())
    updated_at: float = field(default_factory=lambda: __import__("time").time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.confidence = self.n_samples / (self.n_samples + DEFAULT_MIN_ENROLL)

    def update(self, new_emb: np.ndarray, n_new: int = 1):
        """
        增量更新 centroid。
        公式: new_centroid = (old_centroid * n + new_emb) / (n + 1)
        """
        total = self.n_samples + n_new
        self.embedding = normalize_emb(
            (self.embedding * self.n_samples + new_emb * n_new) / total
        )
        self.n_samples = total
        self.updated_at = __import__("time").time()
        self.confidence = self.n_samples / (self.n_samples + DEFAULT_MIN_ENROLL)


# ============================================================================
# Gallery 核心类
# ============================================================================

class Gallery:
    """
    人脸身份库（核心数据结构）。

    与原 gallery_manager 的区别：
    - 不再区分 driver/passenger 槽位
    - 身份 ID 由调用方指定，支持任意字符串
    - 支持增量更新、置信度计算、自动注册
    - 线程安全
    - 持久化到 pkl 文件
    """

    def __init__(
        self,
        gallery_file: str | Path = "gallery.pkl",
        sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    ):
        self.gallery_file = Path(gallery_file)
        self.sim_threshold = sim_threshold
        self._gallery: Dict[str, PersonEntry] = {}
        self._lock = threading.RLock()
        self._save_timer = 120.0  # 异步保存间隔（秒）
        self._save_debounce_time = 0.0
        self._save_lock = threading.Lock()
        self._save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gallery_save")
        self._locked = False  # 锁定模式：固定 Gallery，禁止更新和注册

        if self.gallery_file.exists():
            self._load()
        else:
            log.info(f"Gallery 文件不存在，将创建新 Gallery: {self.gallery_file}")

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _load(self):
        loaded = None
        for path in (self.gallery_file, str(self.gallery_file) + ".bak"):
            try:
                with open(path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    log.info(f"Gallery 已加载: {len(loaded)} 个身份 | {path}")
                    break
            except Exception:
                log.warning(f"Gallery 加载失败: {path}", exc_info=True)
        with self._lock:
            self._gallery = loaded if isinstance(loaded, dict) else {}
        if not isinstance(loaded, dict):
            log.warning("Gallery 文件不可用，从空库启动")

    def save(self) -> bool:
        """同步保存（阻塞），由调用方决定是否使用。"""
        return self._do_save()

    def save_async(self):
        """后台保存，避免在视频识别线程里阻塞。"""
        self._async_save()

    def _do_save(self) -> bool:
        """执行实际保存操作，线程安全。"""
        import os as _os
        tmp_path = str(self.gallery_file) + f".tmp_{id(self)}.pkl"
        try:
            with self._lock:
                snapshot = dict(self._gallery)
            with open(tmp_path, "wb") as f:
                pickle.dump(snapshot, f)
            _os.replace(tmp_path, str(self.gallery_file))
            bak_path = str(self.gallery_file) + ".bak"
            try:
                with open(bak_path, "wb") as f:
                    pickle.dump(snapshot, f)
            except OSError:
                pass
            return True
        except OSError:
            pass
        finally:
            try:
                if _os.path.exists(tmp_path):
                    _os.unlink(tmp_path)
            except OSError:
                pass

    def _async_save(self):
        """异步保存：防抖（每次调用后延迟保存），不阻塞调用方。"""
        with self._save_lock:
            now = time.time()
            if now - self._save_debounce_time < self._save_timer:
                return
            self._save_debounce_time = now

        def _worker():
            self._do_save()
            with self._save_lock:
                self._save_debounce_time = time.time()

        self._save_executor.submit(_worker)

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register(
        self,
        identity_id: str,
        embedding: np.ndarray,
        name: Optional[str] = None,
        n_samples: int = 1,
        is_registered: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        save: bool = True,
    ) -> PersonEntry:
        """
        注册或更新一个身份。

        Args:
            identity_id: 身份唯一标识（任意字符串，如 "driver_001"）
            embedding  : 人脸特征向量（512维）
            name       : 可读名称
            n_samples  : 注册用样本数
            is_registered: 是否正式注册
            metadata   : 额外元数据字典
            save       : 是否立即持久化
        """
        if self._locked:
            log.warning("Gallery 已锁定，禁止注册新身份")
            raise RuntimeError("Gallery 已锁定，无法注册新身份，请先解锁")
        emb = normalize_emb(embedding)
        entry = PersonEntry(
            embedding=emb,
            name=name,
            n_samples=n_samples,
            is_registered=is_registered,
            metadata=metadata or {},
        )
        with self._lock:
            self._gallery[identity_id] = entry

        log.info(
            f"注册身份: id={identity_id}, name={name or '(未命名)'}, "
            f"n_samples={n_samples}"
        )
        if save:
            self.save()
        return entry

    def register_from_frames(
        self,
        identity_id: str,
        embeddings: List[np.ndarray],
        name: Optional[str] = None,
        is_registered: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        save: bool = True,
    ) -> PersonEntry:
        """
        从多帧 embedding 列表计算 centroid 并注册。
        """
        if self._locked:
            log.warning("Gallery 已锁定，禁止注册新身份")
            raise RuntimeError("Gallery 已锁定，无法注册新身份，请先解锁")
        if not embeddings:
            raise ValueError("embeddings 列表为空")
        centroid = normalize_emb(np.mean(embeddings, axis=0))
        return self.register(
            identity_id=identity_id,
            embedding=centroid,
            name=name,
            n_samples=len(embeddings),
            is_registered=is_registered,
            metadata=metadata,
            save=save,
        )

    def unregister(self, identity_id: str, save: bool = True) -> bool:
        """删除指定身份。"""
        with self._lock:
            if identity_id in self._gallery:
                del self._gallery[identity_id]
                log.info(f"删除身份: {identity_id}")
                if save:
                    self.save()
                return True
        return False

    def clear(self, save: bool = True):
        """清空所有身份。"""
        with self._lock:
            self._gallery.clear()
        log.info("Gallery 已清空")
        if save:
            self.save()

    # ------------------------------------------------------------------
    # 识别
    # ------------------------------------------------------------------

    def recognize(
        self,
        embedding: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        位置无关识别：将单个 embedding 与 Gallery 中所有身份逐一比对。

        Args:
            embedding: 人脸特征向量（512维）
            threshold: 识别阈值（默认使用 self.sim_threshold）

        Returns:
            {
                "is_known"  : bool,      # 是否为已知身份
                "identity_id": str | None,# Gallery 中匹配的 ID
                "similarity" : float,     # 与匹配 ID 的 cosine 相似度
                "confidence" : float,     # Gallery 中该身份的置信度
                "name"      : str | None,# 姓名
            }
        """
        emb = normalize_emb(embedding)
        thresh = threshold if threshold is not None else self.sim_threshold

        with self._lock:
            if not self._gallery:
                return {
                    "is_known": False,
                    "identity_id": None,
                    "similarity": -1.0,
                    "confidence": 0.0,
                    "name": None,
                }

            best_id = None
            best_sim = -1.0
            for identity_id, entry in self._gallery.items():
                sim = cosine_sim(emb, entry.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_id = identity_id

            entry = self._gallery[best_id]
            return {
                "is_known": best_sim >= thresh,
                "identity_id": best_id,
                "similarity": round(float(best_sim), 4),
                "confidence": round(entry.confidence, 4),
                "name": entry.name,
            }

    def recognize_all(
        self,
        embeddings: List[np.ndarray],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量位置无关识别。

        每个人脸 embedding 与 Gallery 中所有身份逐一比对，
        返回每个人脸的识别结果（顺序与输入 embeddings 一致）。
        """
        return [self.recognize(emb, threshold) for emb in embeddings]

    def recognize_all_with_positions(
        self,
        faces: List[Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        带位置信息的批量识别。

        Args:
            faces: [{"emb": np.ndarray, "bbox": tuple, "xc": float, "yc": float,
                     "img": np.ndarray, "w": int, "h": int}, ...]
            threshold: 识别阈值

        Returns:
            每个人脸的结果（含位置信息）:
            [{
                "emb": ...,
                "bbox": ...,
                "position": "left" | "right",
                "is_known": bool,
                "identity_id": str | None,
                "similarity": float,
                "confidence": float,
                "name": str | None,
            }, ...]
        """
        thresh = threshold if threshold is not None else self.sim_threshold
        results = []

        for face in faces:
            emb = face["emb"]
            W = face.get("w", 1920)
            match = self.recognize(emb, thresh)

            results.append({
                "emb": face["emb"],
                "bbox": face.get("bbox"),
                "position": "left" if face["xc"] < W / 2 else "right",
                "is_known": match["is_known"],
                "identity_id": match["identity_id"],
                "similarity": match["similarity"],
                "confidence": match["confidence"],
                "name": match["name"],
            })

        return results

    # ------------------------------------------------------------------
    # 批量识别（矩阵加速）
    # ------------------------------------------------------------------

    def batch_recognize(
        self,
        embeddings: List[np.ndarray],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量识别（矩阵运算版，一次计算所有相似度）。

        所有人脸 embedding 与 Gallery 中所有身份同时比对，
        避免 Python for 循环带来的开销。

        Args:
            embeddings: 人脸 embedding 列表（N x 512）
            threshold : 识别阈值

        Returns:
            每人脸的识别结果，顺序与输入一致。
        """
        thresh = threshold if threshold is not None else self.sim_threshold

        with self._lock:
            if not self._gallery:
                return [
                    {
                        "is_known": False,
                        "identity_id": None,
                        "similarity": -1.0,
                        "confidence": 0.0,
                        "name": None,
                    }
                    for _ in embeddings
                ]

            ids = list(self._gallery.keys())
            gallery_matrix = np.stack([self._gallery[iid].embedding for iid in ids])
            # shape: (n_identities, 512)

        if not ids:
            return [{"is_known": False, "identity_id": None,
                     "similarity": -1.0, "confidence": 0.0, "name": None}
                    for _ in embeddings]

        emb_matrix = np.stack([normalize_emb(e) for e in embeddings])
        # shape: (n_faces, 512)

        # 全量相似度矩阵: (n_faces, n_identities)
        sim_matrix = emb_matrix @ gallery_matrix.T

        # 取每行最大值
        best_idx_per_row = sim_matrix.argmax(axis=1)
        best_sim_per_row = sim_matrix[np.arange(len(embeddings)), best_idx_per_row]

        results = []
        for row_i in range(len(embeddings)):
            best_id = ids[best_idx_per_row[row_i]]
            entry = self._gallery[best_id]
            sim = float(best_sim_per_row[row_i])
            results.append({
                "is_known": sim >= thresh,
                "identity_id": best_id,
                "similarity": round(sim, 4),
                "confidence": round(entry.confidence, 4),
                "name": entry.name,
            })
        return results

    def find_candidates(
        self,
        embedding: np.ndarray,
        top_k: int = 3,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        找出与输入 embedding 最相似的 top_k 个候选身份。

        Args:
            embedding: 人脸特征向量
            top_k     : 返回前 k 个候选
            threshold : 过滤阈值（低于此值的候选不返回）

        Returns:
            [{"identity_id": ..., "similarity": ..., "confidence": ...}, ...]
            按相似度从高到低排序。
        """
        thresh = threshold if threshold is not None else self.sim_threshold
        emb = normalize_emb(embedding)

        with self._lock:
            candidates = []
            for identity_id, entry in self._gallery.items():
                sim = cosine_sim(emb, entry.embedding)
                if sim >= thresh:
                    candidates.append({
                        "identity_id": identity_id,
                        "similarity": round(sim, 4),
                        "confidence": round(entry.confidence, 4),
                        "name": entry.name,
                    })

        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Gallery 清理
    # ------------------------------------------------------------------

    def cleanup(
        self,
        min_confidence: float = 0.5,
        max_age_hours: float = 24.0,
        keep_registered: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        清理 Gallery 中的低置信度或长期未更新的临时身份。

        Args:
            min_confidence : 仅删除置信度 < 此值的身份
            max_age_hours  : 删除超过 N 小时未更新的非正式注册身份
            keep_registered: True = 不删除 is_registered=True 的身份
            dry_run        : True = 只返回统计，不实际删除

        Returns:
            {
                "candidates": [...],   # 待删除的身份列表
                "deleted_count": int,  # 本次实际删除数量
                "remaining_count": int, # 清理后剩余数量
                "dry_run": bool,
            }
        """
        import time as _time

        now = _time.time()
        candidates = []

        with self._lock:
            for identity_id, entry in list(self._gallery.items()):
                # 保护正式注册身份
                if keep_registered and entry.is_registered:
                    continue

                # 置信度过滤
                if entry.confidence >= min_confidence:
                    continue

                # 年龄过滤（非正式注册才检查）
                age_hours = (now - entry.updated_at) / 3600
                if age_hours < max_age_hours:
                    continue

                candidates.append({
                    "identity_id": identity_id,
                    "confidence": round(entry.confidence, 4),
                    "n_samples": entry.n_samples,
                    "is_registered": entry.is_registered,
                    "age_hours": round(age_hours, 2),
                    "updated_at": round(entry.updated_at, 2),
                })

        if not dry_run:
            for cand in candidates:
                del self._gallery[cand["identity_id"]]
                log.info(f"清理身份: {cand['identity_id']} "
                         f"(conf={cand['confidence']}, age={cand['age_hours']}h)")
            self.save()

        remaining = len(self._gallery)
        return {
            "candidates": candidates,
            "deleted_count": len(candidates) if not dry_run else 0,
            "remaining_count": remaining,
            "dry_run": dry_run,
        }

    def cleanup_stats(self) -> Dict[str, Any]:
        """
        返回 Gallery 的健康统计信息，辅助判断清理策略。

        Returns:
            {
                "total": int,
                "registered_count": int,
                "temporary_count": int,
                "confidence_distribution": {"<0.3": N, "0.3-0.5": N, "0.5-0.7": N, ">0.7": N},
                "oldest_updated": float,  # 最早更新时间戳
                "newest_updated": float,
            }
        """
        import time as _time

        with self._lock:
            entries = list(self._gallery.values())

        if not entries:
            return {
                "total": 0,
                "registered_count": 0,
                "temporary_count": 0,
                "confidence_distribution": {"<0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, ">0.7": 0},
                "oldest_updated": None,
                "newest_updated": None,
            }

        conf_dist = {"<0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, ">0.7": 0}
        for e in entries:
            c = e.confidence
            if c < 0.3:
                conf_dist["<0.3"] += 1
            elif c < 0.5:
                conf_dist["0.3-0.5"] += 1
            elif c < 0.7:
                conf_dist["0.5-0.7"] += 1
            else:
                conf_dist[">0.7"] += 1

        all_times = [e.updated_at for e in entries]

        return {
            "total": len(entries),
            "registered_count": sum(1 for e in entries if e.is_registered),
            "temporary_count": sum(1 for e in entries if not e.is_registered),
            "confidence_distribution": conf_dist,
            "oldest_updated": round(min(all_times), 2),
            "newest_updated": round(max(all_times), 2),
        }

    # ------------------------------------------------------------------
    # 增量更新
    # ------------------------------------------------------------------

    def update_if_confident(
        self,
        identity_id: str,
        new_emb: np.ndarray,
        n_new: int = 1,
        sim_thresh: float = DEFAULT_UPDATE_SIM,
        save: bool = True,
    ):
        """
        如果新 embedding 与已有身份相似度高，则增量更新其 centroid。
        自动注册的身份（is_registered=False）同样参与更新，以提升置信度。
        """
        with self._lock:
            if identity_id not in self._gallery:
                return
            entry = self._gallery[identity_id]

        if self._locked:
            return
        sim = cosine_sim(normalize_emb(new_emb), entry.embedding)
        if sim >= sim_thresh:
            entry.update(new_emb, n_new)
            if save:
                self._async_save()
            log.debug(f"增量更新 {identity_id}: sim={sim:.4f}")

    # ------------------------------------------------------------------
    # 锁定模式
    # ------------------------------------------------------------------

    def lock(self):
        """锁定 Gallery：禁止注册新身份和增量更新。"""
        with self._lock:
            self._locked = True
        log.info("Gallery 已锁定，识别模式切换为稳定匹配")

    def unlock(self):
        """解锁 Gallery：恢复注册和增量更新功能。"""
        with self._lock:
            self._locked = False
        log.info("Gallery 已解锁，恢复学习和更新模式")

    def is_locked(self) -> bool:
        """返回当前锁定状态。"""
        return self._locked

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get(self, identity_id: str) -> Optional[PersonEntry]:
        with self._lock:
            return self._gallery.get(identity_id)

    def get_metadata(self, identity_id: str) -> Dict[str, Any]:
        """返回指定身份的 metadata 副本，避免调用方直接改内部对象。"""
        with self._lock:
            entry = self._gallery.get(identity_id)
            if entry is None:
                return {}
            return dict(entry.metadata or {})

    def update_metadata(
        self,
        identity_id: str,
        metadata: Dict[str, Any],
        save: bool = True,
    ) -> bool:
        """合并更新指定身份的 metadata。"""
        if not metadata:
            return False
        if self._locked:
            return False
        with self._lock:
            entry = self._gallery.get(identity_id)
            if entry is None:
                return False
            entry.metadata = dict(entry.metadata or {})
            entry.metadata.update(metadata)
            entry.updated_at = time.time()
        if save:
            self._async_save()
        return True

    def get_all(self) -> Dict[str, PersonEntry]:
        with self._lock:
            return dict(self._gallery)

    def get_all_identities(self) -> List[Dict[str, Any]]:
        """返回所有身份的摘要信息列表。"""
        with self._lock:
            return [
                {
                    "identity_id": identity_id,
                    "name": entry.name,
                    "n_samples": entry.n_samples,
                    "confidence": round(entry.confidence, 4),
                    "is_registered": entry.is_registered,
                    "created_at": round(entry.created_at, 2),
                    "updated_at": round(entry.updated_at, 2),
                    "metadata": entry.metadata,
                }
                for identity_id, entry in self._gallery.items()
            ]

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._gallery) == 0

    def count(self) -> int:
        with self._lock:
            return len(self._gallery)

    def summary(self) -> str:
        with self._lock:
            lines = [
                f"Gallery: {len(self._gallery)} identities | file: {self.gallery_file}",
                "=" * 60,
            ]
            for identity_id, entry in self._gallery.items():
                lines.append(
                    f"  [{identity_id}] name={entry.name or '(unnamed)':12s}  "
                    f"n={entry.n_samples:3d}  conf={entry.confidence:.2f}  "
                    f"registered={'Y' if entry.is_registered else 'N'}"
                )
            return "\n".join(lines)
