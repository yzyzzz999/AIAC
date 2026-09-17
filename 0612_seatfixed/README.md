# 座舱乘员感知系统 (Cabin Perception)

车内前排乘员人脸识别、性别年龄识别、身高 BMI、衣着检测、实时监控和 RGB/NIR 离线评估项目。

本版本包含实时 FastAPI 服务、Gallery 身份库、年龄稳定后固定、性别年龄 `torch` / `onnx` 后端切换，以及 RGB/NIR 多帧融合评估流程。

## Features

- 实时摄像头识别：Web 页面预览、主驾/副驾状态、实时统计。
- Gallery 身份库：自动注册、身份匹配、增量更新、锁定/解锁、持久化保存。
- 年龄稳定固定：同一 `identity_id` 多次预测年龄稳定后写入 Gallery，后续直接查询固定年龄。
- 属性识别后端：支持原始 `torch` 后端和部署友好的 `onnxruntime` 后端，预留 TensorRT 入口。
- 离线评估：读取 RGB/NIR 图片序列，分别做单模态融合，再做 RGB/NIR 跨模态融合。
- 结果输出：CSV、JSON、融合可视化图、评估指标和运行日志。

## Directory Layout

```text
cabin_perception/
├── api/
│   └── server.py                    # FastAPI 服务、Web 页面、REST API
├── core/
│   └── gallery.py                   # Gallery 身份库与 metadata 持久化
├── service/
│   ├── detector.py                  # 实时服务检测封装
│   └── recognizer.py                # 实时识别、自动注册、年龄固定
├── scripts/
│   └── export_gender_age_onnx.py    # 性别年龄模型 ONNX 导出
├── data_loader/                     # 标注文件整理脚本
├── FLIP-base-32/                    # 本地 CLIP/FLIP 模型目录
├── main.py                          # 离线评估和摄像头模式入口
├── detector.py                      # 离线流程 InsightFace 检测封装
├── gender_age_backends.py           # onnx / tensorrt 后端工厂
├── gender_age_recognizer.py         # 服务端属性识别适配器
├── requirements.txt
└── README.md
```

数据目录默认可以放在项目外层：

```text
project-root/
├── data/
│   ├── pic1/
│   │   ├── rgb/
│   │   └── nir/
│   └── ...
└── cabin_perception/
```

运行离线评估时使用 `--data-root ../data` 指向外层数据目录。

## Environment

建议使用 Python 3.10 环境。

```bash
conda create -n aiac python=3.10
conda activate aiac
cd cabin_perception
pip install -r requirements.txt
```

如果需要导出 ONNX：

```bash
pip install onnx onnxscript
```

如果使用 NVIDIA GPU，需要按目标机器 CUDA 版本安装匹配的 PyTorch、`onnxruntime-gpu` 或 TensorRT 运行环境。

## Model Preparation

### InsightFace

InsightFace 默认使用 `buffalo_l`，通常缓存于：

```text
~/.insightface/models/buffalo_l/
```

离线部署机器如果不能联网，需要提前准备该模型缓存。

### Gender / Age Model

属性识别模型从本地目录读取：

```text
FLIP-base-32/
```

默认使用 `onnx` 后端运行。首次使用 ONNX 后端前，先导出：

```bash
cd cabin_perception
python3 scripts/export_gender_age_onnx.py --device cpu
```

导出后会生成：

```text
FLIP-base-32/gender_age_image_encoder.onnx
FLIP-base-32/gender_age_text_features.npz
```

`.npz` 文件保存性别/年龄文本特征和 `logit_scale`，用于让 ONNX 概率尺度接近 torch 后端。

## Real-Time Service

启动实时服务：

```bash
cd cabin_perception

CAMERA_ID=0 \
GENDER_AGE_BACKEND=onnx \
GENDER_AGE_ONNX_MODEL=./FLIP-base-32/gender_age_image_encoder.onnx \
GENDER_AGE_TEXT_FEATURES=./FLIP-base-32/gender_age_text_features.npz \
GENDER_AGE_ORT_PROVIDERS=CPUExecutionProvider \
GENDER_AGE_SKIP_PREDICT_WHEN_AGE_FIXED=1 \
python3 -m uvicorn api.server:app --host 0.0.0.0 --port 7860
```

服务启动成功后默认自动打开：

```text
http://127.0.0.1:7860
```

不需要自动打开浏览器时：

```bash
AUTO_OPEN_BROWSER=0 python3 -m uvicorn api.server:app --host 0.0.0.0 --port 7860
```

关闭服务：

```text
在启动服务的终端按 Ctrl+C
```

关闭时会停止识别线程、释放摄像头，并保存 Gallery。

## REST API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web 监控页面 |
| `/preview` | GET | 实时标注视频流 |
| `/stats` | GET | 当前运行状态 |
| `/health` | GET | 健康检查 |
| `/docs` | GET | Swagger 文档 |
| `/recognition/start` | POST | 启动识别线程 |
| `/recognition/stop` | POST | 暂停识别 |
| `/recognition/resume` | POST | 恢复识别 |
| `/recognition/status` | GET | 识别线程状态 |
| `/gallery` | GET | 查看 Gallery |
| `/gallery/lock` | POST | 锁定 Gallery |
| `/gallery/unlock` | POST | 解锁 Gallery |
| `/gallery/clear` | POST | 清空 Gallery |
| `/recognize/image` | POST | 单张图片识别 |

## Offline Evaluation

使用 ONNX 后端评估外层 `data/` 下全部序列：

```bash
cd cabin_perception
rm -rf outputs_sequence

GENDER_AGE_ONNX_MODEL=./FLIP-base-32/gender_age_image_encoder.onnx \
GENDER_AGE_TEXT_FEATURES=./FLIP-base-32/gender_age_text_features.npz \
GENDER_AGE_ORT_PROVIDERS=CPUExecutionProvider \
python3 main.py \
  --source folder \
  --data-root ../data \
  --ctx-id -1 \
  --attr-device cpu \
  --attr-backend onnx \
  --age-thr 0.25
```

输出目录：

```text
outputs_sequence/
├── result_summary_detailed.csv       # RGB/NIR 单模态结果
├── result_summary_rgb_nir_fused.csv  # RGB/NIR 融合结果
├── eval_merged_fused.csv             # 预测与标注合并结果
├── eval_metrics_fused.json           # 评估指标
├── result_stats.json                 # 结果统计
├── run.log                           # 运行日志
├── json/                             # 每个序列逐帧 JSON
└── vis/fused/                        # 融合可视化图
```

## Fusion Logic

### Front-Row Face Selection

每帧先使用 InsightFace 检测所有人脸，再按前排 ROI 选择：

```text
face1 = 图像左侧 = 副驾
face2 = 图像右侧 = 主驾
```

选择逻辑综合 ROI 重叠比例、bbox 面积、是否靠下、检测分数，避免后排人脸误入主驾/副驾结果。

### Single-Frame Filtering

每张脸预测后先做置信度过滤：

```text
gender_score >= gender_thr
age_score >= age_thr
gender / age_group 非空
```

ONNX 与 torch 可能存在轻微数值差异。全量 ONNX 评估时可使用：

```bash
--age-thr 0.25
```

### Frame Weight

有效帧会计算融合权重：

```text
attr_score = sqrt(gender_score * age_score)
bbox_soft_score = bbox 历史稳定性
frame_weight = 0.75 * attr_score + 0.25 * bbox_soft_score
```

清晰、稳定、置信度高的帧权重更高。

### Gender Fusion

性别融合使用加权统计：

```text
weighted_count += frame_weight
weighted_score += frame_weight * gender_score
```

最终选择加权证据最强的性别。

### Age Fusion

年龄融合使用每帧完整 `age_probs`，不是只看最高标签：

```text
quality = 0.50 * frame_weight + 0.30 * age_score + 0.20 * bbox_score
age_prob_stats[label] += quality * age_probs[label]
```

之后选择累计概率最高的年龄段，并做保守后处理。

### Conservative Age Post-Processing

代码对容易混淆的相邻年龄段做保护：

```text
12-17 vs 18-40
18-40 vs 41-59
```

只有当更年轻或更老的年龄段证据明显更强时，才覆盖默认结果，以减少年龄段跳变。

### RGB/NIR Fusion

每个序列先得到：

```text
picX_rgb
picX_nir
```

再做跨模态融合：

```text
性别：优先 RGB，RGB 无结果时使用 NIR。
年龄：比较 RGB / NIR 的质量分，选择更可靠的一路。
```

年龄质量大致为：

```text
log1p(valid_frames) * age_score * frame_weight
```

有效帧越多、置信度越高、帧权重越高，该模态越可信。

## Stable Age Cache

实时服务中，同一 `identity_id` 的年龄会持续观察。默认稳定条件：

```text
AGE_STABLE_WINDOW=8
AGE_STABLE_MIN_COUNT=5
AGE_STABLE_MIN_RATIO=0.75
AGE_STABLE_MIN_SCORE=0.35
```

达到稳定后写入 Gallery metadata：

```text
fixed_age_group
fixed_age_score
fixed_age_samples
fixed_age_ratio
fixed_age_locked_at
```

之后同一 ID 再出现，默认直接查询 Gallery 固定年龄，不再调用属性模型：

```text
GENDER_AGE_SKIP_PREDICT_WHEN_AGE_FIXED=1
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `PORT` | `7860` | 服务端口 |
| `CAMERA_ID` | `0` | 摄像头 ID |
| `AUTO_OPEN_BROWSER` | `1` | 启动后是否自动打开页面 |
| `GALLERY_FILE` | `gallery.pkl` | Gallery 文件 |
| `ENABLE_GENDER_AGE` | `1` | 是否启用性别年龄 |
| `GENDER_AGE_BACKEND` | `onnx` | `onnx` / `tensorrt` |
| `GENDER_AGE_DEVICE` | empty | `cpu` / `cuda` / auto |
| `GENDER_AGE_INTERVAL` | `30` | 同一 ID/座位属性缓存帧间隔 |
| `GENDER_AGE_SKIP_PREDICT_WHEN_AGE_FIXED` | `1` | 年龄固定后跳过属性模型 |
| `GENDER_AGE_ONNX_MODEL` | `FLIP-base-32/gender_age_image_encoder.onnx` | ONNX 图像 encoder |
| `GENDER_AGE_TEXT_FEATURES` | `FLIP-base-32/gender_age_text_features.npz` | ONNX 文本特征 |
| `GENDER_AGE_ORT_PROVIDERS` | CPU | ONNXRuntime providers |
| `AGE_STABLE_WINDOW` | `8` | 年龄稳定观察窗口 |
| `AGE_STABLE_MIN_COUNT` | `5` | 同年龄段最少出现次数 |
| `AGE_STABLE_MIN_RATIO` | `0.75` | 同年龄段最小占比 |
| `AGE_STABLE_MIN_SCORE` | `0.35` | 年龄预测最低置信度 |

## Troubleshooting

- `Driver:0 0 Passenger:0 0`：先查看 `outputs_sequence/run.log` 中 `有效帧 face1/face2` 是否为 0。
- 检测到人脸但有效帧为 0：通常是属性阈值过高，可尝试 `--age-thr 0.25`。
- `No background frame`：该序列没有有效属性帧，代表帧未生成。
- 服务无法打开摄像头：更换 `CAMERA_ID=1` 或检查摄像头权限。
- 端口被占用：更换 `--port 7861` 或设置 `PORT=7861`。

## Generated Files

以下文件通常不需要提交：

```text
outputs_sequence/
__pycache__/
*.pyc
.DS_Store
logs/
server.pid
```
