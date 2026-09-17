# deploy/Dockerfile
# 多阶段构建：构建阶段 + 生产运行阶段

FROM python:3.12-slim AS builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ========================================================================
# 生产镜像
# ========================================================================

FROM python:3.12-slim

LABEL maintainer="lqz"
LABEL description="Vehicle Face Recognition Service"

WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 复制 Python 包
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 复制应用代码
COPY core/ ./core/
COPY service/ ./service/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY requirements.txt .

# 创建工作目录
RUN mkdir -p /app/logs /app/data

# 环境变量
ENV HOST=0.0.0.0
ENV PORT=7860
ENV GALLERY_FILE=/app/data/gallery.pkl
ENV PYTHONPATH=/app

# 暴露端口
EXPOSE 7860

# 启动脚本
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
