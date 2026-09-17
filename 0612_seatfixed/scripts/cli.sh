#!/bin/bash
# scripts/cli.sh
# 命令行工具：支持 recognize / gallery 管理

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
API_URL="${API_URL:-http://localhost:7860}"

CMD="${1:-help}"

usage() {
    cat <<EOF
Vehicle Face Recognition CLI
=============================

用法: ./cli.sh <命令> [参数]

识别命令:
  recognize <图片路径>           识别单张图片
  recognize-batch <目录>        批量识别目录下所有图片
  video <视频路径>             处理视频文件

Gallery 管理:
  gallery list                 查看所有已注册身份
  gallery register <ID> <图片>  注册新身份
  gallery unregister <ID>      删除身份
  gallery clear                清空 Gallery
  gallery swap <ID_A> <ID_B>  互换两个身份

服务管理:
  gallery export               导出 Gallery
  health                      健康检查

示例:
  ./cli.sh recognize /data/test.jpg
  ./cli.sh gallery list
  ./cli.sh gallery register driver_001 /data/driver.jpg
  ./cli.sh video /data/dashcam.mp4

EOF
}

# 检查 API 是否可用
check_api() {
    if ! curl -s "$API_URL/health" > /dev/null 2>&1; then
        echo "[ERROR] API 服务未启动，请先运行: ./start.sh"
        exit 1
    fi
}

case "$CMD" in
    recognize)
        IMAGE="${2:-}"
        if [ -z "$IMAGE" ]; then
            echo "[ERROR] 请指定图片路径"
            exit 1
        fi
        check_api
        echo "[INFO] 识别图片: $IMAGE"
        curl -s -X POST "$API_URL/recognize/image" \
            -F "file=@$IMAGE" \
            -F "return_image=false" | python3 -m json.tool
        ;;

    recognize-batch)
        DIR="${2:-}"
        if [ -z "$DIR" ]; then
            echo "[ERROR] 请指定目录"
            exit 1
        fi
        check_api
        echo "[INFO] 批量识别目录: $DIR"
        for img in "$DIR"/*.jpg "$DIR"/*.png "$DIR"/*.jpeg; do
            [ -f "$img" ] || continue
            echo "--- $img ---"
            curl -s -X POST "$API_URL/recognize/image" \
                -F "file=@$img" | python3 -m json.tool
        done
        ;;

    video)
        VIDEO="${2:-}"
        if [ -z "$VIDEO" ]; then
            echo "[ERROR] 请指定视频路径"
            exit 1
        fi
        check_api
        echo "[INFO] 处理视频: $VIDEO"
        RESPONSE=$(curl -s -X POST "$API_URL/recognize/video" \
            -F "file=@$VIDEO" \
            -F "skip_frames=3")
        echo "$RESPONSE" | python3 -m json.tool
        JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
        echo "[INFO] Job ID: $JOB_ID，查询结果: curl $API_URL/recognize/result/$JOB_ID"
        ;;

    gallery)
        SUBCMD="${2:-list}"
        check_api
        case "$SUBCMD" in
            list)
                curl -s "$API_URL/gallery" | python3 -m json.tool
                ;;
            register)
                ID="${3:-}"
                IMG="${4:-}"
                if [ -z "$ID" ] || [ -z "$IMG" ]; then
                    echo "[ERROR] 用法: gallery register <identity_id> <图片路径>"
                    exit 1
                fi
                curl -s -X POST "$API_URL/gallery/register" \
                    -F "identity_id=$ID" \
                    -F "file=@$IMG" | python3 -m json.tool
                ;;
            unregister)
                ID="${3:-}"
                if [ -z "$ID" ]; then
                    echo "[ERROR] 用法: gallery unregister <identity_id>"
                    exit 1
                fi
                curl -s -X POST "$API_URL/gallery/unregister?identity_id=$ID" | python3 -m json.tool
                ;;
            clear)
                echo "[WARN] 确定要清空 Gallery 吗？(y/N)"
                read -r confirm
                if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                    curl -s -X POST "$API_URL/gallery/clear" | python3 -m json.tool
                fi
                ;;
            swap)
                ID_A="${3:-}"
                ID_B="${4:-}"
                if [ -z "$ID_A" ] || [ -z "$ID_B" ]; then
                    echo "[ERROR] 用法: gallery swap <id_a> <id_b>"
                    exit 1
                fi
                curl -s -X POST "$API_URL/gallery/swap?id_a=$ID_A&id_b=$ID_B" | python3 -m json.tool
                ;;
            export)
                curl -s "$API_URL/gallery/export" | python3 -m json.tool
                ;;
            *)
                echo "[ERROR] 未知子命令: $SUBCMD"
                usage
                ;;
        esac
        ;;

    health)
        curl -s "$API_URL/health" | python3 -m json.tool
        curl -s "$API_URL/stats" | python3 -m json.tool
        ;;

    help|--help|-h|"")
        usage
        ;;
    *)
        echo "[ERROR] 未知命令: $CMD"
        usage
        ;;
esac
