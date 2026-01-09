#!/bin/bash
# 启动教师模型服务（用于在线蒸馏）
# 使用 vLLM 的 OpenAI API 兼容服务器

set -x

# 教师模型配置
TEACHER_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
PORT=15555
TP_SIZE=2  # Tensor Parallel Size，根据 GPU 数量调整
GPU_MEMORY_UTIL=0.9
MAX_MODEL_LEN=32768

# 检查端口是否被占用
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null ; then
    echo "Port ${PORT} is already in use. Killing existing process..."
    kill -9 $(lsof -ti:${PORT})
    sleep 2
fi

echo "Starting teacher model server..."
echo "Model: ${TEACHER_MODEL}"
echo "Port: ${PORT}"
echo "Tensor Parallel Size: ${TP_SIZE}"

# 启动 vLLM 服务器
vllm serve ${TEACHER_MODEL} \
    --port ${PORT} \
    --tensor-parallel-size ${TP_SIZE} \
    --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --trust-remote-code \
    --disable-log-requests

# 如果你想在后台运行，使用以下命令：
# nohup vllm serve ${TEACHER_MODEL} \
#     --port ${PORT} \
#     --tensor-parallel-size ${TP_SIZE} \
#     --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
#     --max-model-len ${MAX_MODEL_LEN} \
#     --trust-remote-code \
#     --disable-log-requests \
#     > teacher_server.log 2>&1 &

# echo "Teacher server started in background. Check teacher_server.log for logs."
# echo "To stop: kill -9 \$(lsof -ti:${PORT})"
