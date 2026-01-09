# 快速开始指南

这是一个快速开始指南，帮助你立即运行搜索工具 GRPO + 在线蒸馏实验。

## 前置条件

1. **GPU 要求**:
   - GRPO 训练（不带蒸馏）: 至少 4 个 GPU
   - GRPO + 蒸馏训练: 至少 6-8 个 GPU（4 个用于训练，2-4 个用于教师模型）

2. **搜索服务**:
   - 确保搜索服务可访问（默认: `http://10.244.209.173:8000/retrieve`）
   - 或修改脚本中的 `retrieval_service_url` 参数

3. **数据**:
   - 确保 `data/asearcher_train/train.parquet` 存在

## 三步快速开始

### 步骤 1: 准备数据（首次运行）

```bash
cd /root/yuxiang/rllm
python3 -m examples.search.prepare_asearcher_data
```

**预期输出**:
```
Loading training data from: data/asearcher_train/train.parquet
Processed XXX training examples
Train dataset registered with XXX examples
```

### 步骤 2: 运行 GRPO 训练（验证工具）

```bash
cd /root/yuxiang/rllm
bash examples/search/train_search_grpo.sh
```

**这一步的目的**:
- 验证搜索工具是否正确工作
- 验证数据加载是否正常
- 验证训练流程是否能运行

**训练时间**: 取决于数据量和硬件，通常需要几小时

### 步骤 3: 运行 GRPO + 蒸馏训练

**3.1 启动教师模型服务**（在终端 1）:

```bash
cd /root/yuxiang/rllm

# 如果需要修改 GPU 或端口，先编辑脚本
# vim examples/search/start_teacher_server.sh

bash examples/search/start_teacher_server.sh
```

**等待教师模型服务启动完成**（看到类似输出）:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:15555
```

**3.2 运行蒸馏训练**（在终端 2）:

```bash
cd /root/yuxiang/rllm
bash examples/search/train_search_distill.sh
```

**预期看到**:
```
============================================================
蒸馏配置:
  教师模型: Qwen/Qwen3-30B-A3B-Instruct-2507
  教师服务地址: http://localhost:15555/v1
  共享 tokenizer: False
============================================================
```

## 监控训练

### Wandb

访问 https://wandb.ai 查看训练曲线：

- 项目: `rllm-search-agent`
- 实验名称:
  - GRPO: `qwen3-4b-grpo-search-asearcher`
  - 蒸馏: `qwen3-4b-grpo-distill-search-asearcher`

### 日志

训练日志会实时输出到终端。

## 常见问题

### Q1: 数据集未找到

```bash
ValueError: Training dataset 'asearcher' not found.
```

**解决**: 运行步骤 1 准备数据

### Q2: 搜索服务连接失败

```
Search API error: Connection Error
```

**解决**: 
1. 检查搜索服务是否运行
2. 修改启动脚本中的 `retrieval_service_url`

### Q3: 教师模型服务端口被占用

```
Port 15555 is already in use.
```

**解决**: 脚本会自动处理。如果仍有问题，手动杀掉进程：
```bash
kill -9 $(lsof -ti:15555)
```

### Q4: GPU 内存不足

**解决**: 
```bash
# 编辑启动脚本，减小以下参数：
data.train_batch_size=32  # 原来是 64
actor_rollout_ref.rollout.n=4  # 原来是 8
```

### Q5: 训练速度慢

可能原因：
1. 搜索服务响应慢 → 检查搜索服务性能
2. 网络延迟 → 确保搜索服务在同一内网
3. GPU 利用率低 → 增加 batch size 或采样数

## 测试工具（可选）

在正式训练前，可以先测试搜索工具：

```bash
cd /root/yuxiang/rllm
python3 -m examples.search.test_search_tool
```

这会执行一个简单的搜索查询，验证工具是否正常工作。

## 停止训练

### 停止训练进程

按 `Ctrl+C` 或：
```bash
pkill -9 -f 'train_search'
```

### 停止教师模型服务

```bash
kill -9 $(lsof -ti:15555)
```

### 清理 Ray 进程

```bash
pkill -9 -f 'ray::WorkerDict'
```

## 下一步

训练完成后，你可以：

1. 查看 wandb 上的训练曲线
2. 评估模型性能
3. 分析蒸馏效果（对比带/不带蒸馏的结果）
4. 调整超参数进一步优化

## 更多信息

- 详细说明: `README_ASEARCHER.md`
- 实施总结: `IMPLEMENTATION_SUMMARY.md`
- 故障排查: `README_ASEARCHER.md` 中的故障排查部分

## 联系支持

如果遇到问题，请检查：
1. 日志输出
2. wandb 训练曲线
3. GPU 内存使用情况
4. 搜索服务可用性
