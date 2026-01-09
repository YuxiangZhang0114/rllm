# 搜索工具 GRPO + 在线蒸馏实验

本目录包含使用搜索工具进行 GRPO 训练和在线蒸馏的实验代码，从 verl 的实现移植到 rllm 框架。

## 目录结构

```
rllm/examples/search/
├── search_utils.py              # 搜索工具辅助函数（从 verl 移植）
├── search_tool.py               # 搜索工具实现（适配 rllm）
├── prepare_asearcher_data.py    # 数据准备脚本
├── train_search_grpo.py         # GRPO 训练脚本（不带蒸馏）
├── train_search_grpo.sh         # GRPO 启动脚本
├── train_search_distill.py      # GRPO + 蒸馏训练脚本
├── train_search_distill.sh      # GRPO + 蒸馏启动脚本
├── start_teacher_server.sh      # 启动教师模型服务
└── README_ASEARCHER.md          # 本文件
```

## 实验流程

### 1. 准备数据集

首先需要准备 asearcher 训练数据：

```bash
cd rllm

# 确保数据文件存在
# 路径：data/asearcher_train/train.parquet

# 运行数据准备脚本
python3 -m examples.search.prepare_asearcher_data
```

这会将数据注册到 rllm 的 DatasetRegistry 中。

### 2. 确保搜索服务可用

搜索工具需要一个检索服务。默认配置使用：

```
http://10.244.209.173:8000/retrieve
```

如果需要修改搜索服务地址，可以在启动脚本中修改 `retrieval_service_url` 参数。

### 3. 运行 GRPO 训练（验证工具正确性）

在启用蒸馏之前，先运行纯 GRPO 训练来验证搜索工具是否正确工作：

```bash
cd rllm

# 给脚本添加执行权限
chmod +x examples/search/train_search_grpo.sh

# 运行训练
bash examples/search/train_search_grpo.sh
```

关键配置：
- 学生模型：`Qwen/Qwen3-4B-Instruct-2507`
- 算法：GRPO（`algorithm.adv_estimator=grpo`）
- 数据集：asearcher
- 搜索工具：从 verl 移植的 SearchTool

### 4. 启动教师模型服务（用于蒸馏）

如果要进行在线蒸馏实验，需要先启动教师模型服务：

```bash
cd rllm

# 给脚本添加执行权限
chmod +x examples/search/start_teacher_server.sh

# 启动教师模型服务（会占用一个终端）
bash examples/search/start_teacher_server.sh
```

这会启动一个 vLLM 服务器，监听在 `http://localhost:15555/v1`。

**注意：**
- 教师模型服务需要至少 2 个 GPU（配置了 tensor parallel size = 2）
- 如果 GPU 不足，可以修改 `start_teacher_server.sh` 中的 `TP_SIZE` 参数
- 确保教师模型服务启动成功后再开始训练

### 5. 运行 GRPO + 在线蒸馏训练

在教师模型服务启动后，在另一个终端运行蒸馏训练：

```bash
cd rllm

# 给脚本添加执行权限
chmod +x examples/search/train_search_distill.sh

# 运行训练
bash examples/search/train_search_distill.sh
```

关键配置：
- 学生模型：`Qwen/Qwen3-4B-Instruct-2507`
- 教师模型：`Qwen/Qwen3-30B-A3B-Instruct-2507`（通过 API 调用）
- 算法：GRPO（`algorithm.adv_estimator=grpo`）
- 蒸馏：启用（`rllm.distill.enable=True`）
- 教师服务：`http://localhost:15555/v1`

## 配置说明

### 搜索工具配置

在启动脚本中可以配置以下参数：

```bash
retrieval_service_url="http://10.244.209.173:8000/retrieve"  # 检索服务地址
search_topk=5          # 返回前 k 个搜索结果
search_timeout=60      # 搜索请求超时时间（秒）
```

### GRPO 配置

```bash
algorithm.adv_estimator=grpo                    # 使用 GRPO 算法
data.train_batch_size=64                        # 训练批大小
actor_rollout_ref.rollout.n=8                   # 每个 prompt 采样 8 个响应
rllm.agent.max_steps=10                         # Agent 最大步数
```

### 蒸馏配置

```bash
rllm.distill.enable=True                                          # 启用蒸馏
rllm.distill.shared_tokenizer=False                               # 教师和学生使用不同的 tokenizer
rllm.distill.teacher_rollout_args.model="Qwen/Qwen3-30B-A3B-Instruct-2507"
rllm.distill.teacher_rollout_args.base_url="http://localhost:15555/v1"
rllm.distill.teacher_rollout_args.api_key="EMPTY"
```

## 监控训练

训练过程会记录到 wandb：

- 项目名称：`rllm-search-agent`
- 实验名称（GRPO）：`qwen3-4b-grpo-search-asearcher`
- 实验名称（蒸馏）：`qwen3-4b-grpo-distill-search-asearcher`

## 故障排查

### 1. 数据集未找到

```
ValueError: Training dataset 'asearcher' not found.
```

**解决方法**：运行数据准备脚本
```bash
python3 -m examples.search.prepare_asearcher_data
```

### 2. 搜索服务连接失败

```
Search API error: Connection Error
```

**解决方法**：
1. 检查搜索服务是否运行
2. 修改启动脚本中的 `retrieval_service_url` 参数

### 3. 教师模型服务连接失败

```
Could not connect to teacher model at http://localhost:15555/v1
```

**解决方法**：
1. 确保教师模型服务已启动：`bash examples/search/start_teacher_server.sh`
2. 检查端口 15555 是否可用：`lsof -i :15555`
3. 检查教师服务日志

### 4. GPU 内存不足

**解决方法**：
1. 减小 `data.train_batch_size`
2. 减小 `actor_rollout_ref.rollout.n`
3. 降低 `gpu_memory_utilization`
4. 如果是教师模型服务，减小 `TP_SIZE` 或降低 `MAX_MODEL_LEN`

## 与 verl 实现的对比

本实现从 verl 的 GRPO 搜索工具训练移植到 rllm，主要差异：

| 特性 | verl | rllm |
|-----|------|------|
| 搜索工具 | `SearchToolSimple` | `SearchTool`（继承自 `Tool`） |
| 工具配置 | YAML 配置文件 | Python 代码配置 |
| 教师模型服务 | ZMQ 自定义服务 | vLLM OpenAI API 服务器 |
| 训练框架 | verl PPO trainer | rllm AgentTrainer |
| 蒸馏实现 | GKD | 在线蒸馏（通过 API） |

## 参考

- verl baseline: `verl/baseline/GRPO/run_grpo_search_r1_like_qwen_4b.sh`
- verl 搜索工具: `verl/verl/tools/search_tool_simple.py`
- rllm countdown 蒸馏: `rllm/examples/countdown/train_countdown_distill.sh`
