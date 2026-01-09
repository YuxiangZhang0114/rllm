# 搜索任务长度配置说明

## 为什么需要大的长度限制？

搜索任务不同于普通的对话任务，它涉及**多轮工具调用**，每轮包括：

1. **模型生成**: 生成搜索查询（几十个 token）
2. **工具返回**: 返回搜索结果（可能有几千个 token！）
3. **模型推理**: 基于搜索结果继续思考和生成下一步
4. **重复**: 可能需要多次搜索才能找到答案

因此，单个响应的长度可能远超常规对话任务。

## 配置参数详解

### 1. 数据处理相关

```bash
data.max_prompt_length=2048        # 用户问题的最大长度
data.max_response_length=20480     # 模型响应的最大长度（包含所有工具调用）
data.filter_overlong_prompts=True  # 过滤超长 prompt
data.truncation='error'            # 遇到截断时报错（而非静默截断）
data.return_raw_chat=True          # 返回原始聊天格式
```

**关键点**:
- `max_response_length=20480` (20K) 足够容纳多轮搜索结果
- 如果设置为 2048，很可能在第一轮搜索后就超长了

### 2. 模型推理相关

```bash
actor_rollout_ref.rollout.max_model_len=32768  # VLLM 支持的最大上下文长度
```

**说明**:
- 这是 VLLM 模型的**总上下文窗口**
- 必须满足: `max_prompt_length + max_response_length ≤ max_model_len`
- Qwen3-4B-Instruct-2507 支持 32K 上下文
- 我们配置: 2048 + 20480 = 22528 < 32768 ✅

### 3. 训练相关

```bash
# PPO 训练批次配置
actor_rollout_ref.actor.ppo_mini_batch_size=64          # 小批次大小
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4  # 每个 GPU 的微批次
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=23000 # 每个 GPU 的最大 token 数
actor_rollout_ref.actor.use_dynamic_bsz=True            # 动态批次大小

# Log probability 计算配置
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4      # 微批次
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True            # 动态批次
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768    # 最大 token 数

# Reference model 配置（类似）
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768
```

**说明**:
- `ppo_max_token_len_per_gpu=23000`: 限制每个 GPU 处理的 token 数，避免 OOM
- `log_prob_max_token_len_per_gpu=32768`: log_prob 计算时的最大 token 数
- `use_dynamic_bsz=True`: 根据实际序列长度动态调整批次大小

## 实际示例

### 一个典型的搜索轨迹

```
用户问题 (~50 tokens):
"What is the capital of France and when was it founded?"

第 1 轮:
- 模型生成 (~30 tokens): search(query="capital of France")
- 工具返回 (~1000 tokens): [包含 5 篇关于巴黎的文档]
- 模型思考 (~100 tokens): "Paris is the capital. Now I need founding date..."

第 2 轮:
- 模型生成 (~30 tokens): search(query="Paris founding date")
- 工具返回 (~1000 tokens): [包含 5 篇关于巴黎历史的文档]
- 模型思考 (~50 tokens): "Founded around 3rd century BC..."

第 3 轮:
- 模型生成 (~150 tokens): "Paris is the capital of France. It was founded..."

总计: ~2410 tokens
```

如果 `max_response_length=2048`，这个轨迹会被截断！

### 更复杂的场景

对于需要 5-10 轮搜索的复杂问题，总长度可能达到 10K-15K tokens。

## 性能考虑

### GPU 内存

长序列需要更多 GPU 内存：

| 配置 | 内存消耗 | 建议 GPU |
|------|---------|---------|
| max_response=2048 | ~低 | 1x A100 40GB |
| max_response=20480 | ~高 | 4x A100 40GB |

**优化策略**:
1. 启用 gradient checkpointing
2. 启用 FSDP offload
3. 使用 dynamic batch size
4. 降低 `ppo_max_token_len_per_gpu`

### 训练速度

长序列训练更慢：
- 2K response: ~1000 samples/hour
- 20K response: ~200 samples/hour

**权衡**:
- 如果任务不需要很长的轨迹，可以降低 `max_response_length`
- 监控实际轨迹长度，调整配置

## 故障排查

### 1. OOM (Out of Memory)

```
RuntimeError: CUDA out of memory
```

**解决**:
```bash
# 降低批次大小
data.train_batch_size=64          # 原来 128
actor_rollout_ref.rollout.n=3     # 原来 5

# 降低 token 限制
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000  # 原来 23000
```

### 2. 序列被截断

```
Warning: Sequence truncated at max_response_length
```

**解决**:
```bash
# 增加 response 长度
data.max_response_length=30720    # 从 20480 增加到 30K

# 确保 max_model_len 足够
actor_rollout_ref.rollout.max_model_len=32768
```

### 3. 训练太慢

**解决**:
```bash
# 如果任务不需要这么长，可以降低
data.max_response_length=10240    # 从 20480 降低到 10K

# 增加采样数而非序列长度
actor_rollout_ref.rollout.n=8     # 从 5 增加到 8
```

## 推荐配置

### 标准搜索任务（HotpotQA, asearcher）

```bash
data.max_prompt_length=2048
data.max_response_length=20480
actor_rollout_ref.rollout.max_model_len=32768
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=23000
```

### 简单搜索任务（单轮搜索）

```bash
data.max_prompt_length=2048
data.max_response_length=4096
actor_rollout_ref.rollout.max_model_len=8192
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8000
```

### 复杂多步推理（需要很多轮）

```bash
data.max_prompt_length=2048
data.max_response_length=30720
actor_rollout_ref.rollout.max_model_len=32768
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=23000
```

## 参考

- verl baseline: `verl/baseline/GRPO/run_grpo_search_r1_like_qwen_4b.sh`
- VLLM 文档: https://docs.vllm.ai/en/latest/models/engine_args.html
- Qwen3 模型卡: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
