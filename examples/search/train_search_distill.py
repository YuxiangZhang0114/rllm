"""使用 GRPO + 在线蒸馏训练搜索 agent"""

import hydra
from omegaconf import OmegaConf

from rllm.agents.system_prompts import SEARCH_SYSTEM_PROMPT
from rllm.agents.tool_agent import ToolAgent
from rllm.data import DatasetRegistry
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import search_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer

from examples.search.search_tool import SearchTool


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    """
    使用 GRPO + 在线蒸馏训练搜索 agent。
    
    配置要求：
    - algorithm.adv_estimator=grpo  # 使用 GRPO
    - rllm.distill.enable=True  # 启用蒸馏
    - rllm.distill.teacher_rollout_args.model="Qwen/Qwen3-30B-A3B-Instruct-2507"
    - rllm.distill.teacher_rollout_args.base_url="http://localhost:15555/v1"
    """
    # 加载数据集
    train_dataset = DatasetRegistry.load_dataset("asearcher", "train")
    val_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")
    
    if train_dataset is None:
        raise ValueError(
            "Training dataset 'asearcher' not found. "
            "Please run prepare_asearcher_data.py first to register the dataset."
        )
    
    if val_dataset is None:
        print("Warning: Validation dataset 'hotpotqa/test' not found. Using training set for validation.")
        val_dataset = train_dataset
    
    # 配置搜索工具
    retrieval_service_url = OmegaConf.select(config, "retrieval_service_url", default="http://10.244.209.173:8000/retrieve")
    topk = OmegaConf.select(config, "search_topk", default=5)
    timeout = OmegaConf.select(config, "search_timeout", default=60)
    
    tool_map = {
        "search": SearchTool(
            retrieval_service_url=retrieval_service_url,
            topk=topk,
            timeout=timeout,
        )
    }
    
    # 环境配置
    max_steps = OmegaConf.select(config, "rllm.agent.max_steps", default=20)
    env_args = {
        "max_steps": max_steps,
        "tool_map": tool_map,
        "reward_fn": search_reward_fn,
    }
    
    # Agent 配置
    parser_name = OmegaConf.select(config, "parser_name", default="qwen")
    agent_args = {
        "system_prompt": SEARCH_SYSTEM_PROMPT,
        "tool_map": tool_map,
        "parser_name": parser_name,
    }
    
    # 打印蒸馏配置信息
    distill_enable = OmegaConf.select(config, "rllm.distill.enable", default=False)
    if distill_enable:
        teacher_model = OmegaConf.select(config, "rllm.distill.teacher_rollout_args.model", default="Not specified")
        teacher_base_url = OmegaConf.select(config, "rllm.distill.teacher_rollout_args.base_url", default="Not specified")
        shared_tokenizer = OmegaConf.select(config, "rllm.distill.shared_tokenizer", default=False)
        
        print("\n" + "=" * 60)
        print("蒸馏配置:")
        print(f"  教师模型: {teacher_model}")
        print(f"  教师服务地址: {teacher_base_url}")
        print(f"  共享 tokenizer: {shared_tokenizer}")
        print("=" * 60 + "\n")
    else:
        print("\n警告: rllm.distill.enable=False 或未配置，将使用纯 GRPO 训练\n")
    
    # 创建训练器
    trainer = AgentTrainer(
        agent_class=ToolAgent,
        env_class=ToolEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_args=agent_args,
        env_args=env_args,
    )
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
