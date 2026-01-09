"""使用 GRPO 训练搜索 agent（不带蒸馏）"""

import hydra

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
    使用 GRPO 训练搜索 agent。
    
    这个脚本用于验证搜索工具的正确性，不包含蒸馏功能。
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
    # 从配置中读取检索服务地址，如果没有则使用默认值
    retrieval_service_url = config.get("retrieval_service_url", "http://10.244.209.173:8000/retrieve")
    topk = config.get("search_topk", 5)
    timeout = config.get("search_timeout", 60)
    
    tool_map = {
        "search": SearchTool(
            retrieval_service_url=retrieval_service_url,
            topk=topk,
            timeout=timeout,
        )
    }
    
    # 环境配置
    max_steps = config.get("rllm", {}).get("agent", {}).get("max_steps", 20)
    env_args = {
        "max_steps": max_steps,
        "tool_map": tool_map,
        "reward_fn": search_reward_fn,
    }
    
    # Agent 配置
    parser_name = config.get("parser_name", "qwen")
    agent_args = {
        "system_prompt": SEARCH_SYSTEM_PROMPT,
        "tool_map": tool_map,
        "parser_name": parser_name,
    }
    
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
