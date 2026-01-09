"""搜索工具 - 适配 rllm 框架"""

import json
import logging
from typing import Any

from rllm.tools.tool_base import Tool, ToolOutput

from examples.search.search_utils import perform_single_search_batch

logger = logging.getLogger(__name__)


class SearchTool(Tool):
    """
    搜索工具，使用远程检索服务进行信息检索。
    
    从 verl 的 SearchToolSimple 移植到 rllm 框架。
    """
    
    NAME = "search"
    DESCRIPTION = "Searches for relevant information based on the given query using a retrieval service"
    
    def __init__(
        self,
        name: str = NAME,
        description: str = DESCRIPTION,
        retrieval_service_url: str = "http://10.244.209.173:8000/retrieve",
        topk: int = 5,
        timeout: int = 60,
    ):
        """
        初始化搜索工具。
        
        Args:
            name: 工具名称
            description: 工具描述
            retrieval_service_url: 检索服务 URL
            topk: 返回前 k 个结果（默认 5）
            timeout: 请求超时时间（秒，默认 60）
        """
        self.retrieval_service_url = retrieval_service_url
        self.topk = topk
        self.timeout = timeout
        
        super().__init__(name=name, description=description)
        
        logger.info(
            f"Initialized SearchTool with url={self.retrieval_service_url}, "
            f"topk={self.topk}, timeout={self.timeout}"
        )
    
    @property
    def json(self) -> dict[str, Any]:
        """返回工具的 JSON schema（OpenAI function calling 格式）"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A fully-formed semantic query string. The tool will return search results for this query.",
                        },
                        "topk": {
                            "type": "integer",
                            "description": f"Number of top results to return. Default is {self.topk}.",
                            "default": self.topk,
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    
    def forward(self, query: str = None, query_list: list[str] = None, topk: int = None, **kwargs) -> ToolOutput:
        """
        执行搜索操作。
        
        Args:
            query: 单个搜索查询字符串
            query_list: 查询列表（用于批量搜索）
            topk: 返回前 k 个结果（可选，默认使用构造时的值）
            **kwargs: 其他参数
        
        Returns:
            ToolOutput: 包含搜索结果或错误信息
        """
        # 处理查询参数
        if query is not None:
            if isinstance(query, str):
                query_list = [query]
            else:
                error_msg = "Error: 'query' must be a string."
                logger.error(f"[SearchTool] {error_msg} Received query: {query}")
                return ToolOutput(name=self.name, error=error_msg)
        elif query_list is not None:
            if not isinstance(query_list, list):
                error_msg = "Error: 'query_list' must be a list."
                logger.error(f"[SearchTool] {error_msg} Received query_list: {query_list}")
                return ToolOutput(name=self.name, error=error_msg)
        else:
            error_msg = "Error: Either 'query' or 'query_list' must be provided."
            logger.error(f"[SearchTool] {error_msg}")
            return ToolOutput(name=self.name, error=error_msg)
        
        # 获取 topk 参数
        if topk is None:
            topk = self.topk
        if not isinstance(topk, int) or topk <= 0:
            topk = self.topk
        
        # 执行搜索
        try:
            result_text, metadata = perform_single_search_batch(
                retrieval_service_url=self.retrieval_service_url,
                query_list=query_list,
                topk=topk,
                timeout=self.timeout,
            )
            
            # 检查搜索状态
            if metadata.get("status") == "api_error":
                return ToolOutput(
                    name=self.name,
                    error=f"Search API error: {metadata.get('api_request_error', 'Unknown error')}",
                    metadata=metadata,
                )
            
            # 返回搜索结果
            return ToolOutput(
                name=self.name,
                output=result_text,
                metadata=metadata,
            )
            
        except Exception as e:
            error_msg = f"Search execution failed: {e}"
            logger.error(f"[SearchTool] {error_msg}")
            return ToolOutput(name=self.name, error=error_msg)


def create_search_tool(
    retrieval_service_url: str = "http://10.244.209.173:8000/retrieve",
    topk: int = 5,
    timeout: int = 60,
) -> SearchTool:
    """
    创建 SearchTool 实例的便捷函数。
    
    Args:
        retrieval_service_url: 检索服务 URL
        topk: 返回前 k 个结果
        timeout: 请求超时时间（秒）
    
    Returns:
        SearchTool 实例
    """
    return SearchTool(
        retrieval_service_url=retrieval_service_url,
        topk=topk,
        timeout=timeout,
    )
