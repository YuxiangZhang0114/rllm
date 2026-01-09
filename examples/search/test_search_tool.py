"""测试搜索工具是否正确工作"""

import logging

logging.basicConfig(level=logging.INFO)

from examples.search.search_tool import SearchTool


def test_search_tool():
    """测试搜索工具"""
    print("=" * 60)
    print("测试搜索工具")
    print("=" * 60)
    
    # 创建搜索工具实例
    search_tool = SearchTool(
        retrieval_service_url="http://10.244.209.173:8000/retrieve",
        topk=3,
        timeout=60,
    )
    
    print(f"\n工具名称: {search_tool.name}")
    print(f"工具描述: {search_tool.description}")
    print(f"\n工具 Schema:")
    import json
    print(json.dumps(search_tool.json, indent=2, ensure_ascii=False))
    
    # 测试单个查询
    print("\n" + "=" * 60)
    print("测试单个查询")
    print("=" * 60)
    
    query = "What is the capital of France?"
    print(f"\n查询: {query}")
    
    result = search_tool.forward(query=query, topk=3)
    
    print(f"\n状态: {'成功' if result.error is None else '失败'}")
    if result.error:
        print(f"错误: {result.error}")
    else:
        print(f"输出 (前 500 字符):\n{str(result.output)[:500]}")
        if result.metadata:
            print(f"\n元数据:")
            print(f"  - 状态: {result.metadata.get('status')}")
            print(f"  - 查询数: {result.metadata.get('query_count')}")
            print(f"  - 结果数: {result.metadata.get('total_results')}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_search_tool()
