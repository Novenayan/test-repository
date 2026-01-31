#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI智能助手 - 集成RRF算法的混合搜索系统
RRF算法融合ES内部的BM25和向量搜索结果，tavily由qwen-agent根据情况判断使用
"""
import os
import json
from typing import List, Dict, Any, Optional

from qwen_agent.agents import Assistant
from qwen_agent.llm.oai import TextChatAtOAI
import re
from elasticsearch import Elasticsearch
from openai import OpenAI
import tavily

# 配置常量
ES_URL = 'https://127.0.0.1:9200'
ES_USERNAME = 'elastic'
ES_PASSWORD = 'u6jbCAwjf2oEjTOAXAJx'

# Tavily 配置
TAVILY_API_KEY = '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'

def connect_to_es():
    """连接到 Elasticsearch"""
    try:
        es = Elasticsearch(
            [ES_URL],
            basic_auth=(ES_USERNAME, ES_PASSWORD),
            verify_certs=False,
            request_timeout=30
        )
        
        if es.ping():
            print("成功连接到 Elasticsearch")
            return es
        else:
            print("连接 Elasticsearch 失败")
            return None
    except Exception as e:
        print(f"连接 Elasticsearch 时出错: {e}")
        return None

def get_docs_files():
    """获取docs目录下的所有文件"""
    file_dir = './docs'
    if not os.path.exists(file_dir):
        return []
    return [os.path.join(file_dir, f) for f in os.listdir(file_dir) 
            if os.path.isfile(os.path.join(file_dir, f))]

def hybrid_search(es, query, index_name='insurance_docs_chunks', size=5):
    """改进的混合搜索（关键词搜索 + 向量搜索）使用RRF算法融合"""
    try:
        # 初始化 OpenAI 客户端用于查询 embedding
        api_key = os.getenv("AGI_API_KEY_GEN")
        if not api_key:
            print("警告: 未设置 AGI_API_KEY_GEN 环境变量，无法使用向量搜索功能")
            # 如果没有API密钥，回退到纯关键词搜索
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "content"],  # 标题权重更高
                        "type": "best_fields"
                    }
                },
                "size": size,
                "_source": ["title", "content", "file_path", "file_name", "page_num"]  # 确保返回所有需要的字段
            }
            
            response = es.search(index=index_name, body=search_body)
            return response

        # 检查API连接性
        try:
            embedding_client = OpenAI(
                api_key=api_key,
                base_url="https://api.siliconflow.cn/v1"
            )
        except Exception as e:
            print(f"警告: 无法初始化embedding客户端: {e}")
            print("回退到纯关键词搜索")
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "content"],  # 标题权重更高
                        "type": "best_fields"
                    }
                },
                "size": size,
                "_source": ["title", "content", "file_path", "file_name", "page_num"]  # 确保返回所有需要的字段
            }
            
            response = es.search(index=index_name, body=search_body)
            return response

        # 获取查询的 embedding
        max_length = 32768  # Qwen/Qwen3-Embedding-0.6B 的最大长度
        if len(query) > max_length:
            query = query[:max_length]
            print(f"查询过长，已截断至 {max_length} 字符")

        try:
            response = embedding_client.embeddings.create(
                model="Qwen/Qwen3-Embedding-0.6B",
                input=query
            )
            query_embedding = response.data[0].embedding
            print(f"查询 embedding 生成成功，维度: {len(query_embedding)}")
        except Exception as e:
            print(f"获取查询 embedding 时出错: {e}")
            print("使用纯关键词搜索")
            # 如果无法生成 embedding，回退到纯关键词搜索
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "content"],  # 标题权重更高
                        "type": "best_fields"
                    }
                },
                "size": size,
                "_source": ["title", "content", "file_path", "file_name", "page_num"]  # 确保返回所有需要的字段
            }
            
            response = es.search(index=index_name, body=search_body)
            return response

        # 执行关键词搜索
        keyword_search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "content"],
                    "type": "best_fields",
                }
            },
            "size": size * 2,  # 获取更多结果用于RRF融合
            "_source": ["title", "content", "file_path", "file_name", "page_num"],
        }
        
        keyword_response = es.search(index=index_name, body=keyword_search_body)
        
        # 执行向量搜索（kNN）
        vector_search_body = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                        "params": {
                            "query_vector": query_embedding
                        }
                    }
                }
            },
            "size": size * 2,  # 获取更多结果用于RRF融合
            "_source": ["title", "content", "file_path", "file_name", "page_num"],
        }
        
        vector_response = es.search(index=index_name, body=vector_search_body)
        
        # 使用RRF算法融合两种搜索结果
        rrf_fused_results = rrf_fusion(keyword_response, vector_response, k=60, top_k=size)
        
        return rrf_fused_results
        
    except Exception as e:
        print(f"混合搜索时出错: {e}")
        # 如果混合搜索完全失败，尝试纯关键词搜索
        try:
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "content"],  # 标题权重更高
                        "type": "best_fields"
                    }
                },
                "_source": ["title", "content", "file_path", "file_name", "page_num"],  # 确保返回所有需要的字段
                "size": size
            }
            
            response = es.search(index=index_name, body=search_body)
            return response
        except Exception as e2:
            print(f"纯关键词搜索也失败: {e2}")
            return None

def rrf_fusion(keyword_response, vector_response, k=60, top_k=5):
    """
    使用RRF (Reciprocal Rank Fusion) 算法融合关键词搜索和向量搜索结果
    算法参考：对多个搜索引擎的结果进行融合，使用公式 1/(rank + k) 计算分数
    
    Args:
        keyword_response: 关键词搜索的ES响应
        vector_response: 向量搜索的ES响应
        k: RRF公式中的平滑参数，默认60
        top_k: 返回的top结果数量
    
    Returns:
        融合后的ES响应，格式与单个搜索响应一致
    """
    try:
        keyword_hits = keyword_response['hits']['hits']
        vector_hits = vector_response['hits']['hits']
        
        # 创建文档ID到原始hit的映射
        all_hits_map = {}
        for hit in keyword_hits + vector_hits:
            doc_id = hit['_id']
            if doc_id not in all_hits_map:
                all_hits_map[doc_id] = hit
        
        # 计算RRF分数 - 算法核心逻辑实现
        rrf_scores = {}
        
        # 为关键词搜索结果分配分数 (使用公式 1/(rank + k))
        for rank, hit in enumerate(keyword_hits):
            doc_id = hit['_id']
            # RRF公式: 1 / (rank + k)，rank从1开始
            rrf_score = 1.0 / (rank + 1 + k)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score
        
        # 为向量搜索结果分配分数 (使用公式 1/(rank + k))
        for rank, hit in enumerate(vector_hits):
            doc_id = hit['_id']
            # RRF公式: 1 / (rank + k)，rank从1开始
            rrf_score = 1.0 / (rank + 1 + k)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score
        
        # 按RRF分数排序 (按融合后的分数重新排序)
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # 重新构建hits列表，按照RRF排序
        fused_hits = []
        for doc_id in sorted_doc_ids[:top_k]:
            hit = all_hits_map[doc_id].copy()
            # 使用RRF分数作为_score
            hit['_score'] = rrf_scores[doc_id]
            fused_hits.append(hit)
        
        # 构建融合后的响应
        fused_response = {
            'hits': {
                'hits': fused_hits,
                'total': {'value': len(fused_hits), 'relation': 'eq'}
            }
        }
        
        print(f"RRF融合完成，共融合了 {len(keyword_hits)} 个关键词搜索结果和 {len(vector_hits)} 个向量搜索结果")
        return fused_response
        
    except Exception as e:
        print(f"RRF融合过程中出错: {e}")
        import traceback
        traceback.print_exc()
        # 出错时回退到关键词搜索结果
        return keyword_response

def retrieve_from_es(es, query):
    """从ES检索相关文档（使用改进的混合搜索）"""
    if not es:
        return []
    
    results = hybrid_search(es, query)
    if not results:
        return []
    
    hits = results['hits']['hits']
    retrieved_docs = []
    
    for hit in hits:
        source = hit['_source']
        clean_content = source.get('content', '')
        
        retrieved_docs.append({
            'title': source.get('title', ''),
            'content': clean_content,
            'file_path': source.get('file_path', ''),
            'file_name': source.get('file_name', ''),
            'page_num': source.get('page_num', 1),  # 默认页码为1
            'score': hit['_score']
        })
    
    return retrieved_docs

def init_agent_service(es=None):
    """初始化智能体服务"""
    files = get_docs_files()
    print('files=', files)

    # 配置 LLM - 优先使用环境变量AGI_API_KEY_GEN，如果不存在则使用默认值
    api_key = os.getenv('AGI_API_KEY_GEN', 'sk-ykjwinvuaqujgipblcfnbvbacauqxpgxpqzkjzuezwolpcxr')
    
    if api_key == 'sk-ykjwinvuaqujgipblcfnbvbacauqxpgxpqzkjzuezwolpcxr':
        print("警告: 使用了默认API密钥，建议设置环境变量以获得更好的安全性")
    else:
        print("已使用环境变量中的API密钥")
    
    llm_cfg = {
        'model': 'Qwen/Qwen3-Omni-30B-A3B-Instruct',  # 使用硅基流动的Qwen3模型
        'api_key': api_key,
        'base_url': 'https://api.siliconflow.cn/v1',  # 使用硅基流动API端点
    }
    
    # 系统指令
    system_instruction = '''你是涅槃搜索 - 专业的智能搜索助手。
你总是用中文回复用户。

你的回答应当专业、准确、有帮助。

重要指令：
1. 当用户提出问题时，优先查看是否已提供相关文档内容
2. 对于保险相关问题，优先基于已提供的保险文档进行回答
3. 当本地文档无法充分回答问题时，可以使用网络搜索获取补充信息
4. 智能判断何时使用本地文档、何时使用网络搜索：对于具体的保险条款、产品细节，优先使用本地文档；对于需要时效性信息、行业新闻或无法在本地文档中找到的问题，可使用网络搜索
5. 回答问题时尽量引用文档来源（文件名、页码等信息），若使用网络信息则说明信息来源
'''
    
    # 如果ES连接成功，修改system_instruction以包含检索信息
    if es:
        system_instruction += '''
        系统已经为你检索了相关文档，文档内容已添加到上下文中，请根据需要使用这些信息回答用户的问题。
        '''
    
    # 尝试初始化LLM，如果失败则给出错误信息
    try:
        # 设置较少的重试次数以避免长时间等待
        llm_cfg_with_retry = llm_cfg.copy()
        llm_cfg_with_retry['generate_cfg'] = llm_cfg.get('generate_cfg', {})
        llm_cfg_with_retry['generate_cfg']['max_retries'] = 0  # 设置为0次重试，避免长时间等待
        
        llm = TextChatAtOAI(llm_cfg_with_retry)
        print("LLM客户端初始化成功")
    except Exception as e:
        print(f"警告: LLM客户端初始化失败: {e}")
        print("请检查网络连接和API密钥配置")
        # 创建一个最小配置的LLM实例，不进行重试以避免长时间等待
        llm_cfg_fallback = {
            'model': 'Qwen/Qwen3-Omni-30B-A3B-Instruct',
            'api_key': api_key,
            'base_url': 'https://api.siliconflow.cn/v1',
            'generate_cfg': {'max_retries': 0}  # 不重试
        }
        try:
            llm = TextChatAtOAI(llm_cfg_fallback)
            print("使用最小配置初始化LLM客户端成功")
        except Exception as e2:
            print(f"LLM客户端初始化完全失败: {e2}")
            # 如果API配置失败，创建一个带错误信息的mock LLM
            class MockLLM:
                def __init__(self, config):
                    self.config = config
                def run(self, messages, **kwargs):
                    yield [{
                        'role': 'assistant',
                        'content': f"API连接失败: 无法连接到模型服务。请检查API密钥配置。"
                    }]
            llm = MockLLM(llm_cfg_fallback)
            print("已创建模拟LLM客户端以避免程序崩溃")
    
    # 导入Tavily工具以确保其被注册
    import search_tools
    # 恢复Tavily工具，让系统能够智能判断何时使用ES数据库、何时使用网络搜索
    kwargs = {'llm': llm, 'system_message': system_instruction, 'function_list': ['tavily_search', 'code_interpreter']}
    if files:
        kwargs['files'] = files
    
    # 创建助手实例并设置名称
    assistant = Assistant(**kwargs)
    assistant.name = '涅槃搜索'  # 确保助手名称是'涅槃搜索'
    
    return assistant, es

def app_gui():
    """图形界面模式"""
    try:
        print("正在启动 Web 界面...")
        
        # 连接到ES
        es = connect_to_es()
        
        # 初始化助手
        bot, es = init_agent_service(es)
        
        # 配置聊天界面，列举典型保险查询问题
        chatbot_config = {
            'prompt.suggestions': [
                '介绍下雇主责任险',
                '工伤保险和雇主险有什么区别？',
                '如何申请保险',
                '帮我分析一下保险文档',
                '雇主责任险最近有什么新闻',
                '保险和AI结合的最新成就',
                '最近保险业的五大新闻是什么'
            ],
            'agent.name': '涅槃搜索',
            'agent.description': '涅槃搜索 - 专业的智能搜索助手',
        }
        print("Web 界面准备就绪，正在启动服务...")
        
        # 使用自定义的WebUI类以实现正确的布局和ES搜索结果显示
        from custom_webui import CustomWebUI
        
        # 使用自定义的WebUI，传入ES客户端以便处理ES搜索
        CustomWebUI(
            bot,
            es_client=es,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和硅基流动API Key配置")

if __name__ == '__main__':
    print("正在启动 WebUI 模式...")
    app_gui()
