import os
import re
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.llm.oai import TextChatAtOAI
from qwen_agent.tools.base import BaseTool, register_tool
import json
from elasticsearch import Elasticsearch
from openai import OpenAI
import tavily

# Elasticsearch 配置
ES_URL = 'https://127.0.0.1:9200'
ES_USERNAME = 'elastic'
ES_PASSWORD = 'u6jbCAwjf2oEjTOAXAJx'

# Tavily 配置
TAVILY_API_KEY = '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'
tavily_client = tavily.TavilyClient(api_key=TAVILY_API_KEY)

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
    """并行混合搜索（关键词搜索 + 向量搜索）"""
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
                "highlight": {
                    "fields": {
                        "content": {
                            "fragment_size": 150,
                            "number_of_fragments": 3
                        }
                    }
                },
                "size": size,
                "_source": ["title", "content", "file_path", "file_name", "page_num"]  # 确保返回所有需要的字段
            }
            
            response = es.search(index=index_name, body=search_body)
            return response

        # 检查API连接性
        try:
            from openai import OpenAI
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
                "highlight": {
                    "fields": {
                        "content": {
                            "fragment_size": 150,
                            "number_of_fragments": 3
                        }
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
                "highlight": {
                    "fields": {
                        "content": {
                            "fragment_size": 150,
                            "number_of_fragments": 3
                        }
                    }
                },
                "size": size,
                "_source": ["title", "content", "file_path", "file_name", "page_num"]  # 确保返回所有需要的字段
            }
            
            response = es.search(index=index_name, body=search_body)
            return response

        # 并行混合搜索：使用 bool 查询组合关键词搜索和向量搜索
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        # 关键词匹配
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^2", "content"],
                                "type": "best_fields",
                                "boost": 0.5  # 关键词搜索权重
                            }
                        },
                        # 向量搜索（kNN）
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                                    "params": {
                                        "query_vector": query_embedding
                                    }
                                }
                            }
                        }
                    ]
                }
            },
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 150,
                        "number_of_fragments": 3
                    }
                }
            },
            "size": size,
            "_source": ["title", "content", "file_path", "file_name", "page_num"]  # 确保返回所有需要的字段
        }
        
        response = es.search(index=index_name, body=search_body)
        return response
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
                "highlight": {
                    "fields": {
                        "content": {
                            "fragment_size": 150,
                            "number_of_fragments": 3
                        }
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
def retrieve_from_es(es, query):
    """从ES检索相关文档（使用混合搜索）"""
    if not es:
        return []
    
    results = hybrid_search(es, query)
    if not results:
        return []
    
    hits = results['hits']['hits']
    retrieved_docs = []
    
    for hit in hits:
        source = hit['_source']
        # 移除高亮标签
        if 'highlight' in hit:
            highlights = hit['highlight'].get('content', [])
            if highlights:
                clean_content = re.sub(r'<[^>]+>', '', highlights[0])
            else:
                clean_content = source.get('content', '')
        else:
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

def get_docs_files():
    """获取docs目录下的所有文件"""
    file_dir = './docs'
    if not os.path.exists(file_dir):
        return []
    return [os.path.join(file_dir, f) for f in os.listdir(file_dir) 
            if os.path.isfile(os.path.join(file_dir, f))]

def init_agent_service(es=None):
    """初始化智能体服务"""
    files = get_docs_files()
    print('files=', files)

    # 配置 LLM - 优先使用环境变量AGI_API_KEY_GEN，如果不存在则使用默认值
    api_key = os.getenv('AGI_API_KEY_GEN')
    if not api_key:
        print("警告: 未设置 AGI_API_KEY_GEN 环境变量")
        print("提示: 建议设置环境变量以获得更好的安全性")
        print("设置方法: set AGI_API_KEY_GEN=your_api_key_here")
        api_key = "sk-ykjwinvuaqujgipblcfnbvbacauqxpgxpqzkjzuezwolpcxr"  # 测试用密钥
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
1. 当用户提出问题时，你必须首先查看是否已提供相关文档内容
2. 对于保险相关问题，优先基于已提供的保险文档进行回答
3. 只有在已提供文档无法回答问题时，才考虑其他方式
4. 绝不允许在有相关文档的情况下忽略文档而去网络搜索
5. 回答问题时务必引用文档来源（文件名、页码等信息）
'''
    
    # 如果ES连接成功，修改system_instruction以包含检索信息
    if es:
        system_instruction += '''
        系统已经为你检索了相关文档，请基于这些文档内容回答用户的问题。
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
    
    # 我们已经在custom_webui.py中实现了ES检索功能，不需要Tavily工具
    # 只保留code_interpreter工具
    kwargs = {'llm': llm, 'system_message': system_instruction, 'function_list': ['code_interpreter']}
    if files:
        kwargs['files'] = files
    
    # 创建助手实例并设置名称
    assistant = Assistant(**kwargs)
    assistant.name = '涅槃搜索'  # 确保助手名称是'涅槃搜索'
    
    return assistant, es

def app_tui():
    """终端交互模式"""
    try:
        # 连接到ES
        es = connect_to_es()
        
        # 初始化助手
        bot, es = init_agent_service(es)
        messages = []
        
        while True:
            try:
                query = input('user question: ')
                file_input = input('file url (press enter if no file): ').strip()
                
                if not query:
                    print('user question cannot be empty！')
                    continue
                
                # 如果ES连接成功，先检索相关文档
                retrieved_docs = []
                if es:
                    print(f"正在检索与 '{query}' 相关的文档...")
                    retrieved_docs = retrieve_from_es(es, query)
                    if retrieved_docs:
                        print(f"找到 {len(retrieved_docs)} 个相关文档:")
                        for i, doc in enumerate(retrieved_docs):
                            print(f"  {i+1}. {doc['title']} (页码: {doc['page_num']}, 相关性: {doc['score']:.2f})")
                            print(f"     内容预览: {doc['content'][:100]}...")
                    else:
                        print("未找到相关文档")
                
                # 构建消息
                if not file_input:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file_input}]})

                print("正在处理您的请求...")
                
                # 如果有检索到的文档，添加到系统消息中
                if retrieved_docs:
                    context = "以下是检索到的相关文档内容：\n\n"
                    for doc in retrieved_docs:
                        context += f"文档: {doc['title']}, 页码: {doc['page_num']}\n内容: {doc['content']}\n\n"
                    
                    # 在消息中加入检索到的上下文
                    if messages and messages[0]['role'] == 'system':
                        messages[0]['content'] += f"\n\n{context}"
                    else:
                        messages = [{'role': 'system', 'content': context}] + messages
                
                for response in bot.run(messages):
                    print('bot response:', response)
                messages.extend(response)
            except KeyboardInterrupt:
                print("\n程序已退出")
                break
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")

def app_gui():
    """图形界面模式"""
    try:
        print("正在启动 Web 界面...")
        
        # 连接到ES
        es = connect_to_es()
        
        # 初始化助手
        bot, es = init_agent_service(es)
        
        # 配置聊天界面，列举3个典型保险查询问题
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
            'agent.avatar': None  # 确保不使用默认头像
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

import time
import threading

def get_user_choice_with_timeout():
    """
    获取用户选择，如果2秒内没有输入则使用默认选择
    使用更简单的方式避免线程问题
    """
    import sys
    import select
    
    print("选择运行模式 (2秒后将自动选择默认选项):")
    print("1: GUI模式 (Web界面)")
    print("2: TUI模式 (终端交互)")
    print("请输入选择 (1 或 2，默认为 1): ", end='', flush=True)
    
    # 在Windows上使用msvcrt实现非阻塞输入
    try:
        import msvcrt
        import time
        
        start_time = time.time()
        user_input = ""
        
        while time.time() - start_time < 2:
            if msvcrt.kbhit():
                char = msvcrt.getch().decode('utf-8')
                if char in ['1', '2', '\r', '\n']:
                    if char in ['1', '2']:
                        user_input += char
                        print(char, end='', flush=True)
                    elif char in ['\r', '\n'] and user_input:
                        print()  # 换行
                        break
                elif char == '\b':  # 退格键
                    if user_input:
                        user_input = user_input[:-1]
                        print('\b \b', end='', flush=True)
            time.sleep(0.01)  # 短暂休眠，避免CPU占用过高
        
        if user_input:
            if user_input in ['1', '2']:
                return user_input
            else:
                print(f"\n无效输入 '{user_input}'，使用默认选项 (GUI模式)")
                return "1"
        else:
            print("\n2秒内未收到选择，使用默认选项 (GUI模式)")
            return "1"
    except ImportError:
        # 如果不是Windows系统或msvcrt不可用，使用简单方式
        print("\n系统不支持非阻塞输入，使用默认选项 (GUI模式)")
        return "1"

if __name__ == '__main__':
    try:
        choice = get_user_choice_with_timeout()
        if choice == "2":
            app_tui()
        else:
            app_gui()
    except:
        app_gui()  # 默认启动GUI模式
