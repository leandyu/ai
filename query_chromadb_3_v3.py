import re
import json
import requests
import time
from typing import Dict, List, Any, Callable
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class DynamicIntentRouterHTTPSDK:
    def __init__(self, chroma_host: str = "localhost", chroma_port: int = 8000,
                 model_api_url: str = "http://localhost:11434/api/generate"):
        # 初始化嵌入模型
        print("正在加载BAAI/bge-m3嵌入模型...")
        self.embedding_model = SentenceTransformer('BAAI/bge-m3')
        print("嵌入模型加载完成")

        # 初始化ChromaDB HTTP客户端
        print("正在初始化ChromaDB HTTP客户端...")
        self.chroma_client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
            settings=Settings(allow_reset=True)
        )

        # 获取集合
        self.collection_name = "script_scenes_v3"
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            print("ChromaDB HTTP客户端连接成功")
        except Exception as e:
            print(f"连接ChromaDB失败: {e}")
            # 创建新集合作为备用方案
            try:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                print("创建了新集合")
            except Exception as e2:
                print(f"创建集合也失败: {e2}")
                raise

        # 本地DeepSeek-LoRA模型API端点
        self.model_api_url = model_api_url

        # 预定义能力映射
        self.capabilities = {
            "character_prequel": {
                "handler": self.handle_character_prequel,
                "description": "为角色创作前传故事"
            },
            "character_sequel": {
                "handler": self.handle_character_sequel,
                "description": "为角色创作后续故事"
            },
            "scene_continuation": {
                "handler": self.handle_scene_continuation,
                "description": "续写现有场景"
            },
            "scene_expansion": {
                "handler": self.handle_scene_expansion,
                "description": "扩展剧本场景"
            },
            "style_imitation": {
                "handler": self.handle_style_imitation,
                "description": "模仿剧本风格创作"
            },
            "plot_restructuring": {
                "handler": self.handle_plot_restructuring,
                "description": "重构剧本结构"
            }
        }

        # 提取已知人物列表（用于意图分析）
        self.known_characters = self.extract_known_characters()
        print(f"从数据库中提取了 {len(self.known_characters)} 个已知人物")

    def extract_known_characters(self) -> List[str]:
        """从数据库中提取已知人物列表"""
        try:
            # 获取所有文档的元数据
            all_metadatas = self.collection.get(include=["metadatas"])["metadatas"]

            # 提取所有人物字段
            characters_set = set()
            for metadata in all_metadatas:
                if "characters" in metadata and metadata["characters"]:
                    # 分割字符串获取人物列表
                    chars = [c.strip() for c in metadata["characters"].split(",") if c.strip()]
                    characters_set.update(chars)

            return list(characters_set)
        except Exception as e:
            print(f"提取已知人物时出错: {e}")
            return []

    def call_local_model(self, prompt: str, max_retries: int = 3) -> str:
        """调用本地Ollama模型，支持重试机制"""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.model_api_url,
                    json={
                        "model": "deepseek-lora_Q4",
                        "prompt": prompt,
                        "max_tokens": 300,
                        "temperature": 0.7,
                        "stream": False
                    },
                    timeout=120  # 增加超时时间
                )
                if response.status_code == 200:
                    return response.json().get("response", "").strip()
                else:
                    print(f"Ollama API返回错误: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"调用Ollama模型失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {2 ** attempt} 秒后重试...")
                    time.sleep(2 ** attempt)  # 指数退避策略
                else:
                    print("已达到最大重试次数")

        return "抱歉，模型服务暂时不可用，请稍后再试。"

    def understand_intent(self, user_query: str) -> Dict[str, Any]:
        """使用本地模型理解用户意图"""
        print("正在分析用户意图...")

        # 简化提示词以提高响应速度
        prompt = f"""
        分析以下用户查询的意图，并以JSON格式返回结果:

        用户查询: "{user_query}"

        返回JSON格式:
        {{
            "intent_type": "意图类型",
            "primary_character": "主要角色",
            "scene_reference": "场景引用",
            "confidence": 0.9,
            "description": "意图描述"
        }}

        可用意图类型: {", ".join(self.capabilities.keys())}
        """

        response = self.call_local_model(prompt)

        try:
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                intent_info = json.loads(json_match.group())
                print(f"意图分析结果: {intent_info}")
                return intent_info
        except json.JSONDecodeError:
            print(f"无法解析模型响应: {response}")

        # 备用方案：基于关键词的简单分析
        return self.fallback_intent_analysis(user_query)

    def fallback_intent_analysis(self, user_query: str) -> Dict[str, Any]:
        """备用意图分析（当模型不可用时使用）"""
        query_lower = user_query.lower()

        # 检测前传请求
        if any(word in query_lower for word in ["前传", "背景故事", "起源"]):
            # 尝试提取人物
            character = self.extract_character(user_query)
            return {
                "intent_type": "character_prequel",
                "primary_character": character,
                "scene_reference": None,
                "confidence": 0.7,
                "description": f"为{character or '某角色'}创作前传故事"
            }

        # 检测续集请求
        if any(word in query_lower for word in ["续集", "后续", "后传"]):
            character = self.extract_character(user_query)
            return {
                "intent_type": "character_sequel",
                "primary_character": character,
                "scene_reference": None,
                "confidence": 0.7,
                "description": f"为{character or '某角色'}创作后续故事"
            }

        # 检测续写请求
        if any(word in query_lower for word in ["续写", "接着", "接下来", "场景"]):
            scene_ref = self.extract_scene_reference(user_query)
            return {
                "intent_type": "scene_continuation",
                "primary_character": None,
                "scene_reference": scene_ref or self.extract_scene_number(user_query),
                "confidence": 0.7,
                "description": f"续写{scene_ref or '当前'}场景"
            }

        # 默认意图
        return {
            "intent_type": "scene_expansion",
            "primary_character": None,
            "scene_reference": None,
            "confidence": 0.5,
            "description": "扩展剧本场景"
        }

    def extract_character(self, query: str) -> str:
        """从查询中提取人物名称"""
        # 优先从已知人物中匹配
        for character in self.known_characters:
            if character in query:
                return character

        # 如果没有匹配的已知人物，尝试提取新的人物名称
        name_pattern = r'[\\u4e00-\\u9fa5]{2,3}'
        names = re.findall(name_pattern, query)

        # 过滤常见非人名词汇
        common_words = ["剧本", "场景", "续集", "前传", "后传", "创作", "编写", "故事", "用户", "希望"]
        for name in names:
            if name not in common_words:
                return name

        return None

    def extract_scene_reference(self, query: str) -> str:
        """从查询中提取场景引用"""
        # 匹配场景编号模式 (如 1-1, 2-3 等)
        scene_pattern = r'\d+-\d+'
        match = re.search(scene_pattern, query)
        if match:
            return match.group(0)

        # 匹配"第X集"模式
        episode_pattern = r'第[零一二三四五六七八九十百千万\d]+集'
        match = re.search(episode_pattern, query)
        if match:
            return match.group(0)

        return None

    def extract_scene_number(self, query: str) -> str:
        """从查询中提取场景编号"""
        # 尝试提取数字格式的场景编号
        scene_pattern = r'(\d+-\d+)'
        match = re.search(scene_pattern, query)
        if match:
            return match.group(1)

        return None

    def retrieve_relevant_scenes(self, intent_info: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关场景"""
        print("正在检索相关场景...")

        # 构建查询
        query_text = intent_info["description"]

        # 生成查询嵌入
        query_embedding = self.embedding_model.encode([query_text]).tolist()[0]

        # 尝试多种检索策略
        strategies = [
            self._retrieve_with_character_and_scene_filter,
            self._retrieve_with_character_filter,
            self._retrieve_with_scene_filter,
            self._retrieve_without_filter
        ]

        for strategy in strategies:
            try:
                results = strategy(query_embedding, intent_info, top_k)
                if results:
                    print(f"使用 {strategy.__name__} 检索到 {len(results)} 个相关场景")
                    return results
            except Exception as e:
                print(f"检索策略 {strategy.__name__} 失败: {e}")
                continue

        print("所有检索策略均失败")
        return []

    def _retrieve_with_character_and_scene_filter(self, query_embedding: List[float], intent_info: Dict[str, Any],
                                                  top_k: int) -> List[Dict[str, Any]]:
        """同时使用人物和场景过滤器检索"""
        character = intent_info.get("primary_character")
        scene_ref = intent_info.get("scene_reference")

        if not character and not scene_ref:
            return []

        # 获取所有文档
        all_results = self.collection.get(include=["metadatas", "documents"])

        # 过滤文档
        filtered_results = []
        for i, (metadata, document) in enumerate(zip(all_results["metadatas"], all_results["documents"])):
            # 检查人物匹配
            character_match = not character or self.is_character_in_metadata(metadata, character)

            # 检查场景匹配
            scene_match = not scene_ref or self.is_scene_in_metadata(metadata, scene_ref)

            if character_match and scene_match:
                # 计算相似度
                embedding = self.embedding_model.encode([document]).tolist()[0]
                distance = self._cosine_similarity(query_embedding, embedding)

                filtered_results.append({
                    "id": all_results["ids"][i],
                    "metadata": metadata,
                    "content": document,
                    "distance": distance
                })

        # 按相似度排序并返回前top_k个
        filtered_results.sort(key=lambda x: x["distance"], reverse=True)
        return filtered_results[:top_k]

    def is_scene_in_metadata(self, metadata: Dict[str, Any], scene_ref: str) -> bool:
        """检查元数据中是否包含指定场景"""
        if "scene_number" not in metadata:
            return False

        # 完全匹配
        if scene_ref == metadata["scene_number"]:
            return True

        # 部分匹配（处理可能的场景格式差异）
        scene_number = self.extract_scene_number(scene_ref)
        if scene_number and scene_number == self.extract_scene_number(metadata["scene_number"]):
            return True

        return False

    def _retrieve_with_scene_filter(self, query_embedding: List[float], intent_info: Dict[str, Any], top_k: int) -> \
    List[Dict[str, Any]]:
        """使用场景过滤器检索"""
        scene_ref = intent_info.get("scene_reference")
        if not scene_ref:
            return []

        # 尝试提取场景编号
        scene_number = self.extract_scene_number(scene_ref)
        if not scene_number:
            return []

        # 构建场景过滤器
        where_filter = {"scene_number": {"$eq": scene_number}}

        # 执行查询
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "documents", "distances"]
        )

        return self._format_results(results)

    def _retrieve_with_character_filter(self, query_embedding: List[float], intent_info: Dict[str, Any], top_k: int) -> \
    List[Dict[str, Any]]:
        """使用人物过滤器检索"""
        character = intent_info.get("primary_character")
        if not character:
            return []

        # 如果primary_character是列表，取第一个元素
        if isinstance(character, list) and character:
            character = character[0]

        # 获取所有文档
        all_results = self.collection.get(include=["metadatas", "documents"])

        # 过滤包含指定人物的文档
        filtered_results = []
        for i, (metadata, document) in enumerate(zip(all_results["metadatas"], all_results["documents"])):
            if self.is_character_in_metadata(metadata, character):
                # 计算相似度
                embedding = self.embedding_model.encode([document]).tolist()[0]
                distance = self._cosine_similarity(query_embedding, embedding)

                filtered_results.append({
                    "id": all_results["ids"][i],
                    "metadata": metadata,
                    "content": document,
                    "distance": distance
                })

        # 按相似度排序并返回前top_k个
        filtered_results.sort(key=lambda x: x["distance"], reverse=True)
        return filtered_results[:top_k]

    def _retrieve_without_filter(self, query_embedding: List[float], intent_info: Dict[str, Any], top_k: int) -> List[
        Dict[str, Any]]:
        """无过滤器检索"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # 获取更多结果以便后续过滤
            include=["metadatas", "documents", "distances"]
        )

        return self._format_results(results)

    def _format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """格式化查询结果"""
        formatted_results = []
        for i, (metadata, document, distance) in enumerate(zip(
                results["metadatas"][0],
                results["documents"][0],
                results["distances"][0]
        )):
            formatted_results.append({
                "id": results["ids"][0][i],
                "metadata": metadata,
                "content": document,
                "distance": distance
            })

        return formatted_results

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0

    def is_character_in_metadata(self, metadata: Dict[str, Any], character: str) -> bool:
        """检查元数据中是否包含指定人物"""
        if "characters" not in metadata or not metadata["characters"]:
            return False

        # 将逗号分隔的字符串转换为列表
        characters_list = [c.strip() for c in metadata["characters"].split(",") if c.strip()]

        # 检查是否包含指定人物（支持部分匹配）
        for char_in_db in characters_list:
            # 完全匹配
            if character == char_in_db:
                return True

            # 部分匹配（处理可能的人物描述）
            if character in char_in_db:
                return True

            # 处理可能的人物别名或简称
            if self.is_character_alias(character, char_in_db):
                return True

        return False

    def is_character_alias(self, query_char: str, db_char: str) -> bool:
        """检查是否为同一人物的不同称呼"""
        # 定义已知的人物别名映射
        alias_map = {
            "楚音韵": ["音韵", "楚小姐", "楚姑娘"],
            "秦云上": ["云上", "秦公子", "秦先生"],
            # 可以继续添加其他人物别名
        }

        # 检查查询人物是否是数据库中人物的别名
        if query_char in alias_map.get(db_char, []):
            return True

        # 检查数据库中人物是否是查询人物的别名
        if db_char in alias_map.get(query_char, []):
            return True

        return False

    def route_query(self, user_query: str) -> Dict[str, Any]:
        """路由查询到适当的处理逻辑"""
        print(f"\n处理用户查询: {user_query}")

        # 理解用户意图
        intent_info = self.understand_intent(user_query)

        # 检索相关场景
        relevant_scenes = self.retrieve_relevant_scenes(intent_info)

        # 获取处理函数
        intent_type = intent_info.get("intent_type", "scene_expansion")
        handler = self.capabilities.get(intent_type, {}).get("handler", self.handle_scene_expansion)

        # 构建上下文
        context = {
            "user_query": user_query,
            "intent_info": intent_info,
            "relevant_scenes": relevant_scenes
        }

        # 调用处理函数
        response = handler(context)

        return {
            "intent_info": intent_info,
            "relevant_scenes": relevant_scenes,
            "response": response
        }

    # 各种处理函数的实现
    def handle_character_prequel(self, context: Dict[str, Any]) -> str:
        """处理人物前传请求"""
        character = context["intent_info"].get("primary_character", "某角色")
        scenes = context["relevant_scenes"]

        # 构建提示
        prompt = f"""
        为角色{character}创作一个前传故事。

        角色相关信息:
        {json.dumps([s['metadata'] for s in scenes[:2]], ensure_ascii=False, indent=2) if scenes else '无'}

        请创作一个简短的前传场景，展示{character}的起源故事。
        保持剧本的原有风格和格式，包括适当的动作指示和对话格式。
        """

        response = self.call_local_model(prompt)
        return response

    def handle_character_sequel(self, context: Dict[str, Any]) -> str:
        """处理人物后传请求"""
        character = context["intent_info"].get("primary_character", "某角色")
        scenes = context["relevant_scenes"]

        prompt = f"""
        为角色{character}创作一个后续故事。

        角色相关信息:
        {json.dumps([s['metadata'] for s in scenes[:2]], ensure_ascii=False, indent=2) if scenes else '无'}

        请创作一个简短的后续场景，展示{character}的未来发展。
        保持剧本的原有风格和格式，包括适当的动作指示和对话格式。
        """

        response = self.call_local_model(prompt)
        return response

    def handle_scene_continuation(self, context: Dict[str, Any]) -> str:
        """处理场景续写请求"""
        scene_ref = context["intent_info"].get("scene_reference", "当前")
        scenes = context["relevant_scenes"]

        # 如果没有检索到相关场景，尝试直接使用场景引用
        if not scenes and scene_ref:
            # 尝试直接获取指定场景
            try:
                scene_results = self.collection.get(
                    where={"scene_number": {"$eq": scene_ref}},
                    include=["metadatas", "documents"]
                )
                if scene_results["ids"]:
                    scenes = [{
                        "id": scene_results["ids"][0],
                        "metadata": scene_results["metadatas"][0],
                        "content": scene_results["documents"][0],
                        "distance": 1.0
                    }]
            except Exception as e:
                print(f"直接获取场景失败: {e}")

        prompt = f"""
        续写{scene_ref}场景。

        相关场景内容:
        {scenes[0]['content'] if scenes else '无'}

        请续写这个场景，保持风格一致，包括适当的动作指示和对话格式。
        """

        response = self.call_local_model(prompt)
        return response

    def handle_scene_expansion(self, context: Dict[str, Any]) -> str:
        """处理场景扩展请求"""
        scenes = context["relevant_scenes"]

        prompt = f"""
        扩展以下剧本场景:

        相关场景内容:
        {scenes[0]['content'] if scenes else '无'}

        请扩展这个场景，添加更多细节和对话，保持风格一致。
        """

        response = self.call_local_model(prompt)
        return response

    def handle_style_imitation(self, context: Dict[str, Any]) -> str:
        """处理风格模仿请求"""
        scenes = context["relevant_scenes"]

        prompt = f"""
        模仿以下剧本风格创作一个新场景:

        风格示例:
        {scenes[0]['content'] if scenes else '无'}

        请创作一个保持相同风格的新场景，包括适当的动作指示和对话格式。
        """

        response = self.call_local_model(prompt)
        return response

    def handle_plot_restructuring(self, context: Dict[str, Any]) -> str:
        """处理剧情重构请求"""
        scenes = context["relevant_scenes"]

        prompt = f"""
        分析以下剧本结构并提供重构建议:

        剧本相关信息:
        {json.dumps([s['metadata'] for s in scenes[:3]], ensure_ascii=False, indent=2) if scenes else '无'}

        请提供剧情重构建议，包括如何调整故事结构、角色发展等。
        保持专业和具体的建议。
        """

        response = self.call_local_model(prompt)
        return response

    def debug_database_content(self, character_name: str = None):
        """调试数据库内容，查看特定人物的存储情况"""
        try:
            # 获取所有文档
            all_data = self.collection.get(include=["metadatas", "documents"])

            print(f"数据库中共有 {len(all_data['ids'])} 个文档")

            # 如果有指定人物，筛选相关文档
            if character_name:
                print(f"\n查找包含人物 '{character_name}' 的文档:")
                found_count = 0
                for i, (metadata, document) in enumerate(zip(all_data["metadatas"], all_data["documents"])):
                    if "characters" in metadata and character_name in metadata["characters"]:
                        found_count += 1
                        print(f"\n文档 {i + 1}:")
                        print(f"ID: {all_data['ids'][i]}")
                        print(f"元数据: {metadata}")
                        print(f"内容摘要: {document[:100]}...")

                print(f"\n找到 {found_count} 个包含 '{character_name}' 的文档")

            # 显示所有人物的统计
            all_characters = {}
            for metadata in all_data["metadatas"]:
                if "characters" in metadata and metadata["characters"]:
                    chars = [c.strip() for c in metadata["characters"].split(",") if c.strip()]
                    for char in chars:
                        all_characters[char] = all_characters.get(char, 0) + 1

            print(f"\n数据库中所有人物及其出现次数:")
            for char, count in sorted(all_characters.items(), key=lambda x: x[1], reverse=True):
                print(f"  {char}: {count}次")

        except Exception as e:
            print(f"调试数据库内容时出错: {e}")

# 主程序
# 在您的DynamicIntentRouterHTTPSDK类中添加上述方法

# 使用示例
if __name__ == "__main__":
    # 从环境变量获取配置，或使用默认值
    import os

    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
    model_api_url = os.getenv("MODEL_API_URL", "http://localhost:11434/api/generate")

    router = DynamicIntentRouterHTTPSDK(chroma_host, chroma_port, model_api_url)

    # 首先调试数据库内容
    print("=" * 50)
    print("调试数据库内容:")
    router.debug_database_content("楚音韵")
    router.debug_database_content("秦云上")

    # 然后测试查询
    print("=" * 50)
    print("测试查询:")
    test_queries = [
        "为楚音韵写一个前传",
        "为秦云上编写续集",
        "续写场景1-1",
        "创新一下这个故事",
        "把剧本从100集缩减到50集"
    ]

    for query in test_queries:
        print("\n" + "=" * 50)
        result = router.route_query(query)

        print(f"\n查询: {query}")
        print(f"识别意图: {result['intent_info']['intent_type']}")
        print(f"意图描述: {result['intent_info']['description']}")
        print(f"相关场景数量: {len(result['relevant_scenes'])}")
        if result['relevant_scenes']:
            print(f"检索到的场景: {[s['id'] for s in result['relevant_scenes']]}")
        print(f"\n响应:\n{result['response'][:200]}...")