import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorDBWriterHTTPSDK:
    def __init__(self, chroma_host: str = "localhost", chroma_port: int = 8000):
        # 设置目录路径
        self.cleaned_dir = Path("data/cleaned")

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

        # 创建或获取集合
        self.collection_name = "script_scenes_v3"
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("ChromaDB HTTP客户端准备就绪")

    def process_all_md_files(self):
        """处理所有MD文件并存储到向量数据库"""
        print("开始处理MD文件...")
        md_files = list(self.cleaned_dir.glob("*.md"))

        if not md_files:
            print(f"在 {self.cleaned_dir} 目录中未找到MD文件")
            return

        for md_file in md_files:
            print(f"\n处理文件: {md_file.name}")

            # 读取MD文件内容
            with open(md_file, "r", encoding="utf-8") as f:
                md_content = f.read()

            # 解析剧本结构
            parsed_script = self.parse_script_structure(md_content)

            # 准备向量化数据
            documents, metadatas, embeddings, ids = self.prepare_vector_data(parsed_script, md_file.stem)

            # 添加到向量数据库
            if documents:
                self.add_to_collection(documents, metadatas, embeddings, ids)

    def parse_script_structure(self, text: str) -> Dict[str, Any]:
        """解析剧本结构"""
        lines = text.strip().split('\n')
        if not lines:
            return {"series_name": "未知", "scenes": []}

        # 提取剧名
        series_name = lines[0].strip()

        scenes = []
        current_episode = None
        current_scene = None
        scene_content = []
        current_characters = []

        for i, line in enumerate(lines[1:]):
            line = line.strip()
            if not line:
                continue

            # 检查是否是集标记
            episode_match = re.search(r'第\s*([零一二三四五六七八九十百千万\d]+)\s*集', line)
            if episode_match:
                current_episode = self.chinese_to_arabic(episode_match.group(1))
                continue

            # 检查是否是场景标记
            scene_match = re.match(r'(\d+-\d+)\s*(.*)', line)
            if scene_match:
                # 保存上一个场景
                if current_scene is not None and scene_content:
                    scenes.append({
                        "episode": current_episode,
                        "scene_number": current_scene["number"],
                        "location": current_scene["location"],
                        "time_of_day": current_scene["time_of_day"],
                        "interior_exterior": current_scene["interior_exterior"],
                        "characters": current_characters,
                        "content": "\n".join(scene_content)
                    })

                # 解析新场景头部
                scene_header = scene_match.group(2)
                scene_info = self.parse_scene_header(scene_header)

                # 开始新场景
                current_scene = {
                    "number": scene_match.group(1),
                    "location": scene_info["location"],
                    "time_of_day": scene_info["time_of_day"],
                    "interior_exterior": scene_info["interior_exterior"]
                }
                current_characters = []
                scene_content = []
                continue

            # 检查是否是人物行
            if line.startswith('人物：'):
                characters_text = line[3:].strip()
                current_characters = self.parse_characters(characters_text)
                continue

            # 添加到当前场景内容
            if current_scene is not None:
                scene_content.append(line)

        # 添加最后一个场景
        if current_scene is not None and scene_content:
            scenes.append({
                "episode": current_episode,
                "scene_number": current_scene["number"],
                "location": current_scene["location"],
                "time_of_day": current_scene["time_of_day"],
                "interior_exterior": current_scene["interior_exterior"],
                "characters": current_characters,
                "content": "\n".join(scene_content)
            })

        result = {
            "series_name": series_name,
            "scenes": scenes
        }

        print(f"解析完成，共{len(scenes)}个场景")
        return result

    def parse_scene_header(self, header: str) -> Dict[str, Any]:
        """解析场景头部信息"""
        # 默认值
        result = {
            "location": "未知",
            "time_of_day": "未知",
            "interior_exterior": "未知"
        }

        # 尝试解析地点、时间、内外景
        parts = [p.strip() for p in header.split('，') if p.strip()]
        if len(parts) >= 3:
            result["location"] = parts[0]
            result["time_of_day"] = parts[1]
            result["interior_exterior"] = parts[2]

        return result

    def parse_characters(self, characters_text: str) -> List[str]:
        """解析人物列表"""
        characters = []
        # 使用逗号分割
        parts = [p.strip() for p in characters_text.split('，') if p.strip()]
        for part in parts:
            # 移除描述部分（如果有）
            if '【' in part and '】' in part:
                char_match = re.match(r'([^【]+)【', part)
                if char_match:
                    characters.append(char_match.group(1).strip())
            else:
                characters.append(part)

        return characters

    def chinese_to_arabic(self, chinese_num: str) -> str:
        """将中文数字转换为阿拉伯数字"""
        # 如果已经是阿拉伯数字，直接返回
        if chinese_num.isdigit():
            return chinese_num

        # 中文数字映射
        chinese_digits = {
            '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
            '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
            '十': 10, '百': 100, '千': 1000, '万': 10000
        }

        # 特殊处理"十"开头的数字
        if chinese_num.startswith('十'):
            chinese_num = '一' + chinese_num

        result = 0
        temp = 0

        for char in chinese_num:
            if char in chinese_digits:
                digit = chinese_digits[char]

                if digit >= 10:  # 十、百、千、万
                    if temp == 0:
                        temp = 1
                    result += temp * digit
                    temp = 0
                else:  # 零到九
                    temp = temp * 10 + digit
            else:
                # 遇到非数字字符，停止转换
                break

        result += temp

        return str(result)

    def prepare_vector_data(self, parsed_script: Dict[str, Any], file_prefix: str) -> Tuple[
        List[str], List[Dict], List[List[float]], List[str]]:
        """准备向量化数据"""
        documents = []
        metadatas = []
        embeddings = []
        ids = []

        for i, scene in enumerate(parsed_script["scenes"]):
            # 创建文档内容
            content = f"""
            剧名: {parsed_script["series_name"]}
            集数: {scene["episode"]}
            场景: {scene["scene_number"]}
            地点: {scene["location"]}
            时间: {scene["time_of_day"]}
            内外景: {scene["interior_exterior"]}
            人物: {', '.join(scene["characters"])}
            内容: {scene["content"]}
            """

            # 创建元数据 - 确保人物字段正确处理
            # 移除人物描述部分，只保留纯名称
            clean_characters = []
            for char in scene["characters"]:
                # 移除描述部分（如果有）
                if '【' in char and '】' in char:
                    char_match = re.match(r'([^【]+)【', char)
                    if char_match:
                        clean_characters.append(char_match.group(1).strip())
                else:
                    clean_characters.append(char)

            metadata = {
                "series_name": parsed_script["series_name"],
                "episode": scene["episode"],
                "scene_number": scene["scene_number"],
                "location": scene["location"],
                "time_of_day": scene["time_of_day"],
                "interior_exterior": scene["interior_exterior"],
                # 使用清理后的人物列表
                "characters": ", ".join(clean_characters),
                "source_file": file_prefix
            }

            # 生成嵌入向量
            embedding = self.embedding_model.encode(content).tolist()

            documents.append(content)
            metadatas.append(metadata)
            embeddings.append(embedding)
            ids.append(f"{file_prefix}_scene_{i}")

        print(f"为 {file_prefix} 准备了 {len(documents)} 个文档")
        return documents, metadatas, embeddings, ids

    def add_to_collection(self, documents: List[str], metadatas: List[Dict], embeddings: List[List[float]],
                          ids: List[str]):
        """将数据添加到集合"""
        print(f"正在将 {len(documents)} 个文档添加到集合...")

        try:
            # 使用SDK的add方法
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
            print("数据已成功添加到向量数据库")
        except Exception as e:
            print(f"添加数据时出错: {e}")
            # 如果出错，尝试分批添加
            self.batch_add_to_collection(documents, metadatas, embeddings, ids)

    def batch_add_to_collection(self, documents: List[str], metadatas: List[Dict], embeddings: List[List[float]],
                                ids: List[str], batch_size: int = 50):
        """分批将数据添加到集合"""
        print(f"尝试分批添加数据，批次大小: {batch_size}")

        total = len(documents)
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            batch_docs = documents[i:end_idx]
            batch_metas = metadatas[i:end_idx]
            batch_embs = embeddings[i:end_idx]
            batch_ids = ids[i:end_idx]

            print(f"添加批次 {i // batch_size + 1}/{(total - 1) // batch_size + 1} ({end_idx - i} 个文档)")

            try:
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=batch_embs,
                    ids=batch_ids
                )
                print(f"批次 {i // batch_size + 1} 添加成功")
            except Exception as e:
                print(f"批次 {i // batch_size + 1} 添加失败: {e}")
                # 如果批次添加失败，尝试逐个添加
                self.single_add_to_collection(batch_docs, batch_metas, batch_embs, batch_ids)

    def single_add_to_collection(self, documents: List[str], metadatas: List[Dict], embeddings: List[List[float]],
                                 ids: List[str]):
        """逐个将数据添加到集合"""
        print("尝试逐个添加文档...")

        for i, (doc, meta, emb, id_) in enumerate(zip(documents, metadatas, embeddings, ids)):
            try:
                self.collection.add(
                    documents=[doc],
                    metadatas=[meta],
                    embeddings=[emb],
                    ids=[id_]
                )
                print(f"文档 {i + 1}/{len(documents)} 添加成功")
            except Exception as e:
                print(f"文档 {i + 1}/{len(documents)} 添加失败: {e}")
                # 如果单个文档添加失败，尝试简化元数据
                self.add_with_simplified_metadata(doc, meta, emb, id_)

    def add_with_simplified_metadata(self, document: str, metadata: Dict, embedding: List[float], id_: str):
        """使用简化元数据添加文档"""
        print(f"尝试使用简化元数据添加文档 {id_}...")

        # 创建简化元数据，只包含基本字段
        simplified_metadata = {
            "series_name": metadata.get("series_name", "未知"),
            "episode": metadata.get("episode", "未知"),
            "scene_number": metadata.get("scene_number", "未知"),
            "source_file": metadata.get("source_file", "未知")
        }

        try:
            self.collection.add(
                documents=[document],
                metadatas=[simplified_metadata],
                embeddings=[embedding],
                ids=[id_]
            )
            print(f"文档 {id_} 使用简化元数据添加成功")
        except Exception as e:
            print(f"文档 {id_} 使用简化元数据添加失败: {e}")


# 主程序
if __name__ == "__main__":
    # 从环境变量获取ChromaDB主机和端口，或使用默认值
    import os

    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))

    writer = VectorDBWriterHTTPSDK(chroma_host, chroma_port)
    writer.process_all_md_files()