import ollama
import traceback
from chromadb import HttpClient  # 官方 Chroma SDK
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# 配置
# -----------------------------
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "my_collection"
# OLLAMA_MODEL_NAME = "deepseek-lora"
OLLAMA_MODEL_NAME = "deepseek-lora"
TOP_K = 3

# 1. 连接 Chroma HTTP 服务
client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# 2. 获取已有 collection，不创建新 collection，不绑定 embedding
collection = client.get_collection(name=COLLECTION_NAME)
# 创建 LangChain 的 Chroma 对象，绑定远程 client
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# -----------------------------
# 循环接收用户问题
# -----------------------------
while True:
    user_input = input("用户问题：")
    if user_input.strip() == "/88":
        print("退出 RAG 系统。")
        break

    try:
        # 直接用 query_texts 查询，SDK 会自动用 collection 的 embedding
        embedding = embedding_model.embed_query(str(user_input))
        results = collection.query(
            query_embeddings=[embedding],
            n_results=5
        )
        # print(results)
        # 拼接上下文
        context_docs = results.get("documents", [[]])[0]
        context_text = "\n".join([str(doc) for doc in context_docs]) if context_docs else "没有相关文档。"

        # 交给 Ollama 生成回答
        prompt = f"请在 500 个字以内回答问题：\n根据以下文档回答问题：\n{context_text}\n\n问题：{user_input}\n请直接回答并在回答前回顾检查一下你的回答的合理性："


        # /88print(prompt)
        # 3️⃣ 调用 Ollama 模型生成回答
        response = ollama.generate(
            model="deepseek-lora",
            prompt=prompt
        )
        # 直接获取响应内容
        generated_text = response['response'] if hasattr(response, '__contains__') and 'response' in response else str(
            response)

        print("RAG 回答：", generated_text)

    except Exception as e:
        print("发生错误：")
        traceback.print_exc()

