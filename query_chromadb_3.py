from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import HttpClient  # 官方 Chroma SDK

# 连接到远程 Chroma Server
client = HttpClient(host="localhost", port=8000)

# 创建 LangChain 的 Chroma 对象，绑定远程 client
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 获取集合（或新建）
# 获取集合（不存在则创建）
collection = client.get_or_create_collection(name="my_collection")

# 查询
query = "沈老二是谁？文中有提及到沈老二吗？"
embedding = embedding_model.embed_query(query)

results = collection.query(
    query_embeddings=[embedding],
    n_results=5
)

print("\n=== 查询结果 ===")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"- {doc[:100]}...  (meta: {meta})")