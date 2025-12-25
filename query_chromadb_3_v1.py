import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_collection(name="langchain")

query = "沈老二是谁？文中有提及到沈老二吗？"
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
query_embedding = embedding_model.embed_query(query)

# 向量检索
semantic_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

# # 关键词检索
# keyword_results = collection.query(
#     query_texts=[query],  # Chroma 支持 query_texts，但效果不强
#     n_results=10,
#     where_document={"$contains": "沈老二"}
# )
#
# # 打印语义结果
# print("\n=== 🔍 语义结果 ===")
# for i, doc in enumerate(semantic_results["documents"][0]):
#     print(f"\nResult {i+1}: {doc}")

# # 打印关键词结果
# print("\n=== 🔍 关键词强制匹配结果 ===")
# for i, doc in enumerate(keyword_results["documents"][0]):
#     print(f"\nResult {i+1}: {doc}")

print("\n=== 查询结果 ===")
for i, doc in enumerate(semantic_results["documents"][0]):
    print(f"\nResult {i+1}: {doc}")