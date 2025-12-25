# scripts/test_vector_store.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = "vector_db"  # 必须与 build_vector_db_2.py 中一致

embed = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embed)

# 可选：打印集合大小（不同版本可能没有 _collection，可 try/except）
try:
    print("Collection size:", db._collection.count())
except Exception:
    pass

query = "宴会厅 恐怖分子 东欧恶龙组织 沈老二"
results = db.similarity_search_with_score(query, k=5)

print("\n=== 检索结果 ===")
for i, (doc, score) in enumerate(results, 1):
    src = doc.metadata.get("source", "")
    preview = doc.page_content[:200].replace("\n", " ")
    print(f"[{i}] score={score:.4f} | {src}\n{preview}\n")
