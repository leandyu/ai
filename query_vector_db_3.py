import os
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

vector_db_dir = os.path.abspath("vector_db")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

db = Chroma(persist_directory=vector_db_dir, embedding_function=embedding_model)

print(f"✅ Document count after insert: {db._collection.count()}")

# 查询
query = "请找出包含人物沈老二的描述"

print("\n=== 🔎 最相关结果 ===")
results = db.similarity_search(query, k=3)
for i, doc in enumerate(results):
    print(f"\nResult {i+1}: {doc.page_content}")


print("\n=== 🔎 最相关结果（句子级） ===")
# 使用 MMR（最大边际相关性），k=10 保留更多候选
results = db.max_marginal_relevance_search(query, k=10, fetch_k=20)

# 提取和 query 相关的句子
def extract_relevant_sentences(text, query, top_n=2):
    sentences = re.split(r"[。！？\n]", text)
    scores = [(s, len(set(query) & set(s))) for s in sentences if s.strip()]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scores[:top_n]]

for i, doc in enumerate(results):
    relevant_sentences = extract_relevant_sentences(doc.page_content, query)
    print(f"\nResult {i+1}:")
    for sent in relevant_sentences:
        print(f"  - {sent}")