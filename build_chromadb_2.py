import os
import uuid
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ==============================
# 1. 连接 Chroma Server
# ==============================
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "my_collection_2"

client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# 删除旧 collection，确保干净写入
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"✅ 已删除旧集合: {COLLECTION_NAME}")
except Exception:
    print(f"ℹ️ 无需删除旧集合: {COLLECTION_NAME}")

# 重新创建 collection
collection = client.create_collection(name=COLLECTION_NAME)
print(f"✅ 创建新集合: {COLLECTION_NAME}")

# ==============================
# 2. 加载本地文档
# ==============================
cleaned_dir = os.path.join(os.path.dirname(__file__), "../data/cleaned")
documents = []

for filename in os.listdir(cleaned_dir):
    if filename.endswith(".md") or filename.endswith(".txt"):
        path = os.path.join(cleaned_dir, filename)
        loader = TextLoader(path, encoding="utf-8")
        documents.extend(loader.load())

print(f"✅ 已加载文档数量: {len(documents)}")

if not documents:
    print("❌ 没有找到文档，请检查 data/cleaned 目录！")
    exit(1)

# ==============================
# 3. 文本切块
# ==============================
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs = splitter.split_documents(documents)
print(f"✅ 切分后文档块数量: {len(docs)}")

# ==============================
# 4. 向量化
# ==============================
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
texts = [d.page_content for d in docs]
metadatas = [d.metadata for d in docs]

print("✅ 正在生成向量，请稍候...")
embeddings = embedding_model.embed_documents(texts)
print(f"✅ 向量生成完成，维度: {len(embeddings[0])}")

# ==============================
# 5. 写入 Chroma Server
# ==============================
ids = [str(uuid.uuid4()) for _ in range(len(docs))]

# 分批写入（避免一次性过大导致 HTTP 超时）
BATCH_SIZE = 100
for i in range(0, len(docs), BATCH_SIZE):
    batch_ids = ids[i:i+BATCH_SIZE]
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_metadatas = metadatas[i:i+BATCH_SIZE]
    batch_embeddings = embeddings[i:i+BATCH_SIZE]

    collection.add(
        ids=batch_ids,
        documents=batch_texts,
        metadatas=batch_metadatas,
        embeddings=batch_embeddings
    )
    print(f"✅ 已写入 {i + len(batch_ids)} / {len(docs)}")

print("🎯 所有数据已成功写入 Chroma Server！")
