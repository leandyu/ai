import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 数据路径
cleaned_dir = os.path.join(os.path.dirname(__file__), "../data/cleaned")
vector_db_dir = os.path.join(os.path.dirname(__file__), "../vector_db")

# 加载文本
documents = []
for filename in os.listdir(cleaned_dir):
    if filename.endswith(".md") or filename.endswith(".txt"):
        filepath = os.path.join(cleaned_dir, filename)
        loader = TextLoader(filepath, encoding="utf-8")
        documents.extend(loader.load())

print(f"✅ Loaded {len(documents)} documents from {cleaned_dir}")

# 文本切分（小 chunk 避免整段返回）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
splits = text_splitter.split_documents(documents)
print(f"✅ Split into {len(splits)} chunks")

# 创建 Embedding
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 初始化 Chroma
db = Chroma.from_documents(
    documents=splits,
    embedding=embedding_model,
    persist_directory=vector_db_dir
)

print(f"✅ Added {len(splits)} chunks to Chroma DB at {vector_db_dir}")


