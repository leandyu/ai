import streamlit as st
import ollama
from chromadb import HttpClient  # 官方 Chroma SDK
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# 配置
# -----------------------------
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "langchain"
OLLAMA_MODEL_NAME = "deepseek-lora"
TOP_K = 3

# 1. 连接 Chroma HTTP 服务
client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# 2. 获取已有 collection，不创建新 collection，不绑定 embedding
collection = client.get_collection(name=COLLECTION_NAME)
# 创建 LangChain 的 Chroma 对象，绑定远程 client
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 创建可访问ollama的客户端对象
client = ollama.Client(host="http://localhost:11434")

# 初始化消息记录
if 'message' not in st.session_state:
    st.session_state['message'] = []

# 添加标题
st.title("短剧创作AI聊天机器人")

# divider：分隔符
st.divider()

# 用户输入问题
prompt = st.chat_input("请输入你的问题：")

# 判断用户输入的问题
if prompt:
    # 将用户提问的问题加入到历史记录中
    st.session_state['message'].append({"role": "user", "content": prompt})

    # 直接用 query_texts 查询，SDK 会自动用 collection 的 embedding
    embedding = embedding_model.embed_query(str(prompt))
    results = collection.query(
        query_embeddings=[embedding],
        n_results=5
    )
    print(results)
    # 拼接上下文
    context_docs = results.get("documents", [[]])[0]
    context_text = "\n".join([str(doc) for doc in context_docs]) if context_docs else "没有相关文档。"

    # 交给 Ollama 生成回答
    prompt = f"请在 500 个字以内回答问题：\n根据以下文档回答问题：\n{context_text}\n\n问题：{prompt}\n请直接回答并在回答前回顾检查一下你的回答的合理性："

    # 遍历消息列表
    for message in st.session_state['message']:
        st.chat_message(message['role']).markdown(message['content'])

    with st.spinner("思考中..."):
        # 调用ollama调用api，并给出回答
        response = client.chat(
            model='deepseek-lora',
            messages=[{"role": "user", "content": prompt}]
        )
        # 从response中取出message和content
        st.session_state['message'].append({"role": "assistant", "content": response['message']['content']})

        # 在页面中显示ai回答信息
        st.chat_message("assistant").markdown(response['message']['content'])

