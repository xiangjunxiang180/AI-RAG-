# 🚀 【核心修复】放在最顶部，强制国内镜像，彻底解决超时
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_DISABLE_RETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# ===================== 仅需修改这里 =====================
YOUR_QWEN_API_KEY = "YOUR API"
DOC_PATH = "./test_data"
VECTOR_INDEX_PATH = "./my_rag_index"
# ========================================================

import os
import gradio as gr
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import dashscope

# ===================== 1. 初始化嵌入模型 =====================
from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("✅ 嵌入模型初始化完成")

# ===================== 2. 加载/构建向量库 =====================
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

if os.path.exists(VECTOR_INDEX_PATH) and os.listdir(VECTOR_INDEX_PATH):
    print("🔍 加载本地向量库...")
    vector_db = FAISS.load_local(
        VECTOR_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True
    )
    print("✅ 本地向量库加载成功")
else:
    print("📄 首次构建向量库...")
    documents = []
    for filename in os.listdir(DOC_PATH):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(DOC_PATH, filename), encoding="utf-8")
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    vector_db = FAISS.from_documents(split_docs, embedding_model)
    vector_db.save_local(VECTOR_INDEX_PATH)
    print("✅ 向量库构建完成")

# ===================== 3. 混合搜索检索器 =====================
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

vector_retriever = vector_db.as_retriever(search_kwargs={"k": 10})
bm25_retriever = BM25Retriever.from_documents(vector_db.docstore._dict.values(), k=10)
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])
print("✅ 混合搜索检索器初始化完成")

# ===================== 4. 轻量级重排序（13MB，永无404） =====================
from sentence_transformers import CrossEncoder
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-2-v2")
print("✅ 重排序模型加载完成")

def rerank_documents(query, documents, top_k=3):
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker_model.predict(pairs)
    scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k]]

# ===================== 5. 查询重写 =====================
dashscope.api_key = YOUR_QWEN_API_KEY
def rewrite_query(user_question):
    prompt = f"""将用户问题重写为3个检索查询，每行一个，无多余内容：{user_question}"""
    response = dashscope.Generation.call(model="qwen-turbo", prompt=prompt, temperature=0.7)
    queries = [q.strip() for q in response.output.text.strip().split("\n") if q.strip()]
    return queries[:3]

# ===================== 6. RAG核心问答函数 =====================
def rag_chat(user_question):
    if not user_question:
        return "请输入你的问题！"
    
    # 1. 查询重写
    queries = rewrite_query(user_question)
    # 2. 混合搜索
    all_docs = []
    seen = set()
    for q in queries:
        docs = ensemble_retriever.get_relevant_documents(q)
        for d in docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                all_docs.append(d)
    # 3. 重排序
    relevant_docs = rerank_documents(user_question, all_docs)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # 4. 生成回答
    prompt = f"""仅根据上下文回答，不编造。不知道就说未找到相关信息。
上下文：{context}
问题：{user_question}"""
    
    response = dashscope.Generation.call(model="qwen-turbo", prompt=prompt, temperature=0.3)
    return response.output.text

# ===================== 7. Gradio 网页界面 =====================
with gr.Blocks(title="菜谱RAG智能问答") as demo:
    gr.Markdown("# 🍳 菜谱智能问答系统")
    gr.Markdown("基于 **RAG + 重排序 + 查询重写** 打造的工业级问答助手，支持300+道菜谱查询")
    
    with gr.Row():
        question_input = gr.Textbox(label="输入你的问题", placeholder="例如：番茄炒蛋放多少盐？红烧肉怎么做？")
    
    with gr.Row():
        submit_btn = gr.Button("开始提问", variant="primary")
    
    answer_output = gr.Textbox(label="AI 回答", lines=8)
    
    # 绑定按钮事件
    submit_btn.click(rag_chat, inputs=question_input, outputs=answer_output)
    # 回车发送
    question_input.submit(rag_chat, inputs=question_input, outputs=answer_output)

# ===================== 启动服务 =====================
if __name__ == "__main__":
    print("\n🎉 网页Demo启动成功！打开浏览器访问下方地址即可使用")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
