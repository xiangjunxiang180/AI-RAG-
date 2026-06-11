
import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import sqlite3
import dashscope
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# ===================== 配置 =====================
app = FastAPI(title="菜谱RAG API")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DOC_PATH = "./test_data"
VECTOR_INDEX_PATH = "./my_rag_index"
DB_PATH = "./rag_data.db"

# 初始化数据库
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_records
                 (id TEXT PRIMARY KEY, user_id TEXT, question TEXT, answer TEXT, 
                  response_time REAL, feedback INTEGER, feedback_text TEXT, 
                  created_at TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# ===================== RAG模型初始化 =====================
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings":True}
)

if os.path.exists(VECTOR_INDEX_PATH) and os.listdir(VECTOR_INDEX_PATH):
    vector_db = FAISS.load_local(VECTOR_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    documents = []
    for f in os.listdir(DOC_PATH):
        if f.endswith(".txt"):
            loader = TextLoader(os.path.join(DOC_PATH,f), encoding="utf-8")
            documents.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(documents)
    vector_db = FAISS.from_documents(splits, embedding_model)
    vector_db.save_local(VECTOR_INDEX_PATH)

vec_ret = vector_db.as_retriever(search_kwargs={"k":10})
bm25_ret = BM25Retriever.from_documents(vector_db.docstore._dict.values(), k=10)
ensemble_ret = EnsembleRetriever(retrievers=[bm25_ret, vec_ret], weights=[0.5,0.5])

dashscope.api_key = DASHSCOPE_API_KEY

# ===================== 数据模型 =====================
class ChatRequest(BaseModel):
    user_id: str
    question: str

class FeedbackRequest(BaseModel):
    record_id: str
    feedback: int  # 1=有用，0=没用
    feedback_text: str = ""

# ===================== API接口 =====================
@app.post("/api/chat")
async def chat(request: ChatRequest):
    start_time = datetime.now()
    
    # 你的RAG核心逻辑
    docs = ensemble_ret.get_relevant_documents(request.question)
    context = "\n\n".join([d.page_content for d in docs[:3]])
    
    prompt = f"""仅根据上下文回答，不编造信息，不知道就说未找到相关菜谱。
上下文：{context}
问题：{request.question}"""
    
    resp = dashscope.Generation.call("qwen-turbo", prompt=prompt, temperature=0.3)
    answer = resp.output.text
    
    # 计算响应时间
    response_time = (datetime.now() - start_time).total_seconds()
    
    # 保存到数据库
    record_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO chat_records 
                 (id, user_id, question, answer, response_time, created_at)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (record_id, request.user_id, request.question, answer, 
               response_time, datetime.now()))
    conn.commit()
    conn.close()
    
    return {"record_id": record_id, "answer": answer, "response_time": response_time}

@app.post("/api/feedback")
async def feedback(request: FeedbackRequest):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''UPDATE chat_records 
                 SET feedback=?, feedback_text=? 
                 WHERE id=?''',
              (request.feedback, request.feedback_text, request.record_id))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.get("/api/stats")
async def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 总提问数
    c.execute("SELECT COUNT(*) FROM chat_records")
    total = c.fetchone()[0]
    
    # 有反馈的数量
    c.execute("SELECT COUNT(*) FROM chat_records WHERE feedback IS NOT NULL")
    feedback_total = c.fetchone()[0]
    
    # 好评率
    c.execute("SELECT COUNT(*) FROM chat_records WHERE feedback=1")
    good_feedback = c.fetchone()[0]
    good_rate = round(good_feedback/feedback_total*100, 2) if feedback_total>0 else 0
    
    # 平均响应时间
    c.execute("SELECT AVG(response_time) FROM chat_records")
    avg_response_time = round(c.fetchone()[0], 2)
    
    conn.close()
    
    return {
        "total_questions": total,
        "feedback_count": feedback_total,
        "good_feedback_rate": f"{good_rate}%",
        "avg_response_time": f"{avg_response_time}s"
    }

# 启动
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
