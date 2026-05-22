# 🍳 中文菜谱 RAG 智能问答系统
基于 **RAG + 混合检索 + 查询重写 + 重排序** 实现的工业级菜谱问答系统，支持公网部署、轻量化运行、高精度问答。

## ✨ 核心功能
- 混合搜索：BM25关键词检索 + FAISS向量检索
- 查询优化：自动重写用户问题，提升检索精度
- 重排序模型：精准筛选最相关菜谱文档
- 网页交互：Gradio可视化界面，开箱即用
- 云端部署：支持 Hugging Face 公网访问

## 🛠️ 技术栈
- 框架：LangChain、Gradio
- 向量库：FAISS
- 大模型：通义千问 Qwen-Turbo
- 嵌入模型：BAAI/bge-small-zh-v1.5
- 语言：Python

## 🚀 快速启动
1. 安装依赖
```bash
pip install -r requirements.txt
