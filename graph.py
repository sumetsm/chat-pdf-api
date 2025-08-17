import os
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.utilities import SearxSearchWrapper

import torch
import requests
import json
import logging
import time
from typing import TypedDict, Optional
from langchain.llms import Ollama
from config import Config

# ----------------- Logging -----------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# โหลดค่า API key จาก environment
api_key = os.getenv("GROQ_API_KEY")

# # ----------------- LLM CONFIG -----------------
# llm = Ollama(model="qwen3:8b", temperature=0)
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    api_key=Config.GROQ_API_KEY
)

print("App running with API key:", bool(Config.GROQ_API_KEY))

def safe_invoke(prompt: str, retries: int = 3, delay: float = 2.0) -> str:
    """Call LLM safely with retry and extract text from AIMessage."""
    for attempt in range(retries):
        try:
            # LLM อาจคืนค่า list ของ AIMessage
            resp = llm.invoke([HumanMessage(content=prompt)])
            if isinstance(resp, list):
                # รวมข้อความจากแต่ละ AIMessage
                return " ".join([m.content for m in resp if hasattr(m, "content")])
            elif hasattr(resp, "content"):
                return resp.content
            else:
                return str(resp)
        except Exception as e:
            logger.warning(f"LLM invoke attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    return "I apologize, but I encountered an error while generating a response."


# ----------------- Embedding Model -----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

# ----------------- Vector Search Tool -----------------
@tool
def vector_search(query: str) -> tuple[str, float]:
    """Searches vector DB for relevant documents."""
    db = Chroma(
        collection_name="pdf_docs",
        persist_directory="./chroma_db",
        embedding_function=embed_model
    )
    # docs = db.similarity_search(query, k=2)
    docs_with_scores = db.similarity_search_with_score(query, k=5)
    if docs_with_scores:
        context = "\n\n".join(
            [f"Document: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc, _ in docs_with_scores]
    )
    score = docs_with_scores[0][1]  # เอา similarity score จริง
    return context, score


# ----------------- Web Search Tool -----------------
# กำหนดตัวแปร search ไว้ล่วงหน้าเพื่อใช้ภายในฟังก์ชัน
search = SearxSearchWrapper(searx_host="http://searxng:8080/")

@tool
def web_search(query: str) -> str:
    """Performs a Searx search and returns result."""
    try:
        results = search.run(query)
        print(f"web result : {results}")
        return results
    except Exception as e:
        print(f"Web search failed: {e}")
        return f"[Web search failed: {e}]"

# ----------------- Workflow State -----------------
class WorkflowState(TypedDict):
    question: str
    clarified_question: Optional[str]
    keyword: Optional[str]           # <-- เพิ่มตรงนี้
    pdf_context: Optional[str]
    vector_score: Optional[float]
    web_context: Optional[str]
    full_context: Optional[str]
    final_answer: Optional[str]
    chat_history: list[dict]  # [{"question": ..., "answer": ...}, ...]


# ----------------- Agent Functions -----------------
def receive_question(state: WorkflowState) -> WorkflowState:
    logger.info(f"Received question: {state['question']}")
    return state

def clarification_agent(state: WorkflowState) -> WorkflowState:
    question = state["question"]
    chat_history = state.get("chat_history", [])

    prompt = PromptTemplate.from_template(
        """You are a context-aware clarification agent and keyword extractor.
Analyze the following question considering the previous conversation, then produce a clear question (if needed) and the most relevant keywords for searching PDF documents or web resources.

Previous conversation (chat history):
{chat_history}

Current question:
{question}

Instructions:
1. If the question is clear, respond with: CLEAR
2. If the question is vague, ambiguous, or uses pronouns referring to previous messages, rewrite it to make it clear.
3. Identify 3-7 keywords from the clarified question and previous chat history that are most relevant for search. Focus on nouns, proper nouns, and technical terms.

Respond strictly in JSON format:
{{
  "clarified_question": "<your clarified question or CLEAR>",
  "keyword": "<comma-separated keywords>"
}}

Your response:"""
    )

    try:
        response = safe_invoke(prompt.format(
            question=question,
            chat_history=json.dumps(chat_history, ensure_ascii=False, indent=2)
        ))
        data = json.loads(response)
        clarified = data.get("clarified_question", question)
        if clarified == "CLEAR":
            clarified = question
        keyword = data.get("keyword", "")
        logger.info(f"Question clarified: {clarified}, Keyword: {keyword}")
        return {**state, "clarified_question": clarified, "keyword": keyword}
    except Exception as e:
        logger.error(f"Clarification failed: {str(e)}")
        return {**state, "clarified_question": question, "keyword": ""}



def retrieve_pdf_context(state: WorkflowState) -> WorkflowState:
    query = state.get("keyword")
    logger.info(f"retrieve_pdf_context query is {query}")
    pdf_context, score = vector_search(query)
    logger.info(f"PDF context retrieved with score {score}")
    if score >= 0.4 :
        return {**state, "pdf_context": pdf_context, "vector_score": score}
    else :
        return {**state, "pdf_context": "", "vector_score": score}

def routing_agent(state: WorkflowState) -> WorkflowState:
    """
    ใช้ LLM ประเมินว่า PDF context เพียงพอสำหรับตอบคำถามหรือไม่
    - ถ้าเพียงพอ → เก็บ PDF context และ web_context อาจจะว่าง
    - ถ้าไม่เพียงพอหรือไม่เกี่ยวข้อง → เรียก web_search
    """
    pdf_context = state.get("pdf_context", "")
    question = state.get("clarified_question", state["question"])
    web_context = state.get("web_context", "")
    query = state.get("keyword")

    if pdf_context:
        # LLM ช่วยประเมินความเพียงพอ
        prompt = f"""
You are an assistant that determines if the following PDF context can be used to answer the question.

PDF Context:
{pdf_context}

Question:
{question}

Instructions:
- Respond with "YES" if the PDF context contains relevant and sufficient information to answer the question.
- Respond with "NO" if the PDF context is insufficient or irrelevant.
"""
        result = safe_invoke(prompt)
        logger.info(f"PDF relevance/sufficiency check result: {result}")

        if "YES" in result.upper():
            logger.info("PDF context sufficient and relevant, keeping PDF context")
            return {**state, "web_context": ""}  # เก็บ PDF context ไว้, web_context ไม่ต้องสร้างใหม่
        else:
            logger.info("PDF context insufficient or irrelevant, performing web search")
            web_context = web_search(query)
            return {**state, "web_context": web_context, "pdf_context": ""}

    else:
        # ไม่มี PDF context → ต้องไป web search
        logger.info("No PDF context found, performing web search")
        web_context = web_search(query)
        return {**state, "web_context": web_context, "pdf_context": ""}

def merge_context(state: WorkflowState) -> WorkflowState:
    pdf_context = state.get("pdf_context")
    web_context = state.get("web_context")

    parts = []
    if pdf_context:
        parts.append(f"**PDF Documents:**\n{pdf_context}")
    if web_context:
        parts.append(f"**Web Search Results:**\n{web_context}")

    full_context = "\n\n" + ("="*50 + "\n\n").join(parts) if parts else ""
    logger.info(
        f"Context merged - PDF: {'Yes' if pdf_context else 'No'}, "
        f"Web: {'Yes' if web_context else 'No'}"
    )
    return {**state, "full_context": full_context}

def answer_generation_agent(state: WorkflowState) -> WorkflowState:
    question = state.get("clarified_question", state["question"])
    clarified = state.get("clarified_question", state["clarified_question"])
    context = state.get("full_context", "")
    chat_history = state.get("chat_history", [])

    if context.strip():
        prompt = f"""Based on the following context and previous conversation, provide a comprehensive answer.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, well-structured answer
- Cite specific information from the context when possible
- If the context doesn't fully answer the question, say so explicitly
- Be concise but thorough

Answer:"""
    else:
        prompt = f"""The user asked: {question}

No relevant information was found in the PDF documents or web search.

Please provide a helpful response indicating the lack of information and suggest alternatives.

Answer:"""
    logger.info("Prompt is HERE !!", prompt)
    answer = safe_invoke(prompt)
    
    # เก็บใน chat_history
    chat_history.append({"question": question, "answer": answer})
    
    logger.info("Answer generated successfully with chat history updated")
    logger.info("chat-history : 123 ",json.dumps(chat_history, ensure_ascii=False, indent=2) ,"321 thia is all")
    logger.info(f"clarification : {clarified} !!")

    return {**state, "final_answer": answer, "chat_history": chat_history}


# ----------------- Build Workflow Graph -----------------
def create_workflow():
    graph = StateGraph(WorkflowState)
    graph.add_node("receive_question", receive_question)
    graph.add_node("clarification_agent", clarification_agent)
    graph.add_node("retrieve_pdf_context", retrieve_pdf_context)
    graph.add_node("routing_agent", routing_agent)
    graph.add_node("merge_context", merge_context)
    graph.add_node("answer_generation_agent", answer_generation_agent)
    graph.set_entry_point("receive_question")
    graph.add_edge("receive_question", "clarification_agent")
    graph.add_edge("clarification_agent", "retrieve_pdf_context")
    graph.add_edge("retrieve_pdf_context", "routing_agent")
    graph.add_edge("routing_agent", "merge_context")
    graph.add_edge("merge_context", "answer_generation_agent")
    graph.add_edge("answer_generation_agent", END)
    workflow = graph.compile()
    logger.info("Workflow created successfully")
    return workflow
