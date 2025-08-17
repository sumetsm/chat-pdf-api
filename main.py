from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import os
import json
from graph import create_workflow, WorkflowState
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chat With PDF API", version="1.0.0")

# Directory to store session JSON files
SESSION_DIR = "./sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"

class QuestionResponse(BaseModel):
    answer: str
    session_id: str
    context_used: str

class ClearMemoryRequest(BaseModel):
    session_id: str = "default"

def load_session(session_id: str) -> Dict[str, Any]:
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"chat_history": []}

def save_session(session_id: str, session_data: Dict[str, Any]):
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

# In-memory workflows per session
workflows: Dict[str, Any] = {}

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        # Load or create workflow
        if request.session_id not in workflows:
            workflows[request.session_id] = create_workflow()
            logger.info(f"Created new workflow for session: {request.session_id}")

        workflow = workflows[request.session_id]

        # Load session history
        session_data = load_session(request.session_id)
        chat_history = session_data.get("chat_history", [])
        logger.info(f"this is chat_history before run grahp : {chat_history}")

        # Prepare input state
        input_state: WorkflowState = {
            "question": request.question,
            "clarified_question": None,
            "pdf_context": None,
            "web_context": None,
            "full_context": None,
            "final_answer": None,
            "chat_history": chat_history  # pass previous chat history
        }

        logger.info(f"Processing question: {request.question}")

        # Run the workflow
        result = workflow.invoke(input_state)

        # Update chat history
        chat_history.append({
            "question": request.question,
            "answer": result.get("final_answer", "No answer generated")
        })
        session_data["chat_history"] = chat_history
        save_session(request.session_id, session_data)

        # Prepare context info
        context_info = []
        if result.get("pdf_context"):
            context_info.append("PDF")
        if result.get("web_context"):
            context_info.append("Web")
        context_used = ", ".join(context_info) if context_info else "None"

        return QuestionResponse(
            answer=result.get("final_answer", "No answer generated"),
            session_id=request.session_id,
            context_used=context_used
        )

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/clear-memory")
async def clear_memory(request: ClearMemoryRequest):
    try:
        # Remove workflow
        if request.session_id in workflows:
            del workflows[request.session_id]
        # Remove session file
        session_file = os.path.join(SESSION_DIR, f"{request.session_id}.json")
        if os.path.exists(session_file):
            os.remove(session_file)
            logger.info(f"Cleared memory for session: {request.session_id}")
            return {"message": f"Memory cleared for session {request.session_id}"}
        else:
            return {"message": f"No active session found for {request.session_id}"}
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing memory: {str(e)}")

@app.get("/")
async def root():
    active_sessions = len([f for f in os.listdir(SESSION_DIR) if f.endswith(".json")])
    return {
        "message": "Chat With PDF API is running",
        "docs": "/docs",
        "active_sessions": active_sessions
    }

@app.get("/health")
async def health_check():
    active_sessions = len([f for f in os.listdir(SESSION_DIR) if f.endswith(".json")])
    return {"status": "healthy", "active_sessions": active_sessions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
