# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
import time
import os

from agent import UnifiedAgent

app = FastAPI(title="Unified Agent with Memory & Tools", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

agent = UnifiedAgent()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: bool = True

class ChatResponse(BaseModel):
    success: bool = True
    response: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    message_history: List[Dict[str, Any]]
    conversation_stats: Dict[str, Any]
    processing_time: str
    tool_used: Optional[str] = None

@app.post("/chat")
async def chat(request: ChatRequest):
    start_time = time.time()
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
    result = await agent.process_message(session_id, request.message, stream=request.stream)
    if request.stream:
        return StreamingResponse(result, media_type="text/plain", headers={"Cache-Control": "no-cache"})
    else:
        tool_used = agent.get_tool_used(session_id)
        info = agent.get_session_info(session_id)
        stats = {
            "total_messages": len(info.get("recent_messages", [])),
            "user_info_available": bool(info.get("user_info")),
            "user_name": info.get("user_info", {}).get("name")
        }
        return ChatResponse(
            response=result,
            session_id=session_id,
            message_history=info.get("recent_messages", []),
            conversation_stats=stats,
            processing_time=f"{time.time() - start_time:.3f}s",
            tool_used=tool_used
        )

@app.get("/chat/sessions/{session_id}")
async def get_history(session_id: str, limit: int = 20):
    info = agent.get_session_info(session_id)
    return info.get("recent_messages", [])[-limit:]

@app.get("/chat/memory/{session_id}")
async def get_memory(session_id: str):
    return agent.get_session_info(session_id)

@app.delete("/chat/sessions/{session_id}")
async def clear_session(session_id: str):
    if agent.clear_session(session_id):
        return {"success": True, "message": f"Session {session_id} cleared"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0", "active_sessions": len(agent.sessions)}

# Serve the chat UI at the root
@app.get("/", response_class=HTMLResponse)
async def get_index():
    return FileResponse("templates/index.html")

# (Optional) If you want a JSON info endpoint, keep it under /api/info
@app.get("/api/info")
async def api_info():
    return {
        "message": "Unified Agent with Memory & Tools",
        "version": "1.0.0",
        "endpoints": {
            "POST /chat": "Chat with streaming",
            "GET /chat/sessions/{id}": "View history",
            "GET /chat/memory/{id}": "Inspect memory",
            "DELETE /chat/sessions/{id}": "Clear session",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))