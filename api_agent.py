from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid, time, os, asyncio, re, requests
from enum import Enum
import warnings
warnings.filterwarnings("ignore")

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

# ============== CONFIGURATION ==============
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "").strip()
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
DEPLOYMENT = os.getenv("DEPLOYMENT", "prorigo-gpt-4o-mini")
API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-08-01-preview")

if not AZURE_ENDPOINT or not API_KEY:
    raise ValueError("AZURE_ENDPOINT and AZURE_OPENAI_API_KEY must be set in .env")

app = FastAPI(title="AI Agent API with Working Memory", version="10.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ============== SESSION MEMORY ==============
class SessionMemory:
    def __init__(self):
        self.sessions = {}
    
    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {"history": [], "user_info": {}}
        self.sessions[session_id]["history"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        if role == "user":
            self._extract_user_info(session_id, content)
    
    def _extract_user_info(self, session_id: str, content: str):
        # Improved regex as in agent.py
        patterns = [
            (r'(?:my name is|i am|call me|name\'s?)\s+(\w+)', 'name'),
            (r'(?:i live in|from|city is)\s+(\w+)', 'location'),
        ]
        for pattern, info_type in patterns:
            match = re.search(pattern, content.lower())
            if match:
                value = match.group(1).strip().title()
                self.sessions[session_id]["user_info"][info_type] = value
    
    def get_user_info(self, session_id: str) -> Dict[str, str]:
        return self.sessions.get(session_id, {}).get("user_info", {}).copy()
    
    def get_name(self, session_id: str) -> Optional[str]:
        return self.get_user_info(session_id).get("name")
    
    def get_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        return self.sessions.get(session_id, {}).get("history", [])[-limit:]
    
    def get_formatted_history(self, session_id: str) -> str:
        history = self.get_history(session_id, 5)
        return "\n".join(f"{'Human' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in history)
    
    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

session_memory = SessionMemory()

# ============== GLOBAL TOOL TRACKER ==============
class GlobalToolTracker:
    _instance = None
    _session_tools = {}
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    @classmethod
    def set_tool_used(cls, session_id: str, tool_name: str):
        cls._session_tools[session_id] = tool_name
    @classmethod
    def get_tool_used(cls, session_id: str) -> Optional[str]:
        return cls._session_tools.get(session_id)
    @classmethod
    def reset(cls, session_id: str):
        if session_id in cls._session_tools:
            del cls._session_tools[session_id]

tracker = GlobalToolTracker()

# ============== TOOLS ==============
def get_current_datetime(query=""):
    now = datetime.now()
    return f"⏰ Current time is {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}"

def calculate_math(expression: str):
    try:
        expression = expression.strip()
        allowed = set('0123456789+-*/(). ')
        if all(c in allowed for c in expression):
            result = eval(expression, {"__builtins__": {}}, {})
            return f"🧮 Calculation: {expression} = {result}"
        return "❌ Invalid calculation expression"
    except Exception as e:
        return f"❌ Calculation error: {str(e)}"

def get_user_location_from_ip(query=""):
    try:
        data = requests.get('https://ipinfo.io/json').json()
        city = data.get('city', 'Unknown')
        region = data.get('region', 'Unknown')
        country = data.get('country', 'Unknown')
        return f"📍 Your approximate location: {city}, {region}, {country}"
    except:
        return "📍 Location service unavailable"

def handle_general_chat(query: str, session_id: str):
    user_info = session_memory.get_user_info(session_id)
    user_name = user_info.get("name")
    
    if any(phrase in query.lower() for phrase in ["my name", "what is my name", "who am i"]):
        return f"👤 Your name is {user_name}." if user_name else "👤 I don't know your name yet. Could you tell me?"
    if any(phrase in query.lower() for phrase in ["where am i", "my location", "what is my location"]):
        location = user_info.get("location")
        return f"📍 You mentioned you are in {location}." if location else "📍 I don't know your location. You can tell me where you are!"
    if any(word in query.lower() for word in ["hello", "hi", "hey"]):
        return f"👋 Hello {user_name}!" if user_name else "👋 Hello! How can I help you today?"
    if "my name is" in query.lower():
        return "👤 Nice to meet you! I'll remember your name."
    if any(phrase in query.lower() for phrase in ["i live in", "i am from", "my city is"]):
        return "📍 Thanks for sharing your location!"
    return f"💬 Hi {user_name}!" if user_name else "💬 How can I assist you today?"

def get_datetime_with_tracking(session_id: str, query=""):
    tracker.set_tool_used(session_id, "clock")
    return get_current_datetime(query)

def calculate_with_tracking(session_id: str, expression: str):
    tracker.set_tool_used(session_id, "calculator")
    return calculate_math(expression)

def get_location_with_tracking(session_id: str, query=""):
    tracker.set_tool_used(session_id, "locator")
    return get_user_location_from_ip(query)

def general_chat_with_tracking(session_id: str, query: str):
    tracker.set_tool_used(session_id, "general_chat")
    return handle_general_chat(query, session_id)

# ============== MODELS ==============
class AgentID(str, Enum):
    GENERAL = "general_agent"
    MATH = "math_agent"
    TIME = "time_agent"

class ChatRequest(BaseModel):
    message: str
    agent_id: AgentID = AgentID.GENERAL
    session_id: Optional[str] = None
    stream: bool = True   # added streaming flag

class ChatResponse(BaseModel):
    success: bool = True
    response: str
    agent_id: str
    agent_name: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    message_history: List[dict]
    conversation_stats: dict
    processing_time: str
    tool_used: Optional[str] = None
    tool_used_display: Optional[str] = None

# ============== AGENT CREATION ==============
def create_working_agent(agent_id: str, session_id: str):
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=API_KEY,
        azure_deployment=DEPLOYMENT,
        api_version=API_VERSION,
        temperature=0.1,
        max_tokens=300,
        timeout=30,
        max_retries=2
    )
    user_info = session_memory.get_user_info(session_id)
    user_info_str = ", ".join([f"{k}: {v}" for k, v in user_info.items()]) if user_info else "None"
    conversation_context = session_memory.get_formatted_history(session_id) or "No previous conversation."
    
    tools = []
    if agent_id == AgentID.GENERAL.value:
        tools = [
            Tool(name="clock", func=lambda q: get_datetime_with_tracking(session_id, q),
                 description="Get current date and time."),
            Tool(name="calculator", func=lambda expr: calculate_with_tracking(session_id, expr),
                 description="Perform math calculations."),
            Tool(name="locator", func=lambda q: get_location_with_tracking(session_id, q),
                 description="Get user's location from IP."),
            Tool(name="general_chat", func=lambda q: general_chat_with_tracking(session_id, q),
                 description="Conversation and personal info.")
        ]
    elif agent_id == AgentID.MATH.value:
        tools = [
            Tool(name="calculator", func=lambda expr: calculate_with_tracking(session_id, expr),
                 description="Perform math calculations."),
            Tool(name="general_chat", func=lambda q: general_chat_with_tracking(session_id, q),
                 description="Conversation.")
        ]
    else:  # TIME
        tools = [
            Tool(name="clock", func=lambda q: get_datetime_with_tracking(session_id, q),
                 description="Get current date and time."),
            Tool(name="general_chat", func=lambda q: general_chat_with_tracking(session_id, q),
                 description="Conversation.")
        ]
    
    prompt = PromptTemplate.from_template("""
You are a helpful assistant with tools.
Tools: {tools}
Use the format:
Question: ...
Thought: ...
Action: one of [{tool_names}]
Action Input: ...
Observation: ...
(Repeat if needed)
Thought: I now know the final answer
Final Answer: ...

User info I know: {user_info}
Conversation context: {conversation_context}
Question: {input}
Thought: {agent_scratchpad}
""")
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True,
                             max_iterations=3, early_stopping_method="generate")
    return executor, user_info_str, conversation_context

# ============== STREAMING HELPER ==============
async def stream_text(text: str):
    """Simulate streaming by yielding characters."""
    words = text.split(' ')
    for i, word in enumerate(words):
        for char in word:
            yield char
            await asyncio.sleep(0.03)
        if i < len(words) - 1:
            yield ' '
            await asyncio.sleep(0.05)

# ============== ENDPOINTS ==============
@app.post("/chat")
async def chat(request: ChatRequest):
    start = time.time()
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
    tracker.reset(session_id)
    session_memory.add_message(session_id, "user", request.message)
    
    try:
        executor, user_info_str, context = create_working_agent(request.agent_id, session_id)
        agent_input = {
            "input": request.message,
            "user_info": user_info_str,
            "conversation_context": context,
            "tool_names": ", ".join([t.name for t in executor.tools]),
            "tools": "\n".join([f"{t.name}: {t.description}" for t in executor.tools])
        }
        response = await executor.ainvoke(agent_input)
        agent_response = response.get("output", str(response))
        tool_used = tracker.get_tool_used(session_id)
    except Exception as e:
        agent_response = f"I encountered an error: {str(e)}"
        tool_used = None
    
    session_memory.add_message(session_id, "assistant", agent_response)
    
    if request.stream:
        return StreamingResponse(stream_text(agent_response), media_type="text/plain")
    else:
        # Return JSON response
        history = session_memory.get_history(session_id, 10)
        stats = {
            "total_messages": len(history),
            "user_messages": sum(1 for m in history if m['role']=='user'),
            "assistant_messages": sum(1 for m in history if m['role']=='assistant'),
            "user_info_available": bool(session_memory.get_user_info(session_id)),
            "user_name": session_memory.get_name(session_id)
        }
        return ChatResponse(
            response=agent_response,
            agent_id=request.agent_id,
            agent_name=request.agent_id.replace('_', ' ').title(),
            session_id=session_id,
            message_history=history,
            conversation_stats=stats,
            processing_time=f"{time.time()-start:.3f}s",
            tool_used=tool_used,
            tool_used_display=tool_used.capitalize() if tool_used else None
        )

@app.get("/chat/sessions/{session_id}")
async def get_history(session_id: str, limit: int = 20):
    history = session_memory.get_history(session_id, limit)
    if not history:
        raise HTTPException(404, "Session not found")
    return history

@app.get("/chat/memory/{session_id}")
async def get_memory(session_id: str):
    return {
        "session_id": session_id,
        "user_info": session_memory.get_user_info(session_id),
        "recent_history": session_memory.get_history(session_id, 5),
        "has_memory": bool(session_memory.get_user_info(session_id) or session_memory.get_history(session_id))
    }

@app.delete("/chat/sessions/{session_id}")
async def clear_session(session_id: str):
    session_memory.clear_session(session_id)
    tracker.reset(session_id)
    return {"success": True, "message": f"Session {session_id} cleared"}

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "10.1.0", "active_sessions": len(session_memory.sessions)}

@app.get("/")
async def root():
    return {"message": "AI Agent with Working Memory", "version": "10.1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))