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

# LangChain imports
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()

# ============== CONFIGURATION ==============
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "").strip()
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
DEPLOYMENT = os.getenv("DEPLOYMENT", "prorigo-gpt-4o-mini")
API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-08-01-preview")

if not AZURE_ENDPOINT or not API_KEY:
    raise ValueError("AZURE_ENDPOINT and AZURE_OPENAI_API_KEY must be set in .env")

app = FastAPI(title="Unified Agent with Memory & Tools", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ============== SESSION STORAGE ==============
class SessionStore:
    """
    Stores per‑session:
      - ConversationBufferMemory (LangChain memory)
      - Personal info dict (extracted from user messages)
    """
    def __init__(self):
        self.sessions = {}
    
    def get_or_create(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "memory": ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="output"
                ),
                "user_info": {}
            }
        return self.sessions[session_id]
    
    def get_memory(self, session_id: str) -> ConversationBufferMemory:
        return self.get_or_create(session_id)["memory"]
    
    def get_user_info(self, session_id: str) -> Dict[str, str]:
        return self.get_or_create(session_id)["user_info"]
    
    def update_user_info(self, session_id: str, info: Dict[str, str]):
        self.get_or_create(session_id)["user_info"].update(info)
    
    def get_formatted_user_info(self, session_id: str) -> str:
        info = self.get_user_info(session_id)
        if not info:
            return "None"
        return ", ".join(f"{k}: {v}" for k, v in info.items())
    
    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

session_store = SessionStore()

# ============== TOOL TRACKER (for analytics) ==============
class ToolTracker:
    def __init__(self):
        self._tools_used = {}
    def set_tool(self, session_id: str, tool_name: str):
        self._tools_used[session_id] = tool_name
    def get_tool(self, session_id: str) -> Optional[str]:
        return self._tools_used.get(session_id)
    def reset(self, session_id: str):
        if session_id in self._tools_used:
            del self._tools_used[session_id]

tracker = ToolTracker()

# ============== TOOL IMPLEMENTATIONS ==============
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
        data = requests.get('https://ipinfo.io/json', timeout=5).json()
        city = data.get('city', 'Unknown')
        region = data.get('region', 'Unknown')
        country = data.get('country', 'Unknown')
        return f"📍 Your approximate location: {city}, {region}, {country}"
    except:
        return "📍 Location service unavailable"

# ============== GENERAL CHAT TOOL WITH MEMORY ACCESS ==============
def general_chat(query: str, session_id: str) -> str:
    """
    This tool handles all conversation and personal questions.
    It uses the extracted user info and conversation history.
    """
    user_info = session_store.get_user_info(session_id)
    user_name = user_info.get("name")
    user_location = user_info.get("location")
    
    query_lower = query.lower()
    
    # Direct answers from stored info
    if any(phrase in query_lower for phrase in ["my name", "what is my name", "who am i"]):
        return f"👤 Your name is {user_name}." if user_name else "👤 I don't know your name yet. Could you tell me?"
    
    if any(phrase in query_lower for phrase in ["where am i", "my location", "what is my location"]):
        return f"📍 You mentioned you are in {user_location}." if user_location else "📍 I don't know your location. You can tell me where you are!"
    
    if any(word in query_lower for word in ["hello", "hi", "hey", "greetings"]):
        return f"👋 Hello {user_name}!" if user_name else "👋 Hello! How can I help you today?"
    
    # If user shares personal info, confirm
    if "my name is" in query_lower:
        return "👤 Nice to meet you! I'll remember your name."
    if any(phrase in query_lower for phrase in ["i live in", "i am from", "my city is"]):
        return "📍 Thanks for sharing your location!"
    
    # Fallback – offer help
    if user_name:
        return f"💬 Hi {user_name}! How can I assist you further?"
    return "💬 How can I assist you today?"

# ============== WRAPPERS WITH TRACKING ==============
def clock_wrapper(session_id: str, query: str = ""):
    tracker.set_tool(session_id, "clock")
    return get_current_datetime(query)

def calculator_wrapper(session_id: str, expression: str):
    tracker.set_tool(session_id, "calculator")
    return calculate_math(expression)

def locator_wrapper(session_id: str, query: str = ""):
    tracker.set_tool(session_id, "locator")
    return get_user_location_from_ip(query)

def general_chat_wrapper(session_id: str, query: str):
    tracker.set_tool(session_id, "general_chat")
    return general_chat(query, session_id)

# ============== REACT AGENT FACTORY ==============
def create_agent(session_id: str):
    """Create a LangChain ReAct agent with memory and tools."""
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
    
    # Tools
    tools = [
        Tool(
            name="clock",
            func=lambda q: clock_wrapper(session_id, q),
            description="Get the current date and time. Use when asked about the time."
        ),
        Tool(
            name="calculator",
            func=lambda expr: calculator_wrapper(session_id, expr),
            description="Perform mathematical calculations. Use for arithmetic operations."
        ),
        Tool(
            name="locator",
            func=lambda q: locator_wrapper(session_id, q),
            description="Get the user's approximate location from IP address."
        ),
        Tool(
            name="general_chat",
            func=lambda q: general_chat_wrapper(session_id, q),
            description="""Handle conversation, greetings, and personal information.
            ALWAYS use this tool for:
            - Questions about the user (name, location)
            - Greetings and casual conversation
            - When the user shares personal info
            - When no other tool is appropriate"""
        )
    ]
    
    # Memory
    memory = session_store.get_memory(session_id)
    
    # Prompt – includes user info and conversation history
    user_info_str = session_store.get_formatted_user_info(session_id)
    
    prompt = PromptTemplate.from_template("""
You are a helpful assistant with access to conversation history and user information.
You have the following tools available:

{tools}

Use the following format:
Question: the input question you must answer
Thought: you should think step by step
Action: the action to take, one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Observation if needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

User information I know: {user_info}

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}
""")
    
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate"
    )
    return executor, user_info_str

# ============== PERSONAL INFO EXTRACTION ==============
def extract_personal_info(message: str) -> Dict[str, str]:
    """Extract name and location using flexible regex."""
    info = {}
    patterns = [
        (r'(?:my name is|i am|call me|name\'s?)\s+(\w+)', 'name'),
        (r'(?:i live in|from|city is)\s+(\w+)', 'location'),
    ]
    for pattern, key in patterns:
        match = re.search(pattern, message.lower())
        if match:
            value = match.group(1).strip().title()
            info[key] = value
    return info

# ============== STREAMING HELPER ==============
async def stream_text(text: str):
    """Simulate streaming by yielding characters with realistic timing."""
    words = text.split(' ')
    for i, word in enumerate(words):
        for char in word:
            yield char
            await asyncio.sleep(0.03)
        if i < len(words) - 1:
            yield ' '
            await asyncio.sleep(0.05)

# ============== REQUEST / RESPONSE MODELS ==============
class AgentID(str, Enum):
    GENERAL = "general_agent"

class ChatRequest(BaseModel):
    message: str
    agent_id: AgentID = AgentID.GENERAL   # for consistency, only one agent now
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
    tool_display: Optional[str] = None

# ============== API ENDPOINTS ==============
@app.post("/chat")
async def chat(request: ChatRequest):
    start_time = time.time()
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
    
    # Reset tool tracker for this session
    tracker.reset(session_id)
    
    # Extract personal info and store
    extracted = extract_personal_info(request.message)
    if extracted:
        session_store.update_user_info(session_id, extracted)
    
    # Add user message to memory (LangChain)
    memory = session_store.get_memory(session_id)
    memory.chat_memory.add_user_message(request.message)
    
    try:
        # Create agent with current memory
        agent_executor, user_info_str = create_agent(session_id)
        
        # Prepare input
        agent_input = {
            "input": request.message,
            "user_info": user_info_str,
            "tool_names": ", ".join([tool.name for tool in agent_executor.tools]),
            "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in agent_executor.tools]),
            "chat_history": memory.load_memory_variables({}).get("chat_history", "")
        }
        
        # Invoke agent
        response = await agent_executor.ainvoke(agent_input)
        agent_response = response.get("output", str(response))
        
        # Track which tool was used
        tool_used = tracker.get_tool(session_id)
        
    except Exception as e:
        agent_response = f"I encountered an error: {str(e)}"
        tool_used = None
    
    # Add assistant response to memory
    memory.chat_memory.add_ai_message(agent_response)
    
    # Build response metadata
    history = memory.chat_memory.messages
    user_messages = sum(1 for m in history if isinstance(m, HumanMessage))
    assistant_messages = sum(1 for m in history if isinstance(m, AIMessage))
    stats = {
        "total_messages": len(history),
        "user_messages": user_messages,
        "assistant_messages": assistant_messages,
        "user_info_available": bool(session_store.get_user_info(session_id)),
        "user_name": session_store.get_user_info(session_id).get("name")
    }
    
    # If streaming requested, return StreamingResponse
    if request.stream:
        return StreamingResponse(
            stream_text(agent_response),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
    else:
        # Return JSON
        return ChatResponse(
            response=agent_response,
            session_id=session_id,
            message_history=[{"role": type(m).__name__, "content": m.content} for m in history[-10:]],
            conversation_stats=stats,
            processing_time=f"{time.time() - start_time:.3f}s",
            tool_used=tool_used,
            tool_display=tool_used.capitalize() if tool_used else None
        )

@app.get("/chat/sessions/{session_id}")
async def get_history(session_id: str, limit: int = 20):
    memory = session_store.get_memory(session_id)
    messages = memory.chat_memory.messages[-limit:]
    return [{"role": type(m).__name__, "content": m.content, "timestamp": datetime.now().isoformat()} for m in messages]

@app.get("/chat/memory/{session_id}")
async def get_memory_info(session_id: str):
    user_info = session_store.get_user_info(session_id)
    memory = session_store.get_memory(session_id)
    messages = memory.chat_memory.messages[-5:]
    return {
        "session_id": session_id,
        "user_info": user_info,
        "recent_messages": [{"role": type(m).__name__, "content": m.content} for m in messages],
        "has_memory": bool(user_info) or len(messages) > 0
    }

@app.delete("/chat/sessions/{session_id}")
async def clear_session(session_id: str):
    session_store.clear_session(session_id)
    tracker.reset(session_id)
    return {"success": True, "message": f"Session {session_id} cleared"}

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0", "active_sessions": len(session_store.sessions)}

@app.get("/")
async def root():
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