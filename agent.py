import os
import re
import asyncio
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()

class UnifiedAgent:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        print(f"🔍 Using Groq model: {self.model_name}")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY missing in .env")
        self.sessions = {}
        self.tool_tracker = {}
    
    def _get_or_create_session(self, session_id: str) -> dict:
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
        return self._get_or_create_session(session_id)["memory"]
    
    def get_user_info(self, session_id: str) -> Dict[str, str]:
        return self._get_or_create_session(session_id)["user_info"]
    
    def update_user_info(self, session_id: str, info: Dict[str, str]):
        self._get_or_create_session(session_id)["user_info"].update(info)
    
    def get_formatted_user_info(self, session_id: str) -> str:
        info = self.get_user_info(session_id)
        if not info:
            return "None"
        return ", ".join(f"{k}: {v}" for k, v in info.items())
    
    def extract_personal_info(self, message: str) -> Dict[str, str]:
        info = {}
        patterns = [
            (r'(?:my name is|i am|call me|name\'s?)\s+(\w+)', 'name'),
            (r'(?:i live in|from|city is)\s+(\w+)', 'location'),
        ]
        for pattern, key in patterns:
            match = re.search(pattern, message.lower())
            if match:
                info[key] = match.group(1).strip().title()
        return info
    
    def _get_tools(self, session_id: str) -> List[Tool]:
        def clock_wrapper(query=""):
            self.tool_tracker[session_id] = "clock"
            now = datetime.now()
            return f"⏰ Current time is {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}"
        
        def calculator_wrapper(expression: str):
            self.tool_tracker[session_id] = "calculator"
            try:
                expression = expression.strip()
                allowed = set('0123456789+-*/(). ')
                if all(c in allowed for c in expression):
                    result = eval(expression, {"__builtins__": {}}, {})
                    return f"🧮 Calculation: {expression} = {result}"
                return "❌ Invalid expression"
            except Exception as e:
                return f"❌ Calculation error: {str(e)}"
        
        def locator_wrapper(query=""):
            self.tool_tracker[session_id] = "locator"
            try:
                data = requests.get('https://ipinfo.io/json', timeout=5).json()
                city = data.get('city', 'Unknown')
                region = data.get('region', 'Unknown')
                country = data.get('country', 'Unknown')
                return f"📍 Your approximate location: {city}, {region}, {country}"
            except:
                return "📍 Location service unavailable"
        
        def general_chat_wrapper(query: str):
            self.tool_tracker[session_id] = "general_chat"
            user_info = self.get_user_info(session_id)
            user_name = user_info.get("name")
            user_location = user_info.get("location")
            q = query.lower()
            
            if any(phrase in q for phrase in ["my name", "what is my name", "who am i"]):
                return f"👤 Your name is {user_name}." if user_name else "👤 I don't know your name yet."
            if any(phrase in q for phrase in ["where am i", "my location", "what is my location"]):
                return f"📍 You mentioned you are in {user_location}." if user_location else "📍 I don't know your location yet."
            if any(word in q for word in ["hello", "hi", "hey"]):
                return f"👋 Hello {user_name}!" if user_name else "👋 Hello!"
            if "my name is" in q:
                return "👤 Nice to meet you! I'll remember your name."
            if any(phrase in q for phrase in ["i live in", "i am from", "my city is"]):
                return "📍 Thanks for sharing your location!"
            return f"💬 Hi {user_name}!" if user_name else "💬 How can I assist you today?"
        
        return [
            Tool(name="clock", func=clock_wrapper, description="Get current date and time."),
            Tool(name="calculator", func=calculator_wrapper, description="Perform mathematical calculations."),
            Tool(name="locator", func=locator_wrapper, description="Get user's location from IP."),
            Tool(name="general_chat", func=general_chat_wrapper, description="Handle conversation and personal info.")
        ]
    
    def _create_agent(self, session_id: str):
        llm = ChatGroq(
            model=self.model_name,
            api_key=self.api_key,
            temperature=0.1,
            max_tokens=300,
            timeout=30
        )
        tools = self._get_tools(session_id)
        memory = self.get_memory(session_id)
        user_info_str = self.get_formatted_user_info(session_id)
        
        memory_vars = memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", "")
        
        prompt_template = """
You are a friendly, helpful assistant with access to conversation history and user information.
You have the following tools available:

{tools}

Use the following format. **Be decisive** – answer directly if you know, or use a tool only if necessary.

Question: the input question you must answer
Thought: you should think step by step
Action: the action to take, one of [{tool_names}] (if needed)
Action Input: the input to the action (if needed)
Observation: the result of the action (if needed)
... (repeat Thought/Action/Observation if needed, but try to finish quickly)
Thought: I now know the final answer
Final Answer: the final answer to the original question

**Critical Instructions:**
- If the user asks for their name or location, **answer directly** from the user information I know – **do not use any tool**.
- If you know the answer from your own knowledge or the conversation history, answer directly without using tools.
- Always respond in **complete, friendly sentences**.

User information I know: {user_info}

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}
"""
        prompt = PromptTemplate.from_template(prompt_template)
        prompt = prompt.partial(user_info=user_info_str, chat_history=chat_history)
        
        agent = create_react_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=10
        )
        return executor
    
    async def process_message(self, session_id: str, message: str, stream: bool = True):
        extracted = self.extract_personal_info(message)
        if extracted:
            self.update_user_info(session_id, extracted)
        
        self.tool_tracker[session_id] = None
        
        try:
            agent = self._create_agent(session_id)
            response = await agent.ainvoke({"input": message})
            agent_response = response.get("output", str(response))
        except Exception as e:
            print(f"🔴 ERROR: {e}")
            agent_response = f"I encountered an error: {str(e)}"
        
        # 🔥 Check if the agent hit the iteration limit – override with direct memory answer
        if "Agent stopped due to iteration limit" in agent_response:
            user_info = self.get_user_info(session_id)
            if "my name" in message.lower() and user_info.get("name"):
                agent_response = f"Your name is {user_info['name']}."
            elif "location" in message.lower() and user_info.get("location"):
                agent_response = f"You are in {user_info['location']}."
            else:
                # If we can't answer from memory, give a generic helpful response
                agent_response = "I'm sorry, I had trouble processing that. Could you please rephrase?"
        
        if stream:
            async def generate():
                words = agent_response.split(' ')
                for i, word in enumerate(words):
                    for char in word:
                        yield char
                        await asyncio.sleep(0.03)
                    if i < len(words) - 1:
                        yield ' '
                        await asyncio.sleep(0.05)
            return generate()
        else:
            return agent_response
    
    def get_session_info(self, session_id: str) -> dict:
        memory = self.get_memory(session_id)
        user_info = self.get_user_info(session_id)
        messages = memory.chat_memory.messages[-10:]
        return {
            "user_info": user_info,
            "recent_messages": [{"role": type(m).__name__, "content": m.content} for m in messages],
            "has_memory": bool(user_info) or len(messages) > 0
        }
    
    def clear_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            if session_id in self.tool_tracker:
                del self.tool_tracker[session_id]
            return True
        return False

    def get_tool_used(self, session_id: str) -> Optional[str]:
        return self.tool_tracker.get(session_id)