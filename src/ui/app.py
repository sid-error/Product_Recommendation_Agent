import streamlit as st
import asyncio
import os
import sys
import queue
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from src.agents.workflow import create_sales_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

st.set_page_config(page_title="Sales Assistant", page_icon="🛍️", layout="centered")
st.title("🛍️ Sales Assistant")

# ── Background event loop (one per process) ─────────────────────────────────
@st.cache_resource
def get_async_loop():
    import threading
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop

_bg_loop = get_async_loop()

def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _bg_loop).result(timeout=180)

# ── ADK Runner (one per process) ────────────────────────────────────────────
@st.cache_resource
def get_runner():
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="sales_agent",
        agent=create_sales_agent(),
        session_service=session_service,
    )
    return runner, session_service

runner, session_service = get_runner()
SESSION_ID = "main-chat-session"
USER_ID    = "user"

# Create session once per browser tab
if "session_created" not in st.session_state:
    try:
        run_async(session_service.create_session(
            app_name="sales_agent", user_id=USER_ID, session_id=SESSION_ID
        ))
    except Exception:
        pass
    st.session_state.session_created = True

# ── Chat history ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Welcome! I'm your Sales Assistant. What product are you looking for today?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ── Handle new user input ────────────────────────────────────────────────────
if prompt := st.chat_input("Type your message…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            q = queue.Queue()

            async def _call_agent():
                try:
                    user_msg = types.Content(role="user", parts=[types.Part(text=prompt)])
                    response = ""
                    async for event in runner.run_async(
                        session_id=SESSION_ID,
                        user_id=USER_ID,
                        new_message=user_msg,
                    ):
                        if event.content and event.content.parts:
                            for part in event.content.parts:
                                if part.text:
                                    response += part.text
                    q.put(("ok", response))
                except Exception as e:
                    q.put(("error", str(e)))

            asyncio.run_coroutine_threadsafe(_call_agent(), _bg_loop)
            kind, text = q.get(timeout=180)

        if kind == "ok" and text.strip():
            st.markdown(text, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": text})
        elif kind == "error":
            st.error(f"⚠️ Error: {text}")
        else:
            st.warning("The agent returned an empty response. Please try rephrasing.")
