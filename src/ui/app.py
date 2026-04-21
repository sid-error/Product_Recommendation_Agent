import streamlit as st
import asyncio
import os
import sys
import re
import json
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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Assistant",
    page_icon="🛍️",
    layout="wide",
)

# ── Inject global CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ------- product card ------- */
.product-card {
    background: #1e1e2e;
    border: 1px solid #2e2e4e;
    border-radius: 14px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    height: 100%;
}
.product-card img {
    width: 100%;
    max-width: 180px;
    height: 180px;
    object-fit: contain;
    border-radius: 10px;
    background: #fff;
}
.product-card .prod-title {
    font-weight: 600;
    font-size: 0.88rem;
    text-align: center;
    color: #e0e0f0;
    line-height: 1.3;
}
.product-card .prod-price {
    font-size: 1.1rem;
    font-weight: 700;
    color: #a78bfa;
}

/* ------- cart sidebar ------- */
[data-testid="stSidebar"] {
    background: #12121f !important;
}
.cart-item {
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid #2e2e4e;
    padding: 10px 0;
}
.cart-item img {
    width: 56px;
    height: 56px;
    object-fit: contain;
    border-radius: 8px;
    background: #fff;
}
.cart-total {
    font-size: 1.15rem;
    font-weight: 700;
    color: #a78bfa;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── Helper: parse PRODUCTS_JSON marker from agent response ────────────────────
_MARKER_RE = re.compile(r"<!--\s*PRODUCTS_JSON:\s*(\[.*?\])\s*-->", re.DOTALL)

def split_response(text: str):
    """Return (chat_text, products_list | None)."""
    m = _MARKER_RE.search(text)
    if not m:
        return text.strip(), None
    chat_text = text[:m.start()].strip()
    try:
        products = json.loads(m.group(1))
    except Exception:
        products = None
    return chat_text, products

# ── Session state ────────────────────────────────────────────────────────────
if "cart" not in st.session_state:
    st.session_state.cart = []          # list of product dicts
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Welcome! I'm your Sales Assistant. What product are you looking for today?", "products": None}
    ]

# ── Background async loop ────────────────────────────────────────────────────
@st.cache_resource
def get_async_loop():
    import threading
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop

_bg_loop = get_async_loop()

# ── ADK runner ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_runner():
    svc = InMemorySessionService()
    r = Runner(app_name="sales_agent", agent=create_sales_agent(), session_service=svc)
    return r, svc

runner, session_service = get_runner()
SESSION_ID = "main-chat-session"
USER_ID    = "user"

if "session_created" not in st.session_state:
    fut = asyncio.run_coroutine_threadsafe(
        session_service.create_session(app_name="sales_agent", user_id=USER_ID, session_id=SESSION_ID),
        _bg_loop
    )
    try:
        fut.result(timeout=10)
    except Exception:
        pass
    st.session_state.session_created = True

# ── Sidebar — Cart ───────────────────────────────────────────────────────────
with st.sidebar:
    cart = st.session_state.cart
    n = len(cart)
    st.markdown(f"## 🛒 Cart &nbsp; <sup style='font-size:0.9rem; background:#a78bfa; border-radius:50%; padding:2px 7px; color:#fff'>{n}</sup>", unsafe_allow_html=True)
    st.markdown("---")

    if not cart:
        st.info("Your cart is empty.")
    else:
        total = 0.0
        to_remove = None
        for i, item in enumerate(cart):
            img_url = item.get("imageUrl", "")
            title   = item.get("title", "Product")
            price   = item.get("offerPrice", 0)
            total  += price
            col_img, col_info = st.columns([1, 3])
            with col_img:
                if img_url:
                    st.markdown(
                        f'<img src="{img_url}" style="width:56px;height:56px;object-fit:contain;border-radius:8px;background:#fff;" referrerpolicy="no-referrer"/>',
                        unsafe_allow_html=True
                    )
            with col_info:
                st.markdown(f"**{title}**")
                st.markdown(f"₹{price:,.0f}")
                if st.button("Remove", key=f"remove_{i}"):
                    to_remove = i
            st.markdown("---")

        if to_remove is not None:
            st.session_state.cart.pop(to_remove)
            st.rerun()

        st.markdown(f"<div class='cart-total'>Total: ₹{total:,.0f}</div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Cart", use_container_width=True):
            st.session_state.cart = []
            st.rerun()

# ── Main chat area ───────────────────────────────────────────────────────────
st.title("🛍️ Sales Assistant")

def render_products(products: list):
    """Render product cards with Add to Cart buttons."""
    if not products:
        return
    cols = st.columns(min(len(products), 5))
    for i, prod in enumerate(products):
        with cols[i % len(cols)]:
            img_url = prod.get("imageUrl", "")
            title   = prod.get("title", "Product")
            price   = prod.get("offerPrice", 0)
            st.markdown(
                f"""
                <div class="product-card">
                    <img src="{img_url}" referrerpolicy="no-referrer"/>
                    <div class="prod-title">{title}</div>
                    <div class="prod-price">₹{price:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("🛒 Add to Cart", key=f"add_{i}_{hash(title)}", use_container_width=True):
                st.session_state.cart.append(prod)
                st.toast(f"Added **{title}** to cart!", icon="✅")
                st.rerun()

def render_message(msg: dict):
    role     = msg["role"]
    content  = msg["content"]
    products = msg.get("products")

    with st.chat_message(role):
        if content:
            st.markdown(content, unsafe_allow_html=True)
        if role == "assistant" and products:
            render_products(products)

# Display history
for msg in st.session_state.messages:
    render_message(msg)

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Type your message…"):
    # Immediately show the user message
    user_msg_dict = {"role": "user", "content": prompt, "products": None}
    st.session_state.messages.append(user_msg_dict)
    render_message(user_msg_dict)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            q = queue.Queue()

            async def _call_agent():
                try:
                    user_content = types.Content(role="user", parts=[types.Part(text=prompt)])
                    response = ""
                    async for event in runner.run_async(
                        session_id=SESSION_ID,
                        user_id=USER_ID,
                        new_message=user_content,
                    ):
                        if event.content and event.content.parts:
                            for part in event.content.parts:
                                if part.text:
                                    response += part.text
                    q.put(("ok", response))
                except Exception as e:
                    q.put(("error", str(e)))

            asyncio.run_coroutine_threadsafe(_call_agent(), _bg_loop)
            kind, raw = q.get(timeout=180)

        if kind == "error":
            st.error(f"⚠️ {raw}")
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {raw}", "products": None})
        elif kind == "ok" and raw.strip():
            chat_text, products = split_response(raw)
            if chat_text:
                st.markdown(chat_text, unsafe_allow_html=True)
            if products:
                render_products(products)
            st.session_state.messages.append({"role": "assistant", "content": chat_text, "products": products})
        else:
            st.warning("The agent returned an empty response. Please try rephrasing.")
