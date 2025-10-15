# src/app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
from retriver import CPIRetriever
import pandas as pd

# ------------------------------
# Streamlit page config & CSS
# ------------------------------
st.set_page_config(page_title="AgroBot 🌾", layout="wide")

st.markdown("""
<style>
     body {
    background-color: #E9F3F0 !important;
}
header {
    background-color: #1EAC53;
    color: white;
    padding: 10px 20px;
    font-size: 20px;
    font-weight: 600;
}
.sidebar .sidebar-content {
    background-color: #1EAC53;
}
.stButton>button {
    background-color: #FFC107;
    color: white;
    border-radius: 10px;
    border: none;
    height: 38px;
    width: 100% !important;
    text-align: left;
    padding-left: 10px;
    margin-bottom: 5px;
}
.stButton>button:hover {
    background-color: #e0a800;
}
.stTextInput>div>div>input {
    border-radius: 25px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Sidebar content
# ------------------------------
with st.sidebar:
    st.markdown("# Agro Bot")
    st.markdown("## About")
    st.markdown(
        "**AgroBot** is your assistant for exploring **CPI data**, **crop prices**, and **market trends** in Pakistan."
    )

    st.markdown("## Quick Questions")
    quick_questions = [
        ("🌾 Wheat trend in Lahore", "What was the price trend of wheat in Lahore during 2023?"),
        ("🏙️ Cheapest city for rice", "Which city had the lowest rice price in 2022?"),
        ("📈 Inflation driver", "Which crop increased most in price last year?"),
    ]
    for label, question in quick_questions:
        if st.button(label, key=label):
            st.session_state.user_input = question

    st.markdown("---")
    st.markdown("### Built by Muhammad Hassan")

# ------------------------------
# Load Model + Retriever (cached)
# ------------------------------
@st.cache_resource
def load_model():
    st.write("⚙️ Loading Mistral-7B in 4-bit (optimized for RTX 3050 GPU)...")
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ✅ Correct device memory mapping
    max_memory = {
        0: "6GiB",      # GPU allocation
        "cpu": "32GiB"  # fallback to CPU
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return pipe


@st.cache_resource
def load_retriever():
    return CPIRetriever()


generator = load_model()
retriever = load_retriever()

# ------------------------------
# Chat system
# ------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

user_input = st.text_input("Ask about CPI, crop prices, or inflation:", st.session_state.user_input)
st.session_state.user_input = ""

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Retrieve relevant context
    retrieved = retriever.search(user_input, k=5)
    context = "\n".join([r[0] for r in retrieved])

    # Build prompt
    prompt = f"""
You are an agricultural and CPI data assistant. 
Use only the context below to answer the user's question concisely.

Context:
{context}

Question: {user_input}
Answer:
"""

    # Generate answer
    with st.spinner("Analyzing data..."):
        response = generator(prompt, max_new_tokens=250, temperature=0.2)
        raw_output = response[0]["generated_text"]

    # 🧹 Extract clean answer only
    if "Answer:" in raw_output:
        bot_reply = raw_output.split("Answer:")[-1].strip()
    else:
        bot_reply = raw_output.strip()

    # Append to history
    st.session_state.chat_history.append({"role": "bot", "content": bot_reply})

    # 📊 Optional: Show top 3 most relevant CPI records (city + price)
    if "wheat" in user_input.lower() or "profit" in user_input.lower():
        st.markdown("### 📍 Top 3 Relevant Cities")
        records = []
        for rec in retrieved[:3]:
            text = rec[0]
            # Simple parser to extract structured info
            parts = dict(
                part.split(": ", 1) for part in text.replace("CPI_RECORD | ", "").split(" | ") if ": " in part
            )
            records.append({
                "City": parts.get("city", "Unknown"),
                "Item": parts.get("item", "N/A"),
                "Price": parts.get("price", "N/A"),
                "Date": parts.get("date", "N/A")
            })
        df = pd.DataFrame(records)
        st.dataframe(df, hide_index=True)

# ------------------------------
# Display chat
# ------------------------------
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.chat_message("user").write(f"**🧑‍🌾 You:** {chat['content']}")
    else:
        st.chat_message("assistant").write(f"**🤖 AgroBot:** {chat['content']}")
