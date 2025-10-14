# src/inference.py
from transformers import pipeline
from retriver import CPIRetriever

def load_model():
    print("🤖 Loading FLAN-T5-Base (lightweight)...")
    model_name = "google/flan-t5-base"
    generator = pipeline("text2text-generation", model=model_name)
    return generator

def ask_question(question, top_k=5):
    retriever = CPIRetriever()
    retrieved = retriever.search(question, k=top_k)
    context = "\n".join([r[0] for r in retrieved])

    prompt = f"""
You are a CPI analyst. Use only the information below to answer the question.

Context:
{context}

Question: {question}
Answer:
"""

    generator = load_model()
    response = generator(prompt, max_new_tokens=200)
    print("\n🧠 Model Response:\n", response[0]["generated_text"])

if __name__ == "__main__":
    q = input("Ask your CPI question: ")
    ask_question(q)
