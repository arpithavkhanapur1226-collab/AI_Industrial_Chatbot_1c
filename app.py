import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Industrial AI Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI-Powered Industrial Chatbot")

st.write(
    "Ask questions about industrial robotics and automation."
)


# ============================================================
# LOAD PDF
# ============================================================

@st.cache_resource
def load_document():

    reader = PdfReader(
        "documents/industrial_robotics.pdf"
    )

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# CREATE CHUNKS
# ============================================================

@st.cache_resource
def create_chunks(text):

    lines = text.split("\n")

    chunks = []
    current = ""

    for line in lines:

        line = line.strip()

        if not line:
            continue

        current += line + " "

        if len(current) >= 700:

            chunks.append(
                current.strip()
            )

            current = ""

    if current.strip():

        chunks.append(
            current.strip()
        )

    return chunks


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# ============================================================
# CREATE FAISS INDEX
# ============================================================

@st.cache_resource
def create_index(
    chunks,
    _embedding_model
):

    embeddings = _embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index


# ============================================================
# LOAD QWEN
# ============================================================

@st.cache_resource
def load_qwen():

    # Smaller model to reduce RAM usage
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    model.eval()

    return tokenizer, model


# ============================================================
# LOAD EVERYTHING
# ============================================================

with st.spinner(
    "Loading industrial chatbot..."
):

    full_text = load_document()

    chunks = create_chunks(
        full_text
    )

    embedding_model = (
        load_embedding_model()
    )

    index = create_index(
        chunks,
        embedding_model
    )

    tokenizer, model = load_qwen()


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# ============================================================
# USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask an industrial question..."
)


if question:

    # ========================================================
    # DISPLAY USER QUESTION
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.write(question)


    # ========================================================
    # CREATE QUESTION EMBEDDING
    # ========================================================

    question_embedding = (
        embedding_model.encode(
            [question],
            convert_to_numpy=True
        )
    )

    question_embedding = (
        question_embedding.astype(
            "float32"
        )
    )


    # ========================================================
    # SEARCH FAISS
    # ========================================================

    distances, indices = index.search(
        question_embedding,
        min(3, len(chunks))
    )


    # ========================================================
    # GET RELEVANT CONTEXT
    # ========================================================

    context = "\n\n".join(
        chunks[i]
        for i in indices[0]
        if i >= 0
    )


    # ========================================================
    # CREATE PROMPT
    # ========================================================

    messages = [

        {
            "role": "system",
            "content": (
                "You are an industrial technical assistant. "
                "Answer ONLY using the provided context. "
                "Do not use outside knowledge. "
                "If the answer is not in the context, say: "
                "'I could not find this information in the "
                "provided document.' "
                "Give clear and concise technical answers."
            )
        },

        {
            "role": "user",
            "content": (
                "Context:\n"
                + context
                + "\n\nQuestion:\n"
                + question
                + "\n\nAnswer:"
            )
        }

    ]


    # ========================================================
    # APPLY CHAT TEMPLATE
    # ========================================================

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    # ========================================================
    # TOKENIZE
    # ========================================================

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False
        )


    # ========================================================
    # GET ONLY NEW TOKENS
    # ========================================================

    input_length = (
        inputs["input_ids"].shape[1]
    )

    generated_tokens = (
        outputs[0][input_length:]
    )


    # ========================================================
    # DECODE ANSWER
    # ========================================================

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    with st.chat_message("assistant"):

        st.write(answer)


    # ========================================================
    # SAVE ANSWER
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )