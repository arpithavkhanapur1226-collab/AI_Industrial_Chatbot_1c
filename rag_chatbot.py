from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# ============================================================
# 1. READ PDF
# ============================================================

pdf_path = "documents/industrial_robotics.pdf"

print("=" * 60)
print("READING INDUSTRIAL DOCUMENT")
print("=" * 60)

reader = PdfReader(pdf_path)

full_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        full_text += text + "\n"

print("Number of pages:", len(reader.pages))
print("PDF loaded successfully!")


# ============================================================
# 2. CREATE CHUNKS
# ============================================================

print("\nCreating text chunks...")

lines = full_text.split("\n")

chunks = []
current_chunk = ""

for line in lines:

    line = line.strip()

    if not line:
        continue

    current_chunk += line + " "

    if len(current_chunk) >= 700:
        chunks.append(current_chunk.strip())
        current_chunk = ""

if current_chunk.strip():
    chunks.append(current_chunk.strip())

print("Number of chunks:", len(chunks))


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# ============================================================
# 4. CREATE EMBEDDINGS
# ============================================================

print("\nCreating embeddings...")

embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True
)

embeddings = embeddings.astype("float32")

print("Embedding shape:", embeddings.shape)


# ============================================================
# 5. CREATE FAISS INDEX
# ============================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS vectors:", index.ntotal)


# ============================================================
# 6. LOAD QWEN
# ============================================================

print("\nLoading Qwen model...")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForCausalLM.from_pretrained(
    model_name
)

print("Qwen loaded successfully!")


# ============================================================
# 7. CHAT LOOP
# ============================================================

print("\n" + "=" * 60)
print("AI INDUSTRIAL CHATBOT")
print("=" * 60)

print("You can ask multiple questions.")
print("Type 'exit' to close the chatbot.")


while True:

    question = input("\nYou: ")

    # Exit chatbot
    if question.lower() == "exit":
        print("\nChatbot closed.")
        break


    # ========================================================
    # 8. EMBED QUESTION
    # ========================================================

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )


    # ========================================================
    # 9. SEARCH FAISS
    # ========================================================

    k = min(3, len(chunks))

    distances, indices = index.search(
        question_embedding,
        k
    )


    # ========================================================
    # 10. RETRIEVE RELEVANT CHUNKS
    # ========================================================

    retrieved_chunks = []

    for i in indices[0]:

        if i >= 0:
            retrieved_chunks.append(chunks[i])


    context = "\n\n".join(retrieved_chunks)


    # ========================================================
    # 11. CREATE PROMPT
    # ========================================================

    messages = [

        {
            "role": "system",
            "content": (
                "You are an industrial technical assistant. "
                "Answer the user's question using ONLY the "
                "provided context. "
                "Do not use outside knowledge. "
                "If the answer is not present in the context, "
                "say: "
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
    # 12. CREATE QWEN PROMPT
    # ========================================================

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    # ========================================================
    # 13. TOKENIZE
    # ========================================================

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )


    # ========================================================
    # 14. GENERATE ANSWER
    # ========================================================

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )


    # ========================================================
    # 15. GET ONLY NEW TOKENS
    # ========================================================

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][input_length:]


    # ========================================================
    # 16. DECODE ANSWER
    # ========================================================

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )


    # ========================================================
    # 17. DISPLAY ANSWER
    # ========================================================

    print("\nBot:", answer)