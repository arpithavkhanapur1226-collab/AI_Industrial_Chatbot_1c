# 🤖 AI-Powered Industrial Chatbot

An AI-powered industrial chatbot that uses **Retrieval-Augmented Generation (RAG)** to answer technical questions related to **industrial robotics and automation** from a provided technical document.

## 📌 Features

* 📄 Reads information from industrial PDF documents
* ✂️ Splits document content into smaller chunks
* 🧠 Generates text embeddings using **Sentence Transformers**
* 🔎 Uses **FAISS** for similarity-based document retrieval
* 🤖 Uses **Qwen2.5-0.5B-Instruct** from Hugging Face to generate answers
* 💬 Provides an interactive **Streamlit** chatbot interface
* 🔐 Restricts answers to information retrieved from the provided document

## 🛠️ Technologies Used

* **Python**
* **Hugging Face Transformers**
* **Qwen2.5-0.5B-Instruct**
* **Sentence Transformers**
* **FAISS**
* **PyTorch**
* **PyPDF**
* **Streamlit**

## 🏗️ System Workflow

```text
Industrial PDF
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Sentence Transformer
     ↓
Text Embeddings
     ↓
FAISS Vector Database
     ↓
User Question
     ↓
Question Embedding
     ↓
Similarity Search
     ↓
Relevant Context
     ↓
Qwen2.5 LLM
     ↓
Generated Answer
     ↓
Streamlit Chatbot
```

## 📂 Project Structure

```text
AI_CHATBOT/
│
├── app.py
├── rag_chatbot.py
├── read_pdf.py
├── chunk_text.py
├── embedding_test.py
├── faiss_test.py
├── test_model.py
│
├── documents/
│   └── industrial_robotics.pdf
│
└── .gitignore
```

## 🚀 Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install the required libraries:

```bash
pip install streamlit transformers torch sentence-transformers faiss-cpu pypdf accelerate numpy
```

## ▶️ Run the Chatbot

From the project directory:

```powershell
python -m streamlit run app.py --server.fileWatcherType none
```

The Streamlit application will open in your browser.

## 💬 Example Questions

```text
What is a PLC?

What is an ultrasonic sensor?

What are proximity sensors?

What is a servo motor?

What is an end effector?

What are conveyor systems?

What are robot safety measures?
```

## 🧠 RAG Approach

The chatbot follows a **Retrieval-Augmented Generation** approach.

First, the industrial document is extracted and divided into chunks. Each chunk is converted into a numerical embedding using `all-MiniLM-L6-v2`. FAISS stores these embeddings and retrieves the chunks most relevant to the user's question.

The retrieved information is then provided as context to the **Qwen2.5-0.5B-Instruct** model, which generates the final response.

## ⚠️ Limitations

* The chatbot is designed specifically for the information available in the provided industrial document.
* The current implementation runs locally on the CPU.
* Response quality depends on the quality and coverage of the source document.
* The project uses RAG and prompt engineering; **the Qwen model is not fine-tuned**.

## 🔮 Future Improvements

* Add more industrial technical documents
* Support multiple PDF documents
* Improve document chunking and retrieval
* Add conversation history
* Deploy the chatbot online
* Use a larger language model when sufficient hardware is available

## 👩‍💻 Author

**Arpitha V Khanapur**

BE – Robotics & Artificial Intelligence


