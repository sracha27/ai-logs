import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai

# === Config ===
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENAI_API_KEY = "AIzaSyB9fD0L8l-y_C-_bA8auAmY6phc03n5sIg"
SUPPORTED_EXTENSIONS = ('.log', '.txt', '.out', '.conf', '.json', '.xml')

# === Init Gemini Client ===
client = genai.Client(api_key=GENAI_API_KEY)

# === Utils ===
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def split_text(text, chunk_size=500, overlap=50):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

def embed_and_store(chunks, db_path):
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name="log_data")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"id_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)

def query_chunks(question, db_path):
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name="log_data")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = embedder.encode([question]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=15)
    return results["documents"]

def get_answer_from_gemini(chunks, question):
    context = "\n".join([c for sub in chunks for c in (sub if isinstance(sub, list) else [sub])])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )
    return response.text

# === UI ===
st.set_page_config(page_title="Log Q&A", layout="centered")
st.title("🧠 Ask Questions on Log Files")

log_dir = st.text_input("📂 Enter the folder path containing log files:")

if log_dir and os.path.isdir(log_dir):
    log_files = [f for f in os.listdir(log_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    if log_files:
        st.write("### 🗂️ Available Log Files:")
        for file in log_files:
            st.markdown(f"- `{file}`")

        selected_file = st.text_input("✏️ Type the exact name of the log file to use:")

        if selected_file in log_files:
            file_path = os.path.join(log_dir, selected_file)
            db_folder = os.path.join("chroma_dbs", selected_file.replace(".", "_"))
            os.makedirs(db_folder, exist_ok=True)

            if st.button("🔄 Vectorize Log"):
                text = load_text(file_path)
                chunks = split_text(text)
                embed_and_store(chunks, db_folder)
                st.success(f"✅ Vectorization complete for `{selected_file}`")

            st.markdown("---")
            question = st.text_input("❓ Ask your question about this log:")
            if question:
                docs = query_chunks(question, db_folder)
                answer = get_answer_from_gemini(docs, question)
                st.markdown("### 💡 Gemini's Answer:")
                st.write(answer)
        elif selected_file:
            st.error("❌ File not found in the listed log files.")
    else:
        st.warning("No supported `.log`, `.txt`, `.out`, `.conf`, `.json`, or `.xml` files found in this folder.")
else:
    st.info("🔍 Please enter a valid folder path above.")
