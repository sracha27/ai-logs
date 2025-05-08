import os
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

# === Load Text File ===
def load_document(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# === Split Text into Chunks ===
def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk.strip())
    return chunks

# === Initialize ChromaDB and Embedder ===
chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection(name="my_txt_collection")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Embed and Store Chunks with Metadata ===
def embed_and_store(chunks: List[str], file_prefix: str):
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{file_prefix}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source_file": file_prefix} for _ in chunks]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)

# === Query with Metadata Filter ===
def query_document(question: str, file_prefix: str, top_k: int = 100) -> List[str]:
    query_embedding = embedder.encode([question]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"source_file": file_prefix}
    )
    return results["documents"]

# === Use Gemini Model for Answer Generation ===
def generate_answer_from_gemini(chunks: List[str], question: str) -> str:
    flat_chunks = [chunk for sublist in chunks for chunk in (sublist if isinstance(sublist, list) else [sublist])]
    context = "\n".join(flat_chunks)

    prompt = f"""
You are a log analysis assistant.

Context:
{context}

---

User Question:
{question}

---

Instructions:
1. First, answer the user's question concisely.
2. If there are any errors, exceptions, or failure traces in the context, identify and explain them clearly.
3. Based on the question and logs, provide recommendations or next steps to resolve the issue or improve the system.
4. If there’s nothing critical, still suggest possible improvements or sanity checks.

Give a helpful and actionable response.
"""
    client = genai.Client(api_key="AIzaSyB9fD0L8l-y_C-_bA8auAmY6phc03n5sIg")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )
    return response.text

# === List Readable Text Files ===
def list_log_files(directory: str) -> List[str]:
    files = []
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    file.read(1024)
                files.append(f)
            except Exception:
                pass
    return files

# === Main Pipeline ===
if __name__ == "__main__":
    # 👇 Ask user for folder name
    log_dir = input("📁 Enter the name of the folder containing log files (or type 'exit'): ").strip()
    if log_dir.lower() == "exit":
        print("👋 Exiting.")
        exit()

    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        print("❌ Folder does not exist or is not a directory.")
        exit(1)

    log_files = list_log_files(log_dir)

    if not log_files:
        print("❌ No readable text files found in the folder.")
        exit(1)

    print("\n🗂 Available log files:")
    for idx, file in enumerate(log_files):
        print(f"{idx + 1}: {file}")

    selected = input("\n🔍 Enter the file numbers to process (comma-separated, or type 'exit'): ")
    if selected.lower() == "exit":
        print("👋 Exiting.")
        exit()

    try:
        indices = [int(i.strip()) - 1 for i in selected.split(",")]
        selected_files = [log_files[i] for i in indices if 0 <= i < len(log_files)]
    except Exception as e:
        print(f"❌ Invalid selection: {e}")
        exit(1)

    if not selected_files:
        print("❌ No valid files selected.")
        exit(1)

    # Process each selected file
    for file in selected_files:
        print(f"\n📄 Processing {file}...")
        text = load_document(os.path.join(log_dir, file))
        chunks = split_into_chunks(text)
        file_prefix = file.replace(".", "_")
        embed_and_store(chunks, file_prefix=file_prefix)

    print("\n✅ All selected documents stored in ChromaDB!")

    # Ask questions interactively
    while True:
        question = input("\n❓ Ask a question (or type 'exit'): ")
        if question.lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break

        for file in selected_files:
            file_prefix = file.replace(".", "_")
            print(f"\n🔎 Answering based on: {file}")
            top_k_chunks = query_document(question, file_prefix=file_prefix, top_k=100)
            answer = generate_answer_from_gemini(top_k_chunks, question)
            print("\n💡 Answer Generated by Gemini:\n")
            print(answer)
