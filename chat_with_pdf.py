import streamlit as st
import os
from openai import OpenAI
from os import environ
import io
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

client = OpenAI(
	api_key=os.environ["API_KEY"],
	base_url="https://api.ai.it.cornell.edu",
)

st.title("📝 File Q&A with OpenAI")
with st.sidebar:
    st.header("RAG Settings")
    chunk_size = st.slider("Chunk size", 300, 1500, 1000, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 200, 10)
    top_k = st.slider("Top-K", 2, 8, 4, 1)
    rebuild = st.button("Rebuild Index")

uploaded_files = st.file_uploader(
    "Upload .txt / .md / .pdf (multiple allowed)",
    type=("txt", "md", "pdf"),
    accept_multiple_files=True
)

disabled = not uploaded_files
question = st.chat_input("Ask something about the article(s)", disabled=disabled)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the article"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

def extract_text_from_pdf(uploaded_file) -> str:
    data = uploaded_file.read()
    reader = PdfReader(io.BytesIO(data))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n\n".join(texts)

if uploaded_files:
    with st.expander("Preview loaded sources"):
        for f in uploaded_files:
            st.write(f"• {f.name}")

if question and uploaded_files:
    # 1) Load all files' text with source
    docs = []  # [{"source": name, "text": "..."}]
    for f in uploaded_files:
        name_lower = f.name.lower()
        if name_lower.endswith(".pdf"):
            text = extract_text_from_pdf(f)
        else:
            text = f.read().decode("utf-8", errors="ignore")
        docs.append({"source": f.name, "text": text})

    # 2) Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks_texts, chunks_metas = [], []
    for d in docs:
        pieces = splitter.split_text(d["text"])
        for i, piece in enumerate(pieces):
            chunks_texts.append(piece)
            chunks_metas.append({"source": d["source"], "chunk_id": i})

    # 3) Embeddings
    embeddings = OpenAIEmbeddings(
        model="openai.text-embedding-3-small",
        api_key=os.environ["API_KEY"],
        base_url="https://api.ai.it.cornell.edu",
    )

    # 4) Vector store (Chroma, persisted)
    PERSIST_DIR = "./chroma_db"
    current_sources = sorted({m["source"] for m in chunks_metas})

    vs = Chroma(
        collection_name="info5940_rag",
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    vs._collection.delete(where={"source": {"$nin": current_sources}})

    ids = [f"{m['source']}#chunk{m['chunk_id']}" for m in chunks_metas]

    vs._collection.delete(where={"id": {"$in": ids}})

    vs.add_texts(texts=chunks_texts, metadatas=chunks_metas, ids=ids)
    vs.persist()

    # 5) Retrieve
    hits = vs.similarity_search(question, k=top_k)

    # 6) Build context with citations
    context_blocks = []
    for i, d in enumerate(hits, 1):
        src = d.metadata.get("source", "unknown")
        cid = d.metadata.get("chunk_id", "?")
        context_blocks.append(f"[{i}] ({src}#chunk{cid})\n{d.page_content}")
    context = "\n\n".join(context_blocks)

    # 7) Chat UI + call model
    st.session_state["messages"].append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful RAG assistant. Answer ONLY using the provided context. "
                "If the answer is not in the context, say you don't know. "
                "Cite sources as [1], [2], ... corresponding to the context blocks."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    with st.chat_message("assistant"):
        try:
            resp = client.chat.completions.create(
                model="openai.gpt-4o-mini",
                messages=messages,
                temperature=0.2,
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"Model request failed: {e}"
        st.write(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})

    # 8) Show sources (for grading)
    with st.expander("Sources"):
        for i, d in enumerate(hits, 1):
            st.markdown(
                f"**[{i}]** {d.metadata.get('source')} (chunk {d.metadata.get('chunk_id')})"
            )