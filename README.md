#  INFO 5940 – Assignment 1: RAG Application  
**Author:** Yiyang Dai  
**Description:**  
This project implements a **Retrieval-Augmented Generation (RAG)** application using **LangChain**, **ChromaDB**, and **Streamlit**.  
Users can upload `.txt` and `.pdf` files, and then interact with their content through a chat interface powered by OpenAI models.

---

##  Features
- Upload multiple documents (`.txt` and `.pdf`)
- Automatic text chunking for large documents  
- Adjustable parameters (chunk size, overlap, top-K retrieval)
- Efficient document retrieval using **ChromaDB**
- Context-aware chat interface via **Streamlit**
- Rebuild index anytime from the sidebar

---

##  Technologies Used
**Streamlit**  Web interface for chat and file upload 
**LangChain**  Text splitting and retrieval logic 
**ChromaDB** | Vector database for semantic search 
**OpenAI API** | Embeddings and answer generation

---

##  How to Run
Open the terminal inside your GitHub Codespace.

Install dependencies (if not already installed):

pip install -r requirements.txt


Set your API key (do not include it in your repository):

export API_KEY="your_actual_API_KEY"


Run the Streamlit application:

streamlit run chat_with_pdf.py


When prompted, click “Open in Browser” to open the app interface.

On the web interface:

Use the sidebar to adjust Chunk size, Overlap, and Top-K values.

Upload one or more .txt, .md, or .pdf files.

Ask questions about your uploaded documents in the chat box.

View retrieved chunks and citations under the Sources section.



