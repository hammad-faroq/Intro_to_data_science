
# ✅ Install required packages first:
# pip install langchain langchain-community llama-cpp-python sentence-transformers faiss-cpu gradio

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

import gradio as gr

# Step 1: Load markdown files
loader = DirectoryLoader("knowledge_case", glob="**/*.md")
docs = loader.load()

# Step 2: Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Optional: Print some chunks
print(f"🔹 Total chunks: {len(chunks)}")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---\n{chunk.page_content[:300]}...")

# Step 3: Generate embeddings using open-source model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Step 4: Load quantized LLM from GGUF (use your model path)
llm = LlamaCpp(
    model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # Change path to your GGUF model
    temperature=0.7,
    max_tokens=512,
    top_p=0.95,
    n_ctx=2048,
    verbose=True
)

# Step 5: Retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Step 6: Gradio interface
def ask_hammad_ai(question):
    result = qa_chain.run(question)
    return result

gr.Interface(
    fn=ask_hammad_ai,
    inputs="text",
    outputs="text",
    title="🤖 Hammad's Personal AI Assistant",
    description="Ask anything about Hammad Farooq — his courses, projects, and work experience."
).launch()
