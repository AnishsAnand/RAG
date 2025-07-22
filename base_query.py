from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from csv_knowledge import load_csv_documents
from pdf_knowledge import load_pdf_documents

CHROMA_PATH = "chroma_combined_knowledge"
def create_vector_db(csv_files, pdf_files):
    csv_docs = load_csv_documents(csv_files)
    pdf_docs = load_pdf_documents(pdf_files)
    all_docs = csv_docs + pdf_docs

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    return db

def create_qa_chain():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    llm = ChatOllama(model="deepseek-v2:latest")

    retriever = db.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True 
    )

    return qa_chain

def main():
    csv_files = ["data/amazon.csv"]
    pdf_files = ["data/bedrock-ug.pdf"]
    create_vector_db(csv_files, pdf_files)
    qa_chain = create_qa_chain()
    print("\nRAG Chatbot is ready. Type 'exit' to quit.\n")
    while True:
        query = input("Your question:")
        if query.lower() == "exit":
            break

        result = qa_chain(query)
        print("\nAnswer:\n", result["result"])

if __name__ == "__main__":
    main()
