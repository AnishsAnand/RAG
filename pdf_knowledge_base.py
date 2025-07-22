import fitz  
from langchain.schema import Document


def load_pdf_documents(file_list):
    documents = []
    for file in file_list:
        with open(file, "rb") as f:
            pdf = fitz.open(stream=f.read(), filetype="pdf")
            for page in pdf:
                text = page.get_text().strip()
                if text:
                    documents.append(Document(page_content=text))
    return documents
