import pandas as pd
from langchain.schema import Document

def detect_sentiment(rating):
    try:
        rating = float(rating)
        if rating == 5:
            return "Excellent"
        elif rating >= 4:
            return "Good"
        elif rating == 3:
            return "Average"
        elif rating >= 2:
            return "Poor"
        else:
            return "Negative"
    except:
        return "Unknown"

def load_csv_documents(file_list):
    documents = []
    for file in file_list:
        df = pd.read_csv(file, encoding="ISO-8859-1", encoding_errors="ignore")
        for _, row in df.iterrows():
            if "reviewText" in df.columns:
                rating = row.get("overall", 3)
                sentiment = detect_sentiment(rating)
                text = f"Review Sentiment: {sentiment}\nReview: {row['reviewText']}\nRating: {rating}"
            elif "question" in df.columns and "answer" in df.columns:
                text = f"FAQ: {row['question']}\nAnswer: {row['answer']}"
            else:
                text = " ".join(
                    [f"{col}: {str(row[col])}" for col in df.columns if pd.notnull(row[col])]
                )
            documents.append(Document(page_content=text))
    return documents
