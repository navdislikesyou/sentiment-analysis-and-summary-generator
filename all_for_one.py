import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import joblib
import warnings
from tqdm import tqdm
import sys
import re
import csv
from io import StringIO, BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import sqlite3
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import os

# --- Initial Setup ---
warnings.filterwarnings("ignore", category=FutureWarning)

# Download NLTK stopwords is now handled by the build.sh script
# So we just run the download command here.
# This will succeed instantly as the data is already present.
nltk.download("stopwords")

# --- Configuration ---
MODEL_NAME = 'ProsusAI/finbert'
BATCH_SIZE = 16
MAX_LENGTH = 128
ENSEMBLE_WEIGHT = 0.7
DATABASE_FILE = "analysis_results.db"

label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
inverse_label_map = {v: k for k, v in label_map.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Global Models & DB Connection ---
predictor = None
t5_tokenizer = None
t5_model = None
db_conn = None

# --- Text Cleaning Function ---
contractions_dict = {
    "ain't": "am not", "aint": "am not", "aren't": "are not", "can't": "cannot",
    "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
    "let's": "let us", "ma'am": "madam", "might've": "might have", "mightn't": "might not",
    "must've": "must have", "mustn't": "must not", "needn't": "need not",
    "o'clock": "of the clock", "oughtn't": "ought not", "shan't": "shall not",
    "sha'n't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "so've": "so have",
    "so's": "so is", "that'd": "that would", "that's": "that is", "there'd": "there would",
    "there's": "there is", "they'd": "they would", "they'll": "they will",
    "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
    "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what'll": "what will", "what're": "what are", "what's": "what is",
    "what've": "what have", "when's": "when is", "where'd": "where did",
    "where's": "where is", "where've": "where have", "who'll": "who will",
    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
    "will've": "will have", "won't": "will not", "would've": "would have",
    "wouldn't": "would not", "y'all": "you all", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have"
}
stop_words = set(stopwords.words("english"))

def clean_text(text):
    try:
        text = str(text).lower()
        for contraction, expanded in contractions_dict.items():
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expanded, text)
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = " ".join([word for word in text.split() if word not in stop_words])
        return text
    except Exception as e:
        print(f"Error cleaning text: {e}", file=sys.stderr)
        return ""

# --- Sentiment Analysis Classes/Functions ---
class SentimentDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        text = str(self.texts[index])
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten()}

def get_transformer_probs(model, data_loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for d in data_loader:
            input_ids, attention_mask = d["input_ids"].to(device), d["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_probs.extend(torch.softmax(outputs.logits, dim=1).cpu().numpy())
    return np.array(all_probs)

class HybridSentimentPredictor:
    def __init__(self, finbert_path, svm_path, tfidf_path):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.finbert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_map))
        self.finbert_model.load_state_dict(torch.load(finbert_path, map_location=device, weights_only=False))
        self.finbert_model.to(device)
        self.finbert_model.eval()
        self.svm_model = joblib.load(svm_path)
        self.tfidf_vectorizer = joblib.load(tfidf_path)
    def predict(self, texts):
        if not texts or all(not t.strip() for t in texts): return ["neutral"] * len(texts)
        finbert_dataset = SentimentDataset(texts, self.tokenizer, MAX_LENGTH)
        finbert_loader = DataLoader(finbert_dataset, batch_size=min(BATCH_SIZE, len(texts)))
        finbert_probs = get_transformer_probs(self.finbert_model, finbert_loader, device)
        tfidf_features = self.tfidf_vectorizer.transform(texts)
        svm_decision_values = self.svm_model.decision_function(tfidf_features)
        if svm_decision_values.ndim == 1: svm_decision_values = svm_decision_values.reshape(1, -1)
        svm_probs = np.exp(svm_decision_values) / np.sum(np.exp(svm_decision_values), axis=1, keepdims=True)
        combined_probs = (ENSEMBLE_WEIGHT * finbert_probs) + ((1 - ENSEMBLE_WEIGHT) * svm_probs)
        final_preds_indices = np.argmax(combined_probs, axis=1)
        return [inverse_label_map[idx] for idx in final_preds_indices]

# --- Summarization Function ---
def summarize_text(text, tokenizer, model):
    def generate_summary_for_chunk(chunk_text, min_length=50, max_length=150):
        input_text = "summarize: " + chunk_text
        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
        summary_ids = model.generate(
            inputs.input_ids,
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            early_stopping=True,
            length_penalty=2.0
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    positive_keywords = ["Positive Impacts", "The Good News", "Benefits", "Pros"]
    negative_keywords = ["Negative Impacts", "The Not-So-Good News", "Challenges", "Cons", "Concerns"]
    positive_section, negative_section = None, None
    sections = re.split(r'### (.+)', text, flags=re.IGNORECASE)

    if len(sections) > 1:
        for i in range(1, len(sections), 2):
            heading, content = sections[i].strip(), sections[i+1].strip()
            if any(kw.lower() in heading.lower() for kw in positive_keywords):
                positive_section = content
            elif any(kw.lower() in heading.lower() for kw in negative_keywords):
                negative_section = content

    if positive_section and negative_section:
        print("Found and summarizing 'Positive' and 'Negative' sections.")
        pos_summary = generate_summary_for_chunk(
            positive_section,
            min_length=max(20, len(positive_section.split()) // 8),
            max_length=max(50, len(positive_section.split()) // 3)
        )
        neg_summary = generate_summary_for_chunk(
            negative_section,
            min_length=max(20, len(negative_section.split()) // 8),
            max_length=max(50, len(negative_section.split()) // 3)
        )
        return f"Positive Aspects: {pos_summary}\n\nConcerns: {neg_summary}"
    else:
        return generate_summary_for_chunk(
            text,
            min_length=max(10, len(text.split()) // 10),
            max_length=max(100, len(text.split()) // 4)
        )

# --- FastAPI Implementation ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, t5_tokenizer, t5_model, db_conn
    print("Loading models and setting up database...")
    try:
        # Load sentiment models directly from the current directory
        finbert_path = "FINBERT_FINAL.BIN"
        svm_path = "SVM_FINAL.PKL"
        tfidf_path = "TFIDF_VECTORIZER_FINAL.PKL"

        predictor = HybridSentimentPredictor(finbert_path, svm_path, tfidf_path)
        
        # Load summarization models
        t5_model_name = "t5-base"
        t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        t5_model.to(device)
        
        # Setup database
        db_conn = sqlite3.connect(DATABASE_FILE)
        cursor = db_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                summary TEXT NOT NULL
            )
        """)
        db_conn.commit()
        print("Models and database loaded successfully.")
    except Exception as e:
        print(f"Error during startup: {e}", file=sys.stderr)
        sys.exit(1)
    yield
    print("Shutting down...")
    if db_conn:
        db_conn.close()

app = FastAPI(lifespan=lifespan)

class SentimentResult(BaseModel):
    text: str
    sentiment: str
    summary: str

async def process_texts(texts: List[str]) -> List[SentimentResult]:
    results = []
    cursor = db_conn.cursor()
    for i, text in enumerate(texts):
        sentiment = predictor.predict([text])[0]
        summary = summarize_text(text, t5_tokenizer, t5_model)
        result = SentimentResult(text=text, sentiment=sentiment.upper(), summary=summary)
        results.append(result)
        cursor.execute("INSERT INTO results (text, sentiment, summary) VALUES (?, ?, ?)",
                         (result.text, result.sentiment, result.summary))
        print(f"{i + 1} done")
    db_conn.commit()
    return results

@app.post("/sentiment", response_model=List[SentimentResult])
async def analyze_sentiment(file: Optional[UploadFile] = File(None), text_data: Optional[str] = Form(None)):
    if not file and not text_data:
        raise HTTPException(status_code=400, detail="Either a CSV file or text_data must be provided.")
    texts_to_analyze = []
    if file:
        try:
            contents = await file.read()
            csv_file = StringIO(contents.decode('utf-8'))
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None)  # Skip header
            for row in csv_reader:
                if row: texts_to_analyze.append(row[0])
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Error processing CSV file: {e}")
    elif text_data:
        texts_to_analyze = [line.strip() for line in text_data.split('\n') if line.strip()]
    if not texts_to_analyze:
        raise HTTPException(status_code=422, detail="No valid text found in the input.")
    return await process_texts(texts_to_analyze)

@app.post("/wordcloud")
async def generate_wordcloud(file: Optional[UploadFile] = File(None), text_data: Optional[str] = Form(None)):
    if not file and not text_data:
        raise HTTPException(status_code=400, detail="Either a CSV file or text_data must be provided.")
    texts_to_process = []
    if file:
        try:
            contents = await file.read()
            csv_file = StringIO(contents.decode('utf-8'))
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None)  # Skip header
            for row in csv_reader:
                if row: texts_to_process.append(row[0])
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Error processing CSV file: {e}")
    elif text_data:
        texts_to_process = [line.strip() for line in text_data.split('\n') if line.strip()]
    if not texts_to_process:
        raise HTTPException(status_code=422, detail="No valid text found in the input.")
    
    cleaned_text = " ".join([clean_text(text) for text in texts_to_process])
    if not cleaned_text:
        raise HTTPException(status_code=422, detail="No processable words found after cleaning.")
    
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(cleaned_text)
    
    buf = BytesIO()
    wordcloud.to_image().save(buf, format='PNG')
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")

if __name__ == '__main__':
    import uvicorn
    print("Starting FastAPI server...")
    print("Access the API at http://127.0.0.1:8000")
    print("API documentation is available at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)