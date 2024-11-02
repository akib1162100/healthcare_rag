import os
import pickle
import pandas as pd
import numpy as np
import re
import faiss
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")


class DataRepository:
    def __init__(
        self, index_path="faiss_index.index", metadata_path="documents_metadata.pkl"
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.faiss_index = None
        self.documents = []

        self.csv_files = [
            "./documents/ATCDPP.csv",
            "./documents/VOSDCI.csv",
            "./documents/Gal.csv",
            "./documents/Ggr_Link.csv",
            "./documents/Hyr.csv",
            "./documents/Ir.csv",
            "./documents/MP.csv",
            "./documents/MPP.csv",
            "./documents/Sam.csv",
            "./documents/Stof.csv",
        ]
        self.excel_file = "./documents/msd_cleaned_file.xlsx"

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\\s]", "", text)
        text = re.sub(r"\\s+", " ", text).strip()
        return text

    def load_and_preprocess_csvs(self):
        """Load, preprocess, and combine all CSV files."""
        all_data = pd.DataFrame()
        for file in self.csv_files:
            df = pd.read_csv(file, delimiter=";")
            df = df.applymap(lambda x: self.preprocess_text(str(x)))
            all_data = pd.concat([all_data, df], ignore_index=True)
        all_data["combined_text"] = all_data.apply(
            lambda row: " ".join(row.values.astype(str)), axis=1
        )
        return all_data["combined_text"].tolist()

    def preprocess_excel_data(self, file_path):
        """Load and preprocess a new Excel file."""
        data = pd.read_excel(file_path)

        def preprocess_text_with_lemmatization(text):
            if pd.isna(text):
                return ""
            text = text.lower()
            text = re.sub(r"[^a-zA-Z\\s]", "", text)
            tokens = text.split()
            stop_words = set(stopwords.words("english"))
            tokens = [token for token in tokens if token not in stop_words]
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            return " ".join(tokens)

        text_columns = data.select_dtypes(include=["object"]).columns
        data[text_columns] = data[text_columns].applymap(
            preprocess_text_with_lemmatization
        )

        cleaned_file_path = "cleaned_msd_data.xlsx"
        data.to_excel(cleaned_file_path, index=False)
        data["combined_text"] = data.apply(
            lambda row: " ".join(row.values.astype(str)), axis=1
        )
        return data["combined_text"].tolist()

    def load_and_preprocess_excel(self):
        """Load and preprocess the initial Excel file."""
        return self.preprocess_excel_data(self.excel_file)

    def load_data(self):
        """Load data from both CSV and Excel files, and combine."""
        csv_texts = self.load_and_preprocess_csvs()
        excel_texts = self.load_and_preprocess_excel()
        self.documents = csv_texts + excel_texts

    def create_index(self):
        """Create and save a FAISS index using IndexIVFFlat to save memory."""
        self.load_data()
        vectors = (
            self.vectorizer.fit_transform(self.documents).astype(np.float32).toarray()
        )
        dimension = vectors.shape[1]

        quantizer = faiss.IndexFlatL2(dimension)

        nlist = 100
        self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

        self.faiss_index.train(vectors)

        self.faiss_index.add(vectors)

        self.save_index_and_metadata()

    def save_index_and_metadata(self):
        """Save the FAISS index and metadata to disk."""
        faiss.write_index(self.faiss_index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.documents, f)

    def load_index_and_metadata(self):
        """Load FAISS index and metadata from disk."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.faiss_index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.documents = pickle.load(f)
            self.vectorizer.fit(self.documents)

    def add_new_data(self, new_data_path):
        """Preprocess and add new data from an Excel file and update the FAISS index."""
        new_combined_texts = self.preprocess_excel_data(new_data_path)

        self.documents.extend(new_combined_texts)
        self.vectorizer.fit(self.documents)
        new_vectors = (
            self.vectorizer.transform(new_combined_texts).astype(np.float32).toarray()
        )

        self.faiss_index.add(new_vectors)
        self.save_index_and_metadata()
