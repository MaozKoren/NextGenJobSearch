import math
import re

import nltk
import torch
from gensim.summarization.summarizer import summarize

nltk.download('stopwords')

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import config
from sentence_transformers import SentenceTransformer, util

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

porter_stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def basic_clean_data(line):
    return line


def summerize_text(line):
    summary = None
    try:
        summary = summarize(line.encode("ascii", errors="ignore").decode(), word_count=50)
    except ValueError as ex:
        sentences = re.split(r':', line)
        line = '\n'.join(sentences)
        summary = summarize(line.encode("ascii", errors="ignore").decode(), word_count=20)

    return summary


def clean_data_with_stemming(line):
    return porter_stemmer.stem(line)


def clean_data_with_stop_words(line):
    return ' '.join([word for word in word_tokenize(line) if not word in stopwords.words()])


def clean_text(line):
    return re.sub(r'[^\w\s]', '', line, re.UNICODE) \
        .lower().encode('ascii', errors='ignore') \
        .decode()


def clean_data_multiple_methods(line):
    text = re.sub(r'[^\w\s]', '', line, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    return " ".join(text)


def make_short_description(line):
    list_from_description = re.split('; |, |\*|\n', line)
    embeddings = sbert_model.encode(list_from_description, convert_to_tensor=True, show_progress_bar=True)
    query_embedd = sbert_model.encode(config.GENERAL_DATA_SCIENCE_TEXT, convert_to_tensor=True, show_progress_bar=True)
    cos_scores = util.cos_sim(query_embedd, embeddings)[0]
    top_results = torch.topk(cos_scores, k=math.ceil(0.3 * len(list_from_description)))
    return ' '.join([list_from_description[i] for i in top_results[1]])
