import json
import pickle

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

import config
from clean_data_helpers import clean_data_multiple_methods

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
sentences = []

TOP_K = 20

embedder = SentenceTransformer('bert-base-nli-mean-tokens')


def cosine(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def run_query(model_path, query):
    with open(model_path, 'rb') as f, open(config.JOBS_TAGGING) as ff:
        jobs = json.load(ff)
        sbert_model_trained = pickle.load(f)

    query_embedding = embedder.encode(clean_data_multiple_methods(query), convert_to_tensor=True)

    # Find the top sentences of the corpus for each query sentence based on cosine similarity
    cos_scores = util.cos_sim(query_embedding, sbert_model_trained)[0]
    top_results = torch.topk(cos_scores, k=config.LENGTH)

    job_desc_query_scores = {}
    for idx, score in zip(top_results[1][:TOP_K], top_results[0][:TOP_K]):
        job_desc_query_scores[int(idx)] = '{:.2f}'.format(float(score))

    job_desc = [{f'job_description': jobs[str(key)]['job_description'], 'score': value} for key, value in
                job_desc_query_scores.items()]
    return job_desc
