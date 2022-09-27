import json
import pickle

#
# This is good stuff
# https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/
#
#
from sentence_transformers import SentenceTransformer

import clean_data_helpers
import config
from config import MODELS_PATH, JOBS_TAGGING, JOB_TAGS_EMBDDDINGS_MODEL

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
sentences = []


def train_model(clean_data, header):
    with open(JOBS_TAGGING, encoding='utf-8') as f:
        print('starting to process jobs')

        jobs = json.load(f)
        for job in jobs.values():
            job_sentence = clean_data(job[header])
            sentences.append(job_sentence)

        print('finished to process jobs')

    print('starting to train sbert')
    sentence_embeddings = sbert_model.encode(sentences,
                                             convert_to_tensor=True,
                                             show_progress_bar=True)
    print('finished to train sbert')
    WORD_EMBDDDINGS_MODEL = MODELS_PATH / f'sentence_embeddings_{header}_{clean_data.__name__}_{config.LENGTH}.bin'

    print('saving model')
    with open(WORD_EMBDDDINGS_MODEL, 'wb') as f:
        pickle.dump(sentence_embeddings, f)

    print('model train finished !!!')


def train_model_for_job_taggins(clean_data, header):
    with open(JOBS_TAGGING, encoding='utf-8') as f:
        print('starting to process jobs')

        tags = []
        jobs = json.load(f)
        for job in jobs.values():
            tag = clean_data(job[header])
            tags.append(tag)

        print('finished to process tag')

    print('starting to train sbert')
    job_tags_embeddings = sbert_model.encode(tags,
                                             convert_to_tensor=True,
                                             show_progress_bar=True)
    print('finished to train sbert')

    print('saving model')
    with open(JOB_TAGS_EMBDDDINGS_MODEL, 'wb') as f:
        pickle.dump(job_tags_embeddings, f)

    print('model train finished !!!')


# train_model(clean_data_helpers.clean_data_with_text_summeraztion, 'clean_data_with_text_summeraztion')
train_model(clean_data_helpers.make_short_description, 'job_description')
# train_model(clean_data_helpers.summerize_text, 'job_summary')
#train_model_for_job_taggins(clean_data_helpers.clean_data_multiple_methods, 'tag')
