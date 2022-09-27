from pathlib import Path

DATA_FILE_TRAIN = Path.cwd().parent / 'data' / 'jobs.csv'
JOBS_TAGGING = Path.cwd().parent / 'data' / 'jobs_taggings.json'
DATA_NER_TRAIN = Path.cwd().parent / 'data' / 'train_NER.csv'
MODELS_PATH = Path.cwd().parent / 'data' / 'models'
MODELS_SCORES_PATH = Path.cwd().parent / 'data' / 'models_top_k_scores'
JOB_TAGS_EMBDDDINGS_MODEL = MODELS_PATH / f'jobs_tags_embedding.bin'
LENGTH = 250

MODELS = ['basic_clean_data', 'clean_data_with_text_summeraztion', 'clean_data_with_stemming']

MODEL_PATH_DESC = MODELS_PATH / 'sentence_embeddings_job_description_make_short_description_250.bin'

MODEL_PATH_SUMMARY = MODELS_PATH / 'sentence_embeddings_job_summary_clean_data_multiple_methods_250.bin'

GENERAL_DATA_SCIENCE_TEXT = 'data scientist machine learning python jave sql'


class Query:
    QUERY_1_long = 'Senior data science health indusrty java python hadoop sql'
    QUERY_1_short = 'data science python'

    QUERY_2_long = 'junior data science python analytics spark'
    QUERY_2_short = 'data science'

    QUERY_3_long = 'statistical tools data enegineer SQL analytical models optimization predictive python complex data sets'
    QUERY_3_short = 'data enegineer SQL analytical models'


class DOC_HEADERS:
    DESCRIPTION = 12
    TITLE = 2
