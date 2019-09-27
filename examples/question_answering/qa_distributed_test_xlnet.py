QUICK_RUN = False

import os
import sys

import torch
import numpy as np

nlp_path = os.path.abspath('../../')
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)

from utils_nlp.dataset.squad import load_pandas_df
from utils_nlp.models.transformers.question_answering_distributed import AnswerExtractor
from utils_nlp.models.transformers.qa_utils_distributed import (QADataset, 
                                                    postprocess_answer, 
                                                    evaluate_qa, 
                                                    TOKENIZER_CLASSES,
                                                    get_qa_dataloader_distributed
                                                   )
from utils_nlp.common.timer import Timer

TRAIN_DATA_USED_PERCENT = 1
DEV_DATA_USED_PERCENT = 1
NUM_EPOCHS = 5

if QUICK_RUN:
    TRAIN_DATA_USED_PERCENT = 0.001
    DEV_DATA_USED_PERCENT = 0.01
    NUM_EPOCHS = 1

if torch.cuda.is_available() and torch.cuda.device_count() >= 4:
    MAX_SEQ_LENGTH = 384
    DOC_STRIDE = 128
    BATCH_SIZE = 2
else:
    MAX_SEQ_LENGTH = 128
    DOC_STRIDE = 64
    BATCH_SIZE = 2

print("Max sequence length: {}".format(MAX_SEQ_LENGTH))
print("Document stride: {}".format(DOC_STRIDE))
print("Batch size: {}".format(BATCH_SIZE))
    
SQUAD_VERSION = "v1.1" 
CACHE_DIR = "./temp"

# MODEL_NAME = "bert-large-uncased-whole-word-masking"
# DO_LOWER_CASE = True

MODEL_NAME = "xlnet-large-cased"
DO_LOWER_CASE = False

MAX_QUESTION_LENGTH = 64
LEARNING_RATE = 3e-5

DOC_TEXT_COL = "doc_text"
QUESTION_TEXT_COL = "question_text"
ANSWER_START_COL = "answer_start"
ANSWER_TEXT_COL = "answer_text"
QA_ID_COL = "qa_id"
IS_IMPOSSIBLE_COL = "is_impossible"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if  torch.cuda.device_count() > 0:
    torch.cuda.manual_seed_all(RANDOM_SEED)

train_df = load_pandas_df(local_cache_path=".", squad_version="v1.1", file_split="train")
train_df = train_df.sample(frac=TRAIN_DATA_USED_PERCENT).reset_index(drop=True)
train_dataset = QADataset(df=train_df,
                          doc_text_col=DOC_TEXT_COL,
                          question_text_col=QUESTION_TEXT_COL,
                          qa_id_col=QA_ID_COL,
                          is_impossible_col=IS_IMPOSSIBLE_COL,
                          answer_start_col=ANSWER_START_COL,
                          answer_text_col=ANSWER_TEXT_COL)
train_dataloader = get_qa_dataloader_distributed(train_dataset, 
                                    model_name=MODEL_NAME, 
                                    is_training=True,
                                    to_lower=DO_LOWER_CASE,
                                    batch_size=BATCH_SIZE
                                        )

qa_extractor = AnswerExtractor(model_name=MODEL_NAME, cache_dir=CACHE_DIR)

with Timer() as t:
    qa_extractor.fit(train_dataloader=train_dataloader,
                     num_epochs=NUM_EPOCHS,
                     learning_rate=LEARNING_RATE,
                     gradient_accumulation_steps = 6,
                     cache_model=True)
print("Training time : {:.3f} hrs".format(t.interval / 3600))

