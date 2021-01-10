# -*- coding:utf-8 -*-
import os
import sys
import numpy as np
import joblib
from tqdm import notebook, trange
import re
import random
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup
from neural_network import FinBertTransformer
from base_functions import data_preprocess, data_split, train

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPu:', torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print('No GPU available, using the CPU instead.')

if __name__ == '__main__':
    SEED = 333
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    MAX_NEWS_LENGTH = 50
    MAX_NEWS_NUM = 128
    BATCH_SIZE = 2
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 1e-2
    EPSILION = 1e-8
    RANDOM_SEED = 333
    GRADIENT_ACCUMULATION_STEPS = 1
    WARMUP_PROPORTION = 0.1
    global_step = 0
    df = data_preprocess(path="./data/history_news.txt", deadline='15:00:00')
    model_path = './pretrained_models/FinBERT_L-12_H-768_A-12_pytorch'
    print('Loading BERT tokenizer from <==', model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    train_dataloader, valid_dataloader, test_dataloader = data_split(df, tokenizer=tokenizer,
                                                                     max_news_num=MAX_NEWS_NUM,
                                                                     max_news_length=MAX_NEWS_LENGTH,
                                                                     batch_size=BATCH_SIZE)
    del df
    print('Building the model and load pre-trained parameters from <==', model_path)
    model = FinBertTransformer(pretrain_path=model_path, sents_num=MAX_NEWS_NUM, sent_hidden=[96, 24],
                               nhead=1, num_layers=1, news_hidden=[12, 1])

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILION)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    train(model=model, optimizer=optimizer, scheduler=scheduler, epochs=EPOCHS,
          train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, device=device)


