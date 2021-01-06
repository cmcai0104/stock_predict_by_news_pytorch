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
from base_functions import data_preprocess, data_split

if __name__ == '__main__':
    MAX_SENT_LENGTH = 50
    MAX_SENT_NUM = 50
    BATCH_SIZE = 2
    EPOCHS = 100
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 1e-2
    EPSILION = 1e-8
    RANDOM_SEED = 333
    GRADIENT_ACCUMULATION_STEPS = 1
    WARMUP_PROPORTION = 0.1
    global_step=0
    print('Reading the history news data......')
    df = data_preprocess(path="./data/history_news.txt", deadline='15:00:00')
    print('Encoding texts by tokenizer, and spliting the datasets into training, valid and test......')
    model_path = './pretrained_models/FinBERT_L-12_H-768_A-12_pytorch'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    train_dataloader, _, valid_dataloader, _, test_dataloader, _ = data_split(df, tokenizer=tokenizer, max_sent_num=MAX_SENT_NUM,
                                                                                max_sent_length=MAX_SENT_LENGTH, batch_size=BATCH_SIZE)
    del df
    print('Building the model and load pretrained parameters......')
    model = FinBertTransformer(pretrain_path=model_path, sents_num=MAX_SENT_NUM, sent_hidden=[96, 24],
                               nhead=1, num_layers=1, news_hidden=[12, 1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILION)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    model.train()
    print('Starting training the model......')
    for i in trange(int(EPOCHS), desc='Epoch'):
        train_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(notebook.tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids = batch
            del batch
            y_predict = model(input_ids, input_mask)
            loss =  torch.nn.functional.mse_loss(y_predict, label_ids)
            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            print("Loss: %s" % loss)

            train_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            if ((step + 1) % GRADIENT_ACCUMULATION_STEPS) == 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    i + 1, step, len(train_dataloader), 100. *
                    step / len(train_dataloader), loss.item()
                ))
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
    output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for input_ids, input_mask, label_ids in notebook.tqdm(valid_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            y_predict = model(input_ids, input_mask, labels=None)
        tmp_valid_loss =  torch.nn.functional.mse_loss(y_predict, label_ids)
        eval_loss += tmp_valid_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(y_predict.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], y_predict.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.squeeze(preds)
    result = compute_metrics(TASK_NAME, all_label_ids.numpy(), preds)
    result['eval_loss'] = eval_loss

    output_eval_file = os.path.join(REPORTS_DIR, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in (result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    