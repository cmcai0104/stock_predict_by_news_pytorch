# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import tushare as ts
import time
from datetime import datetime, timedelta
import re
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from tqdm import notebook, trange
import matplotlib.pyplot as plt
import seaborn as sns


def affected_date(series: pd.Series, trade_dates: pd.Series, deadline='15:00:00'):
    """
    根据新闻的创建时间获得该新闻影响的交易日
    Args:
        deadline: 新闻归入当前交易日的截止时间
        series: 一条新闻
        trade_dates: 交易日序列

    Returns: 日期

    """
    date = series['date']
    _time = series['time']
    if (_time <= datetime.strptime(deadline, "%H:%M:%S").time()) and (date in trade_dates.to_list()):
        return date
    else:
        return trade_dates[trade_dates > series['create_time']].min()


def data_preprocess(path="./data/history_news.txt", deadline='15:00:00'):
    """
    读取数据，关联上证综指，并进行数据预处理。
    Args:
        deadline: 归入下一个交易日的截止时间
        path: 历史新闻保存路径

    Returns: DataFrame

    """
    print('Reading the history news data!')
    df = pd.read_csv(path, sep="#split#", engine='python', encoding='utf-8')
    df = df.drop(columns=['modified_time', 'news_content'])
    df = df.astype(
        {'create_time': 'datetime64', 'title': 'str', 'reading_num': 'int', 'share_num': 'int', 'recommend_num': 'int'})
    df = df[df['title'] != 'nan']
    df['date'] = df['create_time'].dt.date
    df['time'] = df['create_time'].dt.time
    pro = ts.pro_api('0d52800e5aed61cb4188bfde75dceff83fef0a928a0363a12a3c27d2')
    szzz = pro.index_daily(ts_code='000001.SH',
                           start_date=datetime.strftime(df['date'].min(), '%Y%m%d'),
                           end_date=datetime.strftime(datetime.today(), '%Y%m%d'))
    szzz = szzz[['trade_date', 'pct_chg']].astype({'trade_date': 'datetime64', 'pct_chg': 'float'})
    df['affected_date'] = df.apply(lambda x: affected_date(x, trade_dates=szzz['trade_date'], deadline=deadline),
                                   axis=1)
    df = df.merge(szzz, how='left', left_on='affected_date', right_on='trade_date')
    return df


def daily_filter(daily_df: pd.DataFrame, tokenizer: BertTokenizer, max_news_num: int, max_news_length: int):
    """
    对每个交易日的新闻进行indices，并保存每条新闻词的数量，以及当天上证综指涨跌幅/10
    Args:
        tokenizer:
        daily_df:
        max_news_num:
        max_news_length:

    Returns:

    """
    daily_df = daily_df.sort_values(by=['share_num', 'recommend_num', 'reading_num'], ascending=False)
    daily_df = daily_df[['create_time', 'title', 'reading_num', 'share_num', 'recommend_num', 'affected_date']]
    news_num = len(daily_df)
    daily_df = daily_df.iloc[:max_news_num, :]
    sentences = daily_df['title'].to_list()
    sentences = [re.sub(r"(\u3000|\n|\t|\r|：|)", "", sent) for sent in sentences]
    news_length = max([len(sent) for sent in sentences])
    input_ids = []
    att_masks = []
    for sent in sentences:
        encode_sent = tokenizer.encode(sent, add_special_tokens=True, padding='max_length',
                                       truncation='longest_first', max_length=max_news_length)
        input_ids.append(encode_sent)
        att_mask = [int(token_id > 0) for token_id in encode_sent]
        att_masks.append(att_mask)
    while len(input_ids) < max_news_num:
        input_ids.append([0] * max_news_length)
        att_masks.append([0] * max_news_length)
    return input_ids, att_masks, news_num, news_length


def generate_dataloader(input_ids, att_masks, labels, batch_size):
    input_ids = torch.tensor(input_ids)
    att_masks = torch.tensor(att_masks)
    labels = torch.tensor(labels).float()
    data = TensorDataset(input_ids, att_masks, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader


def data_split(df, tokenizer, max_news_num, max_news_length, batch_size):
    """
    Spliting the DataFrame into training, validation and test. Encoding the texts by tokenizer,
    and create the masks and sentences length list at the same time.
    Args:
        df: Dataframe
        tokenizer: encoder tokenizer
        max_news_num: max sentence number
        max_news_length: max sentence length
        batch_size:

    Returns: Dataloader, sentences_length(inclouding training, validation and test)
    """
    print('Encoding texts by tokenizer, and splitting the dataset into training, valid and test......')
    news_num_list = []
    news_length_list = []
    train_input_ids = []
    valid_input_ids = []
    test_input_ids = []
    train_att_masks = []
    valid_att_masks = []
    test_att_masks = []
    train_labels = []
    valid_labels = []
    test_labels = []
    daily_df_list = [daily_df[1] for daily_df in df.groupby('affected_date')]
    for daily_df in daily_df_list:
        input_id, att_mask, news_num, news_length = daily_filter(daily_df, tokenizer=tokenizer,
                                                                 max_news_num=max_news_num,
                                                                 max_news_length=max_news_length)
        news_num_list.append(news_num)
        news_length_list.append(news_length)
        if daily_df['affected_date'].unique()[0] < np.datetime64('2020-01-01'):
            train_input_ids.append(input_id)
            train_att_masks.append(att_mask)
            train_labels.append(daily_df['pct_chg'].unique() / 10)
        elif daily_df['affected_date'].unique()[0] < np.datetime64('2020-07-01'):
            valid_input_ids.append(input_id)
            valid_att_masks.append(att_mask)
            valid_labels.append(daily_df['pct_chg'].unique() / 10)
        else:
            test_input_ids.append(input_id)
            test_att_masks.append(att_mask)
            test_labels.append(daily_df['pct_chg'].unique() / 10)

    print("Maximum length of news's title is %s, we just take the first %s words." % (
        max(news_length_list), max_news_length))
    print("The day with the most news releases was %s, we just take the top %s." % (max(news_num_list), max_news_num))
    train_dataloader = generate_dataloader(train_input_ids, train_att_masks, train_labels, batch_size)
    valid_dataloader = generate_dataloader(valid_input_ids, valid_att_masks, valid_labels, batch_size)
    test_dataloader = generate_dataloader(test_input_ids, test_att_masks, test_labels, batch_size)

    return train_dataloader, valid_dataloader, test_dataloader


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(timedelta(seconds=elapsed_rounded))


def save_checkpoint(save_path, model, valid_loss):
    if save_path is None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, device):
    if load_path is None:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path is None:
        return
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path, device):
    if load_path is None:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def training_history_plot(plt_save_path, destination_folder, device):
    train_loss_list, valid_loss_list, global_steps_list = load_metrics(destination_folder + '/metrics.pt', device)
    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, valid_loss_list, label='Valid')
    plt.title('Training and Validation loss of the model')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plt_save_path)
    plt.show()
    print('The training and valid loss curves have already saved to ==> {plt_save_path}')


def train(model, optimizer, scheduler, epochs, train_dataloader, valid_dataloader, device):
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_loss = 0
        model.train()
        for step, train_batch in enumerate(train_dataloader):
            if step % 10 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            train_batch = tuple(inp.to(device) for inp in train_batch)
            input_ids, input_mask, label_ids = train_batch
            y_predict = model(input_ids, input_mask)
            loss = torch.nn.functional.mse_loss(y_predict, label_ids)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            # clip the norm of the gradients to 1.0, to help prevent the "exploding gradients".
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)
        train_loss_list.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        print("")
        print("Running Validation...")
        t0 = time.time()
        model.eval()
        total_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for step, valid_batch in enumerate(valid_dataloader):
            valid_batch = tuple(inp.to(device) for inp in valid_batch)
            input_ids, input_mask, label_ids = valid_batch
            with torch.no_grad():
                y_predict = model(input_ids, input_mask)
                loss = torch.nn.functional.mse_loss(y_predict, label_ids)
            total_loss += loss.item()
            avg_valid_loss = total_loss / len(valid_dataloader)
            valid_loss_list.append(avg_valid_loss)
            # Track the number of batches
            nb_eval_steps += 1
        print("  Accuracy: {0:.2f}".format(total_loss / nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
    print("")
    print("Training complete!")




def train1(model, optimizer, train_loader, valid_loader, train_epochs, save_file_path, device, eval_every=None,
          best_valid_loss=float("Inf")):
    print('Start training the model!')
    if eval_every is None:
        eval_every = len(train_loader) // 2
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    model.train()
    for epoch in trange(int(train_epochs), desc='Epoch'):
        for _, train_batch in enumerate(notebook.tqdm(train_loader)):
            train_batch = tuple(t.to(device) for t in train_batch)
            input_ids, input_mask, label_ids = train_batch
            y_predict = model(input_ids, input_mask)
            loss = torch.nn.functional.mse_loss(y_predict, label_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    for _, valid_batch in enumerate(valid_loader):
                        valid_batch = tuple(t.to(device) for t in valid_batch)
                        input_ids, input_mask, label_ids = valid_batch
                        y_predict = model(input_ids, input_mask)
                        loss = torch.nn.functional.mse_loss(y_predict, label_ids)
                        valid_running_loss += loss.item()
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                print(
                    'Epoch [{}/{}], Step [{}/{}], Train Loss:{:.4f}, Valid Loss:{:.4f}'.format(epoch + 1, train_epochs,
                                                                                               global_step,
                                                                                               train_epochs * len(
                                                                                                   train_loader),
                                                                                               average_train_loss,
                                                                                               average_valid_loss))
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(save_file_path + '/model.pt', model, best_valid_loss)
                    save_metrics(save_file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(save_file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


def evaluate(model, test_loader, device):
    print('Start evaluating the model!')
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids = batch
            y_predict = model(input_ids, input_mask)
            y_pred.extend(y_predict.tolist())
            y_true.extend(label_ids.tolist())

    # print('Classification Report:')
    # print(classification_report(y_true, y_pred, label=[1, 0], digits=4))
    # cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    # ax = plt.subplot()
    # sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='d')
    # ax.set_title()
    # ax.set_xlabel()
    # ax.set_ylabel()
    # ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    # ax.yaxis.set_ticklabels(['FAKE', 'REAL'])
