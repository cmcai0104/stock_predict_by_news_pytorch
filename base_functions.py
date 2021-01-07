# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import tushare as ts
from datetime import datetime
import re
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer
from tqdm import notebook, trange
import matplotlib.pyplot as plt
import seaborn as sns


def affected_date(series: pd.Series, trade_dates: pd.Series, deadline='15:00:00'):
    """
    根据新闻的创建时间获得该新闻影响的交易日
    Args:
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


def daily_filter(daily_df: pd.DataFrame, tokenizer: BertTokenizer, max_sent_num: int, max_sent_length: int):
    """
    对每个交易日的新闻进行indices，并保存每条新闻词的数量，以及当天上证综指涨跌幅/10
    Args:
        daily_df:
        max_sent_num:
        max_sent_length:

    Returns:

    """
    daily_df = daily_df.sort_values(by=['share_num', 'recommend_num', 'reading_num'], ascending=False)
    daily_df = daily_df[['create_time', 'title', 'reading_num', 'share_num', 'recommend_num', 'affected_date']]
    daily_df = daily_df.iloc[:max_sent_num, :]
    sentences = daily_df['title'].to_list()
    sentences = [re.sub(r"(\u3000|\n|\t|\r|：|)", "", sent) for sent in sentences]
    text = []
    atten_masks = []
    sent_len = []
    for sent in sentences:
        indices = tokenizer.encode(sent[:max_sent_length])
        seq_mask = [1.] * len(indices)
        seq_mask.extend([0.] * (max_sent_length + 2 - len(indices)))
        atten_masks.append(seq_mask)
        sent_len.append(len(indices))
        if len(indices) < max_sent_length + 2:
            indices.extend([0] * (max_sent_length + 2 - len(indices)))
        text.append(indices)
    while len(text) < max_sent_num:
        text.append([0] * (max_sent_length + 2))
        atten_masks.append([0.] * (max_sent_length + 2))
        sent_len.append(0)
    return text, atten_masks, sent_len


def data_split(df, tokenizer, max_sent_num, max_sent_length, batch_size):
    """
    Spliting the DataFrame into training, validation and test. Encoding the texts by tokenizer,
    and create the masks and sentences length list at the same time.
    Args:
        df: Dataframe
        tokenizer: encoder tokenizer
        max_sent_num: max sentence number
        max_sent_length: max sentence length
        batch_size:

    Returns: Dataloader, sentences_length(inclouding training, validation and test)
    """
    print('Encoding texts by tokenizer, and splitting the dataset into training, valid and test......')
    texts_train = []
    texts_valid = []
    texts_test = []
    atten_masks_train = []
    atten_masks_valid = []
    atten_masks_test = []
    sentences_length_train = []
    sentences_length_valid = []
    sentences_length_test = []
    labels_train = []
    labels_valid = []
    labels_test = []
    daily_df_list = [daily_df[1] for daily_df in df.groupby('affected_date')]
    for daily_df in daily_df_list:
        text, atten_mask, sent_len = daily_filter(daily_df, tokenizer=tokenizer, max_sent_num=max_sent_num,
                                                  max_sent_length=max_sent_length)
        if daily_df['affected_date'].unique()[0] < np.datetime64('2020-01-01'):
            texts_train.append(text)
            atten_masks_train.append(atten_mask)
            sentences_length_train.append(sent_len)
            labels_train.append(daily_df['pct_chg'].unique() / 10)
        elif daily_df['affected_date'].unique()[0] < np.datetime64('2020-07-01'):
            texts_valid.append(text)
            atten_masks_valid.append(atten_mask)
            sentences_length_valid.append(sent_len)
            labels_valid.append(daily_df['pct_chg'].unique() / 10)
        else:
            texts_test.append(text)
            atten_masks_test.append(atten_mask)
            sentences_length_test.append(sent_len)
            labels_test.append(daily_df['pct_chg'].unique() / 10)
    texts_train = torch.tensor(texts_train)
    texts_valid = torch.tensor(texts_valid)
    texts_test = torch.tensor(texts_test)
    atten_masks_train = torch.tensor(atten_masks_train)
    atten_masks_valid = torch.tensor(atten_masks_valid)
    atten_masks_test = torch.tensor(atten_masks_test)
    sentences_length_train = torch.tensor(sentences_length_train)
    sentences_length_valid = torch.tensor(sentences_length_valid)
    sentences_length_test = torch.tensor(sentences_length_test)
    labels_train = torch.tensor(labels_train).float()
    labels_valid = torch.tensor(labels_valid).float()
    labels_test = torch.tensor(labels_test).float()

    train_data = TensorDataset(texts_train, atten_masks_train, labels_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    valid_data = TensorDataset(texts_valid, atten_masks_valid, labels_valid)
    valid_sampler = RandomSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    test_data = TensorDataset(texts_test, atten_masks_test, labels_test)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return train_dataloader, sentences_length_train, valid_dataloader, sentences_length_valid, test_dataloader, sentences_length_test


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
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plt_save_path)
    plt.show()
    print('The training and valid loss curves have already saved to ==> {plt_save_path}')


def train(model, optimizer, train_loader, valid_loader, train_epochs, save_file_path, device, eval_every=None,
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
        for _, batch in enumerate(notebook.tqdm(train_loader)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids = batch
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
                    for _, batch in enumerate(valid_loader):
                        batch = tuple(t.to(device) for t in batch)
                        input_ids, input_mask, label_ids = batch
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
        for (labels, title, text, titletext), _ in test_loader:
            labels = labels.type(torch.LongTensor).to(device)
            titletext = titletext.type(torch.LongTensor).to(device)
            loss, _ = model(titletext, labels)
            y_pred.extend(torch.argmax(loss, 1).tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, label=[1, 0], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='d')
    ax.set_title()
    ax.set_xlabel()
    ax.set_ylabel()
    ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax.yaxis.set_ticklabels(['FAKE', 'REAL'])
