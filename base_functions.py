# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import tushare as ts
from datetime import datetime
import re
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer


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
