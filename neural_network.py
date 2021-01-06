from pytorch_transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class FinBertTransformer(nn.Module):
    def __init__(self, pretrain_path='./pretrained_models/FinBERT_L-12_H-768_A-12_pytorch',
                 sents_num=50, sent_hidden=[192, 48], nhead=8, num_layers=6, news_hidden=[48, 1]):
        super(FinBertTransformer, self).__init__()
        self.sents_num = sents_num
        self.sent_hidden = sent_hidden
        self.pretrain_model = BertModel.from_pretrained(pretrain_path)
        #transform_encoderlayer = nn.TransformerEncoderLayer(d_model=768, nhead=nhead)
        #self.transformer_encoder = nn.TransformerEncoder(transform_encoderlayer, num_layers=num_layers)
        self.sh_fc1 = nn.Linear(768, sent_hidden[0])
        self.sh_dropout1 = nn.Dropout(0.2)
        self.sh_fc2 = nn.Linear(sent_hidden[0], sent_hidden[1])
        self.sh_dropout2 = nn.Dropout(0.2)
        self.nh_fc1 = nn.Linear(sent_hidden[1] * sents_num, news_hidden[0])
        self.nh_dropout1 = nn.Dropout(0.2)
        self.nh_fc2 = nn.Linear(news_hidden[0], news_hidden[-1])
        self.nh_dropout2 = nn.Dropout(0.2)
        if news_hidden[-1] > 1:
            self.act = F.softmax
        else:
            self.act = torch.tanh

    def forward(self, x, mask=None):
        x = x.view(-1, x.shape[-1])
        if mask is not None:
            mask = mask.view(-1, mask.shape[-1])
        last_hidden, pooler_output = self.pretrain_model(
            x, attention_mask=mask)
        x = pooler_output.view(-1, self.sents_num, 768)
        x = F.leaky_relu(self.sh_fc1(self.sh_dropout1(x)))
        x = F.leaky_relu(self.sh_fc2(self.sh_dropout2(x)))
        #x = self.transformer_encoder(x)
        x = x.view(-1, self.sents_num * self.sent_hidden[-1])
        x = F.leaky_relu(self.nh_fc1(self.nh_dropout1(x)))
        x = self.act(self.nh_fc2(self.nh_dropout2(x)))
        return x


