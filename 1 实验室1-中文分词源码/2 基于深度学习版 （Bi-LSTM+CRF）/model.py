import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertTokenizer

class CWS(nn.Module):

    def __init__(self, tag2id, hidden_dim, num_layers=2, bert_model_name='bert-base-chinese'):
        # tag2id: �ӱ�ǩ��Ψһ����ID��ӳ���ֵ䣬���ڽ��ַ�����ʽ�ı�ǩת��Ϊ�������ܹ������������ʽ
        super(CWS, self).__init__()
        self.hidden_dim = hidden_dim
        self.tag2id = tag2id
        self.tagset_size = len(tag2id) # ��ǩ�ռ�Ĵ�С�������п��ܱ�ǩ������

        # ����Ԥѵ���� BERT ģ�ͺͷִ���
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # ���� BERT �Ĳ�������ѡ��
        for param in self.bert.parameters():
            param.requires_grad = False  # �����΢�� BERT����������Ϊ True

        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim // 2, num_layers=num_layers,
                            bidirectional=True, batch_first=True) # ˫�� LSTM�����ڲ�׽��������Ϣ

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size) # ���Բ㣬�� LSTM ���ת��Ϊ��ǩ����

        self.crf = CRF(4, batch_first=True) # CRF �㣬���ڼ�����ʧ��Ԥ��

        # CRF�����ҪĿ�����ڱ�ǩ����֮������ȫ��Լ����ȷ��������ı�ǩ���з�������ѧ����
        # ��ͨ���������п��ܱ�ǩ���еĸ��ʣ���ѡ��������ı�ǩ������Ϊ���ս��
        # ��Ȼ���Բ��Ѿ�Ϊÿ��ʱ�䲽�����˱�ǩ���ʷֲ�������Щ�ֲ��Ƕ����ģ������˱�ǩ֮���������ϵ��(�򵥡���Ч���ʺϳ���Ԥ��)
        # �����ķִ������У���ǩ֮������ϸ��˳���������ϵ�����磺
        # B ����ֻ�ܽ� M �� E��
        # ...
        # �������ʹ�����Բ�Ԥ�⣬���ܻ����ɲ����Ϲ���ı�ǩ���У����� [B, S, M]��
        # CRFͨ����ģ��ǩ֮���ת�Ƹ��ʣ��ܹ��������ֲ�����������

        # ���Բ�Ԥ�����ÿ��ʱ�䲽�ϵľֲ����ű�ǩ����ȫ�����ŵı�ǩ���в�һ���ɾֲ�������ɡ�
        # CRFͨ�������п��ܵı�ǩ���н������֣�ѡ������÷���ߵ����У��Ӷ���֤��������ȫ�����š�

    def init_hidden(self, batch_size, device):
        # ע�⣺num_layers �� bidirectional ����������״̬����״
        return (torch.randn(2*2, batch_size, self.hidden_dim // 2, device=device),
                torch.randn(2*2, batch_size, self.hidden_dim // 2, device=device))

    def _get_lstm_features(self, input_ids, attention_mask, length):
        # ʹ�� BERT ��ȡ���ӵ�������Ƕ��
        with torch.no_grad():  # ��������� BERT ����������Ҫ�ݶȼ���
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embeds = bert_outputs.last_hidden_state  # (batch_size, seq_len, bert_hidden_size)

        # LSTM ���봦��
        embeds = pack_padded_sequence(embeds, length, batch_first=True, enforce_sorted=False)

        # LSTM forward
        self.hidden = self.init_hidden(input_ids.size(0), input_ids.device)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, input_ids, tags, mask, attention_mask, length):
        # length ӦΪԭʼ���ȣ�DataLoader���ص�original_lengths��
        emissions = self._get_lstm_features(input_ids, attention_mask, length)
        loss = -self.crf(emissions, tags, mask, reduction='mean')
        # BERT+LSTM���������[CLS]/[SEP]λ�õķ����������CRF��Ҫͨ��mask�ų���䲿�֣�[PAD]��
        return loss

    def infer(self, input_ids, mask, attention_mask, length):
        emissions = self._get_lstm_features(input_ids, attention_mask, length)
        return self.crf.decode(emissions, mask)
