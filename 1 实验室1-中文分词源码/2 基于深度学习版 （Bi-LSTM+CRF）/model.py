import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CWS(nn.Module):

    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim):
        # tag2id: �ӱ�ǩ��Ψһ����ID��ӳ���ֵ䣬���ڽ��ַ�����ʽ�ı�ǩת��Ϊ�������ܹ������������ʽ
        super(CWS, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tagset_size = len(tag2id) # ��ǩ�ռ�Ĵ�С�������п��ܱ�ǩ������

        self.word_embeds = nn.Embedding(vocab_size + 1, embedding_dim)
        # ��Ƕ��㣬�������ǽ��������ɢ�ʻ�������ͨ����������ӳ��Ϊ�����ĳ���������ʾ
        # ������������������״��������Ԫ�صؽ����滻����
        # Ƕ�����Ĳ�����ģ��ѵ��������ͨ�����򴫲����ϸ���

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1,
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
        return (torch.randn(2, batch_size, self.hidden_dim // 2, device=device),
                torch.randn(2, batch_size, self.hidden_dim // 2, device=device))

    def _get_lstm_features(self, sentence, length):
        # sentence: һ��������ӵĴ�������ʽ����״ͨ���� (batch_size, seq_len)
        batch_size, seq_len = sentence.size(0), sentence.size(1)

        # idx->embedding
        embeds = self.word_embeds(sentence.view(-1)).reshape(batch_size, seq_len, -1)
        # ������Ķ�ά���� (batch_size, seq_len) չƽΪһά���� (batch_size * seq_len,)������Ϊ�˷��������������д�������
        embeds = pack_padded_sequence(embeds, length, batch_first=True)
        # ���ǲ�ϣ��������pad���ݣ�һ��Ϊ0������GRU����LSTMģ�飬һ���˷���Դ�����ǿ�����ɾ��ӱ�����׼ȷ��
        # ����pack_padded_sequence ��Ӧ�˶�������Ҫ�Ƕ����������ݽ���ѹ����

        # LSTM forward
        self.hidden = self.init_hidden(batch_size, sentence.device)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence, tags, mask, length):
        emissions = self._get_lstm_features(sentence, length) #  LSTM �����ɵķ����������״Ϊ (batch_size, seq_len, tagset_size)
        loss = -self.crf(emissions, tags, mask, reduction='mean')
        # CRF�㱾����ֱ�ӽ���ԭʼ�ľ��ӣ�x����Ϊ���룬���ǽ��վ���ģ�ʹ����ķ��������Emission Scores���ͱ�ע���У�Ground Truth Tags, y��
        return loss

    def infer(self, sentence, mask, length):
        emissions = self._get_lstm_features(sentence, length)
        return self.crf.decode(emissions, mask)
