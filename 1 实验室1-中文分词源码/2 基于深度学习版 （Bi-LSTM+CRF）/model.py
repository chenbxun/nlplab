import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CWS(nn.Module):

    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim):
        # tag2id: 从标签到唯一整数ID的映射字典，用于将字符串形式的标签转换为神经网络能够处理的数字形式
        super(CWS, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tagset_size = len(tag2id) # 标签空间的大小，即所有可能标签的数量

        self.word_embeds = nn.Embedding(vocab_size + 1, embedding_dim)
        # 词嵌入层，其作用是将输入的离散词汇索引（通常是整数）映射为连续的稠密向量表示
        # 不限制输入张量的形状，它会逐元素地进行替换操作
        # 嵌入矩阵的参数在模型训练过程中通过反向传播不断更新

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True) # 双向 LSTM，用于捕捉上下文信息
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size) # 线性层，将 LSTM 输出转换为标签概率

        self.crf = CRF(4, batch_first=True) # CRF 层，用于计算损失和预测

        # CRF层的主要目的是在标签序列之间引入全局约束，确保解码出的标签序列符合语言学规则。
        # 它通过计算所有可能标签序列的概率，并选择概率最大的标签序列作为最终结果
        # 虽然线性层已经为每个时间步生成了标签概率分布，但这些分布是独立的，忽略了标签之间的依赖关系。(简单、高效，适合初步预测)
        # 在中文分词任务中，标签之间存在严格的顺序和依赖关系，例如：
        # B 后面只能接 M 或 E。
        # ...
        # 如果仅仅使用线性层预测，可能会生成不符合规则的标签序列，比如 [B, S, M]。
        # CRF通过建模标签之间的转移概率，能够避免这种不合理的情况。

        # 线性层预测的是每个时间步上的局部最优标签，但全局最优的标签序列不一定由局部最优组成。
        # CRF通过对所有可能的标签序列进行评分，选择整体得分最高的序列，从而保证解码结果的全局最优。

    def init_hidden(self, batch_size, device):
        return (torch.randn(2, batch_size, self.hidden_dim // 2, device=device),
                torch.randn(2, batch_size, self.hidden_dim // 2, device=device))

    def _get_lstm_features(self, sentence, length):
        # sentence: 一批输入句子的词索引形式，形状通常是 (batch_size, seq_len)
        batch_size, seq_len = sentence.size(0), sentence.size(1)

        # idx->embedding
        embeds = self.word_embeds(sentence.view(-1)).reshape(batch_size, seq_len, -1)
        # 将输入的二维张量 (batch_size, seq_len) 展平为一维张量 (batch_size * seq_len,)。这是为了方便批量处理所有词索引。
        embeds = pack_padded_sequence(embeds, length, batch_first=True)
        # 我们不希望其填充的pad数据（一般为0）进入GRU或是LSTM模块，一是浪费资源，二是可能造成句子表征不准确。
        # 所以pack_padded_sequence 类应运而生。主要是对填充过的数据进行压缩。

        # LSTM forward
        self.hidden = self.init_hidden(batch_size, sentence.device)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence, tags, mask, length):
        emissions = self._get_lstm_features(sentence, length) #  LSTM 层生成的发射分数，形状为 (batch_size, seq_len, tagset_size)
        loss = -self.crf(emissions, tags, mask, reduction='mean')
        # CRF层本身并不直接接收原始的句子（x）作为输入，而是接收经过模型处理后的发射分数（Emission Scores）和标注序列（Ground Truth Tags, y）
        return loss

    def infer(self, sentence, mask, length):
        emissions = self._get_lstm_features(sentence, length)
        return self.crf.decode(emissions, mask)
