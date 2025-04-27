import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertTokenizer

class CWS(nn.Module):

    def __init__(self, tag2id, hidden_dim, num_layers=2, bert_model_name='bert-base-chinese'):
        # tag2id: 从标签到唯一整数ID的映射字典，用于将字符串形式的标签转换为神经网络能够处理的数字形式
        super(CWS, self).__init__()
        self.hidden_dim = hidden_dim
        self.tag2id = tag2id
        self.tagset_size = len(tag2id) # 标签空间的大小，即所有可能标签的数量

        # 加载预训练的 BERT 模型和分词器
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # 冻结 BERT 的参数（可选）
        for param in self.bert.parameters():
            param.requires_grad = False  # 如果想微调 BERT，可以设置为 True

        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim // 2, num_layers=num_layers,
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
        # 注意：num_layers 和 bidirectional 决定了隐藏状态的形状
        return (torch.randn(2*2, batch_size, self.hidden_dim // 2, device=device),
                torch.randn(2*2, batch_size, self.hidden_dim // 2, device=device))

    def _get_lstm_features(self, input_ids, attention_mask, length):
        # 使用 BERT 获取句子的上下文嵌入
        with torch.no_grad():  # 如果冻结了 BERT 参数，则不需要梯度计算
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embeds = bert_outputs.last_hidden_state  # (batch_size, seq_len, bert_hidden_size)

        # LSTM 输入处理
        embeds = pack_padded_sequence(embeds, length, batch_first=True, enforce_sorted=False)

        # LSTM forward
        self.hidden = self.init_hidden(input_ids.size(0), input_ids.device)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, input_ids, tags, mask, attention_mask, length):
        # length 应为原始长度（DataLoader返回的original_lengths）
        emissions = self._get_lstm_features(input_ids, attention_mask, length)
        loss = -self.crf(emissions, tags, mask, reduction='mean')
        # BERT+LSTM的输出包含[CLS]/[SEP]位置的发射分数，但CRF需要通过mask排除填充部分（[PAD]）
        return loss

    def infer(self, input_ids, mask, attention_mask, length):
        emissions = self._get_lstm_features(input_ids, attention_mask, length)
        return self.crf.decode(emissions, mask)
