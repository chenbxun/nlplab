# coding=gb2312
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class Sentence(Dataset):# 自定义数据集类
    def __init__(self, x, y, batch_size=10):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        assert len(self.x[idx]) == len(self.y[idx])
        return self.x[idx], self.y[idx]

    # 默认情况下，如果不提供collate_fn，PyTorch会使用一个内置的函数来处理批次的创建。
    # 这个默认的collate_fn能够处理大多数情况，它会将数据堆叠成一个新的维度（批次维度），并将NumPy数组和Python数值转换为PyTorch张量。
    # 然而，在某些情况下，数据样本可能无法直接堆叠，例如，当数据元素的长度不同时。
    # 在这种情况下，就需要自定义collate_fn来处理数据的堆叠。例如，对于序列数据，可能需要在堆叠之前对其进行填充以确保所有序列具有相同的长度。

    @staticmethod # 静态方法的主要特点是它们与类相关联，但是它们不访问类的任何属性或方法。这使得静态方法成为封装独立功能的理想选择，因为它们不依赖于类的状态。
    def collate_fn(train_data): 
        # train_data: [batch_size, (input_sequence, target_sequence)]
        train_data.sort(key=lambda data: len(data[0]), reverse=True) # 将批次中的样本按输入序列长度从大到小排序
        data_length = [len(data[0]) for data in train_data]
        data_x = [torch.LongTensor(data[0]) for data in train_data]
        data_y = [torch.LongTensor(data[1]) for data in train_data]
        mask = [torch.ones(l, dtype=torch.uint8) for l in data_length]
        data_x = pad_sequence(data_x, batch_first=True, padding_value=0) # 对输入的一批序列进行填充（padding），使得这些序列在指定维度上具有相同的长度
        data_y = pad_sequence(data_y, batch_first=True, padding_value=0)
        mask = pad_sequence(mask, batch_first=True, padding_value=0)
        return data_x, data_y, mask, data_length


if __name__ == '__main__':
    # test
    with open('./data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    train_dataloader = DataLoader(Sentence(x_train, y_train), batch_size=10, shuffle=True, collate_fn=Sentence.collate_fn)

    for input, label, mask, length in train_dataloader: # 返回堆叠后的数据
        print(input, label, mask, length)
        break