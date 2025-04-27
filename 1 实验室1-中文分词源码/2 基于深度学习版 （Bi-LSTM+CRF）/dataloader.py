# coding=gb2312
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class Sentence(Dataset):# 自定义数据集类
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # self.batch_size = batch_size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        assert len(self.x[idx]) == len(self.y[idx])
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

    # 默认情况下，如果不提供collate_fn，PyTorch会使用一个内置的函数来处理批次的创建。
    # 这个默认的collate_fn能够处理大多数情况，它会将数据堆叠成一个新的维度（批次维度），并将NumPy数组和Python数值转换为PyTorch张量。
    # 然而，在某些情况下，数据样本可能无法直接堆叠，例如，当数据元素的长度不同时。
    # 在这种情况下，就需要自定义collate_fn来处理数据的堆叠。例如，对于序列数据，可能需要在堆叠之前对其进行填充以确保所有序列具有相同的长度。

    @staticmethod # 静态方法的主要特点是它们与类相关联，但是它们不访问类的任何属性或方法。这使得静态方法成为封装独立功能的理想选择，因为它们不依赖于类的状态。
    def collate_fn(batch):
        # 按长度降序排序
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        input_ids = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        lengths = torch.tensor([len(x) for x in input_ids]) # 原始长度（不含特殊token）
        
        # 添加[CLS]和[SEP]特殊token（关键修改）
        input_ids = [torch.cat([torch.tensor([101]), x, torch.tensor([102])]) for x in input_ids]
        labels = [torch.cat([torch.tensor([-100]), y, torch.tensor([-100])]) for y in labels]  # 特殊token标签设为-100
        
        # 生成attention_mask（覆盖所有token，包括特殊token）
        attention_mask = [torch.ones(len(x), dtype=torch.long) for x in input_ids]
        
        # 填充
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # 填充标签用-100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'lengths': lengths
        }

# 测试代码调整
if __name__ == '__main__':
    with open('./data/datasave.pkl', 'rb') as inp:
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
    
    print(f"首个训练样本 - 原始输入长度: {len(x_train[0])}, 标签长度: {len(y_train[0])}")

    # for x, y in zip(x_train, y_train):
    #     if len(x) != len(y):
    #         print(f"长度不一致！x: {len(x)}, y: {len(y)}")
    #         print(f"x: {x}")
    #         print(f"y: {y}")
    #         break

    train_dataloader = DataLoader(
        Sentence(x_train, y_train),
        batch_size=32,
        collate_fn=Sentence.collate_fn
    )
    
    # 获取第一个batch并验证
    batch = next(iter(train_dataloader))
    print("\n=== Batch验证 ===")
    
    # 1. 检查各字段形状是否一致
    print(f"input_ids形状: {batch['input_ids'].shape}")  # 应为 [batch, max_len+2]
    print(f"labels形状:    {batch['labels'].shape}")     # 应与input_ids相同
    print(f"attention_mask形状: {batch['attention_mask'].shape}")
    print(f"lengths内容:   {batch['lengths']}")         # 应为原始长度（不含特殊token）
    
    # 2. 检查特殊token是否正确添加
    print("\n首个样本的token和标签：")
    print("input_ids[0]:", batch['input_ids'][0])
    print("labels[0]:   ", batch['labels'][0])
    
    # 3. 验证attention_mask
    print("\nattention_mask验证：")
    print("mask[0]:     ", batch['attention_mask'][0])
    print("实际有效长度:", batch['attention_mask'][0].sum().item())  # 应等于 length+2
    
    # 4. 检查填充是否正确
    print("\n填充位置验证：")
    pad_positions = torch.where(batch['input_ids'][0] == 0)[0]
    print("填充位置索引:", pad_positions)
    print("对应标签值:  ", batch['labels'][0][pad_positions])  # 应为全-100
    
    # 5. 长度一致性检查
    assert batch['input_ids'].shape == batch['labels'].shape == batch['attention_mask'].shape, "形状不一致！"
    assert (batch['attention_mask'].sum(dim=1) == batch['lengths'] + 2).all(), "attention_mask长度错误！"
    assert (batch['labels'][:, 0] == -100).all() and (batch['labels'][:, -1] == -100).all(), "特殊token标签未设置-100！"
    
    print("\n所有检查通过！DataLoader输出符合预期。")