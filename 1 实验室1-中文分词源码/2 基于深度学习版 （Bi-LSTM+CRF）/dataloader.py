# coding=gb2312
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class Sentence(Dataset):# �Զ������ݼ���
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # self.batch_size = batch_size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        assert len(self.x[idx]) == len(self.y[idx])
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

    # Ĭ������£�������ṩcollate_fn��PyTorch��ʹ��һ�����õĺ������������εĴ�����
    # ���Ĭ�ϵ�collate_fn�ܹ�����������������Ὣ���ݶѵ���һ���µ�ά�ȣ�����ά�ȣ�������NumPy�����Python��ֵת��ΪPyTorch������
    # Ȼ������ĳЩ����£��������������޷�ֱ�Ӷѵ������磬������Ԫ�صĳ��Ȳ�ͬʱ��
    # ����������£�����Ҫ�Զ���collate_fn���������ݵĶѵ������磬�����������ݣ�������Ҫ�ڶѵ�֮ǰ������������ȷ���������о�����ͬ�ĳ��ȡ�

    @staticmethod # ��̬��������Ҫ�ص�������������������������ǲ���������κ����Ի򷽷�����ʹ�þ�̬������Ϊ��װ�������ܵ�����ѡ����Ϊ���ǲ����������״̬��
    def collate_fn(batch):
        # �����Ƚ�������
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        input_ids = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        lengths = torch.tensor([len(x) for x in input_ids]) # ԭʼ���ȣ���������token��
        
        # ���[CLS]��[SEP]����token���ؼ��޸ģ�
        input_ids = [torch.cat([torch.tensor([101]), x, torch.tensor([102])]) for x in input_ids]
        labels = [torch.cat([torch.tensor([-100]), y, torch.tensor([-100])]) for y in labels]  # ����token��ǩ��Ϊ-100
        
        # ����attention_mask����������token����������token��
        attention_mask = [torch.ones(len(x), dtype=torch.long) for x in input_ids]
        
        # ���
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # ����ǩ��-100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'lengths': lengths
        }

# ���Դ������
if __name__ == '__main__':
    with open('./data/datasave.pkl', 'rb') as inp:
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
    
    print(f"�׸�ѵ������ - ԭʼ���볤��: {len(x_train[0])}, ��ǩ����: {len(y_train[0])}")

    # for x, y in zip(x_train, y_train):
    #     if len(x) != len(y):
    #         print(f"���Ȳ�һ�£�x: {len(x)}, y: {len(y)}")
    #         print(f"x: {x}")
    #         print(f"y: {y}")
    #         break

    train_dataloader = DataLoader(
        Sentence(x_train, y_train),
        batch_size=32,
        collate_fn=Sentence.collate_fn
    )
    
    # ��ȡ��һ��batch����֤
    batch = next(iter(train_dataloader))
    print("\n=== Batch��֤ ===")
    
    # 1. �����ֶ���״�Ƿ�һ��
    print(f"input_ids��״: {batch['input_ids'].shape}")  # ӦΪ [batch, max_len+2]
    print(f"labels��״:    {batch['labels'].shape}")     # Ӧ��input_ids��ͬ
    print(f"attention_mask��״: {batch['attention_mask'].shape}")
    print(f"lengths����:   {batch['lengths']}")         # ӦΪԭʼ���ȣ���������token��
    
    # 2. �������token�Ƿ���ȷ���
    print("\n�׸�������token�ͱ�ǩ��")
    print("input_ids[0]:", batch['input_ids'][0])
    print("labels[0]:   ", batch['labels'][0])
    
    # 3. ��֤attention_mask
    print("\nattention_mask��֤��")
    print("mask[0]:     ", batch['attention_mask'][0])
    print("ʵ����Ч����:", batch['attention_mask'][0].sum().item())  # Ӧ���� length+2
    
    # 4. �������Ƿ���ȷ
    print("\n���λ����֤��")
    pad_positions = torch.where(batch['input_ids'][0] == 0)[0]
    print("���λ������:", pad_positions)
    print("��Ӧ��ǩֵ:  ", batch['labels'][0][pad_positions])  # ӦΪȫ-100
    
    # 5. ����һ���Լ��
    assert batch['input_ids'].shape == batch['labels'].shape == batch['attention_mask'].shape, "��״��һ�£�"
    assert (batch['attention_mask'].sum(dim=1) == batch['lengths'] + 2).all(), "attention_mask���ȴ���"
    assert (batch['labels'][:, 0] == -100).all() and (batch['labels'][:, -1] == -100).all(), "����token��ǩδ����-100��"
    
    print("\n���м��ͨ����DataLoader�������Ԥ�ڡ�")