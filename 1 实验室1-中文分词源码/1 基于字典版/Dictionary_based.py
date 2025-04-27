class Tokenizer(object):
    def __init__(self, words, max_len):
        self.words = words
        self.max_len = max_len

    def fmm_split(self, text):
        '''
        正向最大匹配分词算法
        :param text: 待分词字符串
        :return: 分词结果，以list形式存放，每个元素为分出的词
        '''
        result = []
        while text != '':
            # 首先按最大长度切分
            cur_len = min(self.max_len, len(text))
            # 若为单字则直接退出
            while cur_len > 1:
                W = text[: cur_len]
                # 匹配成功
                if W in self.words:
                    break
                cur_len -= 1

            result.append(text[: cur_len])
            text = text[cur_len: ]
        return result


    def rmm_split(self, text):
        '''
        逆向最大匹配分词算法
        :param text: 待分词字符串
        :return: 分词结果，以list形式存放，每个元素为分出的词
        '''
        result = []
        while text != '':
            # 首先按最大长度切分
            cur_len = min(self.max_len, len(text))
            # 若为单字则直接退出
            while cur_len > 1:
                # 从倒数第len个字到最后
                W = text[-cur_len: ]
                # 匹配成功
                if W in self.words:
                    break
                cur_len -= 1

            result.append(text[-cur_len: ])
            text = text[: -cur_len]
        result.reverse()
        return result

    def bimm_split(self, text):
        '''
        双向最大匹配分词算法
        :param text: 待分词字符串
        :return: 分词结果，以list形式存放，每个元素为分出的词
        '''
        fmm_res = self.fmm_split(text)
        rmm_res = self.rmm_split(text)

        # 正反向分词结果词数不同，取分词数量少的那个
        if len(fmm_res) != len(rmm_res):
            return fmm_res if len(fmm_res) < len(rmm_res) else rmm_res
        
        # 分词结果相同，没有歧义，返回任意一个
        if all(fmm_res[i] == rmm_res[i] for i in range(len(fmm_res))):
            return fmm_res
        # 分词结果不同，返回其中单字数量较少的那个
        else:
            fmm_single_count = sum(1 for word in fmm_res if len(word) == 1)
            rmm_single_count = sum(1 for word in rmm_res if len(word) == 1)
            return fmm_res if fmm_single_count < rmm_single_count else rmm_res


def load_dict(path):
    tmp = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split(' ')[0]
            tmp.add(word)
    return tmp

if __name__ == '__main__':
    # 加载词典文件，并返回一个包含所有词语的集合
    # 打开文件并逐行读取。
    # 每行按空格分割，取第一个字段作为词语。
    # 将所有词语存储在一个集合中，便于快速查找。
    words = load_dict('dict.txt')
    max_len = max(map(len, [word for word in words]))# 对列表中的每个词语调用 len 函数，返回一个包含每个词语长度的迭代器

    # test
    tokenizer = Tokenizer(words, max_len)
    texts = [
        '研究生命的起源',
        '无线电法国别研究',
        '人要是行，干一行行一行，一行行行行行，行行行干哪行都行。',
        '实验任务包括基础任务与选做任务。基础任务中需实现基于词典和基于统计的中文分词算法，完成后可获得实验课程基础分。选做任务中需要对基础任务中的分词器进行优化，选做部分的分数通过分词器在测试集上的表现决定。最终提交的实验报告中应包括基础任务完成情况与选做任务中采取的优化措施。',
        '雨是最寻常的，一下就是三两天。可别恼，看，像牛毛，像花针，像细丝，密密地斜织着，人家屋顶上全笼着一层薄烟。树叶儿却绿得发亮，小草儿也青得逼你的眼。傍晚时候，上灯了，一点点黄晕的光，烘托出一片安静而和平的夜。在乡下，小路上，石桥边，有撑起伞慢慢走着的人，地里还有工作的农民，披着蓑戴着笠。他们的房屋，稀稀疏疏的，在雨里静默着。'
    ]
    for text in texts:
        # 前向最大匹配
        print('前向最大匹配:', '/'.join(tokenizer.fmm_split(text)))
        # 后向最大匹配
        print('后向最大匹配:', '/'.join(tokenizer.rmm_split(text)))
        # 双向最大匹配
        print('双向最大匹配:', '/'.join(tokenizer.bimm_split(text)))
        print('')

    with open("test_data.txt", "r") as f:
        for line in f:
            ' '.join(tokenizer.bimm_split(line))