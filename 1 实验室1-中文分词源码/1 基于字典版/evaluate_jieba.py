from Dictionary_based import Tokenizer
import jieba

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
    entity_predict = set() # 模型预测的实体的起始和结束位置的集合
    entity_label = set()
    cur_pre = 0
    cur_lab = 0
    predict_output = open('predict_jieba.txt', 'w', encoding='utf-8')
    label_output = open('label_jieba.txt', 'w', encoding='utf-8')
    with open('test_data.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip() # 移除字符串头尾指定的字符（默认为空格或换行符）或字符序列

            predict = tokenizer.bimm_split(line)
            for word in predict:
                entity_predict.add((cur_pre, cur_pre + len(word) - 1))
                cur_pre += len(word)
            print(' '.join(predict), file=predict_output)

            label = jieba.lcut(line)
            for word in label:
                entity_label.add((cur_lab, cur_lab + len(word) - 1))
                cur_lab += len(word)
            print(' '.join(label), file=label_output)

    right_predict = [i for i in entity_predict if i in entity_label]
    if len(right_predict) != 0:
        precision = float(len(right_predict)) / len(entity_predict)
        recall = float(len(right_predict)) / len(entity_label)
        print("precision: %f" % precision)
        print("recall: %f" % recall)
        print("fscore: %f" % ((2 * precision * recall) / (precision + recall)))
    else:
        print("precision: 0")
        print("recall: 0")
        print("fscore: 0")
# precision: 0.904776
# recall: 0.936776
# fscore: 0.920498