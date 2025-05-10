import torch
import pickle
from run import entity_split
import thulac

if __name__ == '__main__':
    model = torch.load('save/model_epoch4.pkl', map_location=torch.device('cpu'))
    thu1 = thulac.thulac(seg_only=True)
    entity_predict = set()
    entity_label = set()
    cur_pre = 0
    cur_lab = 0
    predict_output = open('predict_thulac.txt', 'w', encoding='utf-8')
    label_output = open('label_thulac.txt', 'w', encoding='utf-8')

    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    with open('test_data.txt', 'r', encoding='utf-8') as f:
        for test in f:
            test = test.strip()

            x = torch.LongTensor(1, len(test))
            mask = torch.ones_like(x, dtype=torch.uint8)
            length = [len(test)]
            for i in range(len(test)):
                if test[i] in word2id:
                    x[0, i] = word2id[test[i]]
                else:
                    x[0, i] = len(word2id)

            predict = model.infer(x, mask, length)[0]
            for i in range(len(test)):
                print(test[i], end='', file=predict_output)
                if id2tag[predict[i]] in ['E', 'S']:
                    print(' ', end='', file=predict_output)
            print(file=predict_output)

            entity_split(x[0], predict, id2tag, entity_predict, cur_pre)
            cur_pre += len(test)

            # label = jieba.lcut(test)
            label = thu1.cut(test, text=True) # 我 爱 北京 天安门
            for word in label.split(" "): # ['我', '爱', '北京', '天安门']
                entity_label.add((cur_lab, cur_lab + len(word) - 1))
                cur_lab += len(word)
            print(label, file=label_output)
    
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

# precision: 0.829389
# recall: 0.791576
# fscore: 0.810041