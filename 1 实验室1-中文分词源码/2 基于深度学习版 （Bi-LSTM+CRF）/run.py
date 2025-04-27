#coding=gb2312
import pickle
import logging
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import CWS
from dataloader import Sentence

def get_param():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--bert_model', default='bert-base-chinese')
    parser.add_argument('--freeze_bert', default=True)  # 是否冻结BERT参数
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--cuda', action='store_true', default=False)
    return parser.parse_args()


def set_logger():
    log_file = os.path.join('save', 'log.txt')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m%d %H:%M:%S',
        filename=log_file,
        filemode='w',
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def entity_split(x, y, id2tag, entities, cur):
    start, end = -1, -1
    for j in range(len(x)):
        if id2tag[y[j]] == 'B':
            start = cur + j
        elif id2tag[y[j]] == 'M' and start != -1:
            continue
        elif id2tag[y[j]] == 'E' and start != -1:
            end = cur + j
            entities.add((start, end))
            start, end = -1, -1
        elif id2tag[y[j]] == 'S':
            entities.add((cur + j, cur + j))
            start, end = -1, -1
        else:
            start, end = -1, -1


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()

    with open('data/datasave.pkl', 'rb') as inp:
        # word2id = pickle.load(inp)
        # id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    # model = CWS(len(word2id), tag2id, args.embedding_dim, args.hidden_dim)
    model = CWS(tag2id, args.hidden_dim, bert_model_name=args.bert_model)
    if args.freeze_bert:
        for param in model.bert.parameters():
            param.requires_grad = False

    if use_cuda:
        model = model.cuda()
    for name, param in model.named_parameters():
        logging.debug('%s: %s, require_grad=%s' % (name, str(param.shape), str(param.requires_grad)))

    optimizer = Adam(model.parameters(), lr=args.lr)

    train_data = DataLoader(
        dataset=Sentence(x_train, y_train),
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=6
    )

    test_data = DataLoader(
        dataset=Sentence(x_test[:1000], y_test[:1000]),
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=6
    )

    for epoch in range(args.max_epoch):
        step = 0
        log = []
        # for sentence, label, mask, length in train_data:
        for batch in train_data:  # 改为直接取batch
            # if use_cuda:
            #     sentence = sentence.cuda()
            #     label = label.cuda()
            #     mask = mask.cuda()
            if use_cuda:
                batch = {
                'input_ids': batch['input_ids'].cuda(),
                'labels': batch['labels'].cuda(),
                'attention_mask': batch['attention_mask'].cuda(),
                'lengths': batch['lengths']  # 保持CPU
            }

            # forward
            # loss = model(sentence, label, mask, length)
            loss = model(
                input_ids=batch['input_ids'],
                tags=batch['labels'],
                mask=batch['attention_mask'],  # 作为CRF的mask
                attention_mask=batch['attention_mask'],  # 给BERT的mask
                length=batch['lengths']  # 原始长度
            )
            log.append(loss.item())

            # 反向传播保持不变
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 100 == 0:
                logging.debug('epoch %d-step %d loss: %f' % (epoch, step, sum(log)/len(log)))
                log = []

        # test
        entity_predict = set()
        entity_label = set()
        with torch.no_grad():
            model.eval()
            cur = 0
            # for sentence, label, mask, length in test_data:
            #     if use_cuda:
            #         sentence = sentence.cuda()
            #         label = label.cuda()
            #         mask = mask.cuda()
            for batch in test_data:
                if use_cuda:
                    batch = {
                    'input_ids': batch['input_ids'].cuda(),
                    'labels': batch['labels'].cuda(),
                    'attention_mask': batch['attention_mask'].cuda(),
                    'lengths': batch['lengths']  # 保持CPU
                }

                # predict = model.infer(sentence, mask, length)
                predict = model.infer(
                    input_ids=batch['input_ids'],
                    mask=batch['attention_mask'],
                    attention_mask=batch['attention_mask'],
                    length=batch['lengths']
                )

                # for i in range(len(length)):
                #     entity_split(sentence[i, :length[i]], predict[i], id2tag, entity_predict, cur)
                #     entity_split(sentence[i, :length[i]], label[i, :length[i]], id2tag, entity_label, cur)
                #     cur += length[i]
                # 实体抽取需调整（注意跳过特殊token）
                for i in range(len(batch['lengths'])):
                    # 取有效部分（跳过[CLS]和[SEP]）
                    valid_tokens = batch['input_ids'][i, 1:1+batch['lengths'][i]]  # 跳过[CLS]
                    valid_predict = predict[i][1:1+batch['lengths'][i]]  # 对应预测标签
                    valid_label = batch['labels'][i, 1:1+batch['lengths'][i]]  # 跳过[CLS]标签
                    
                    entity_split(valid_tokens, valid_predict, id2tag, entity_predict, cur)
                    entity_split(valid_tokens, valid_label, id2tag, entity_label, cur)
                    cur += batch['lengths'][i]

            # 评估指标计算保持不变
            right_predict = [i for i in entity_predict if i in entity_label]
            if len(right_predict) != 0:
                precision = float(len(right_predict)) / len(entity_predict)
                recall = float(len(right_predict)) / len(entity_label)
                logging.info("precision: %f" % precision)
                logging.info("recall: %f" % recall)
                logging.info("fscore: %f" % ((2 * precision * recall) / (precision + recall)))
            else:
                logging.info("precision: 0")
                logging.info("recall: 0")
                logging.info("fscore: 0")
            model.train()

        path_name = "./save/model_epoch" + str(epoch) + ".pkl"
        torch.save(model, path_name)
        logging.info("model has been saved in  %s" % path_name)


if __name__ == '__main__':
    set_logger()
    main(get_param())
# 数据加载改为字典形式，通过key获取不同字段
# 前向传播和推理调用时参数名需与模型定义严格匹配
# 实体抽取时需跳过[CLS]和[SEP]对应的位置
# 评估指标计算逻辑不变，但输入已是处理后的有效部分
