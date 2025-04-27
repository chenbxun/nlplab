import codecs
from sklearn.model_selection import train_test_split
import pickle
from transformers import BertTokenizer

INPUT_DATA = "train.txt"
SAVE_PATH = "./datasave.pkl"
id2tag = ['B', 'M', 'E', 'S']  # B：分词头部 M：分词词中 E：分词词尾 S：独立成词
tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
# word2id = {}
# id2word = []

# 初始化BERT分词器，强制单字切分（关键修改）
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_basic_tokenize=True)
# 模型需要提前下载，使用 HF_ENDPOINT 国内镜像：export HF_ENDPOINT=https://hf-mirror.com

def getList(input_str):
    '''
    单个分词转换为tag序列
    :param input_str: 单个分词
    :return: tag序列
    '''
    outpout_str = []
    if len(input_str) == 1:
        outpout_str.append(tag2id['S'])
    elif len(input_str) == 2:
        outpout_str = [tag2id['B'], tag2id['E']]
    else:
        M_num = len(input_str) - 2
        M_list = [tag2id['M']] * M_num
        outpout_str.append(tag2id['B'])
        outpout_str.extend(M_list)
        outpout_str.append(tag2id['E'])
    return outpout_str


def handle_data():
    '''
    处理数据，并保存至savepath
    :return:
    '''
    x_data = []
    y_data = []
    # wordnum = 0

    with open(INPUT_DATA, 'r', encoding="utf-8") as ifp:
        for line in ifp: # 对每一行文本
            line = line.strip()
            if not line:
                continue

            # 关键修改：先去除所有空格（保留分词信息用于标签生成）
            raw_text = line.replace(" ", "")  # 去除空格后的原始文本
            words = line.split()             # 分词列表（用于生成标签）

            # 生成input_ids（强制单字编码）
            input_ids = []
            for char in raw_text:
                # 直接调用convert_tokens_to_ids避免tokenizer合并字符
                token_id = tokenizer.convert_tokens_to_ids(char)
                input_ids.append(token_id)

            # line_x = [] # 保存汉字id序列
            # for i in range(len(line)):
            #     if line[i] == " ":
            #         continue
            #     if (line[i] in id2word): # 未建立从字到id的映射
            #         line_x.append(word2id[line[i]])
            #     else:
            #         id2word.append(line[i])
            #         word2id[line[i]] = wordnum
            #         line_x.append(wordnum)
            #         wordnum = wordnum + 1
            # x_data.append(line_x) # 保存的实际上是字id序列不是字本身
            
            # 生成标签（基于原始分词信息）
            line_y = []
            for word in words:
                line_y.extend(getList(word))

            # 最终检查
            if len(input_ids) != len(line_y):
                print(f"错误：长度不一致！")
                print(f"原始文本: {line}")
                print(f"去空格文本: {raw_text} (字符数: {len(raw_text)})")
                print(f"字符列表: {list(raw_text)}")
                print(f"input_ids: {input_ids} (长度: {len(input_ids)})")
                print(f"标签: {line_y} (长度: {len(line_y)})")
                print(f"BERT tokenizer切分: {tokenizer.tokenize(raw_text)}")
                raise ValueError("数据对齐失败！")

            x_data.append(input_ids)
            y_data.append(line_y)


    # print(x_data[0])
    # print([id2word[i] for i in x_data[0]])
    # print(y_data[0])
    # print([id2tag[i] for i in y_data[0]])
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=43)
    with open(SAVE_PATH, 'wb') as outp:
        # pickle.dump(word2id, outp)
        # pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)


if __name__ == "__main__":
    handle_data()
