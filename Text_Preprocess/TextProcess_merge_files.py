from os import listdir
import re
import string

# 将多个文件合并为一个文件

"""
###################### Using Guide ##############################

path_row_neg = ""
path_row_pos = ""

path_neg_processed = ""
path_pos_processed = ""


# preprocess for input
TextProcess_for_Input(path_neg_comments_row, path_pos_comments_row, path_neg_comments_processed, path_pos_comments_processed )

##########################################################
"""


def merge_file_txt(path_neg_row,path_pos_row,path_neg_processed,path_pos_processed):

    # 加载一篇txt文档
    def load_doc(filename):

        # 只读方式打开一个文档
        file = open(filename, 'r', encoding='utf-8')

        # 读取文档所有内容，text为字符串类型
        text = file.read()

        # 关闭连接
        file.close()

        return text

    # 清洗数据
    def clean_doc(doc):

        tokens = doc.split(' ')

        # 移除文本中的标点符号
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        tokens = [re_punc.sub('', w) for w in tokens]

        # 去除回车符\r和换行符\n和tab键
        tokens = [word.replace('\n', '').replace('\r', '').replace('\t', '') for word in tokens]

        # 去除非ACCII字符
        # tokens = [word for word in tokens if word.isascii()]

        return tokens

    # one comment convert to one line
    def doc_to_line(filename):
        # 加载文件
        doc = load_doc(filename)

        # 调用清洗函数，去除标点符号、数字等
        tokens = clean_doc(doc)

        return ' '.join(tokens)


    def process_docs(directory):
        lines = []
        # 依次读取folder中的所有file
        for filename in listdir(directory):
            # 跳过没有以正常的扩展名(txt)结束的文件
            if not filename.endswith('.txt'):
                continue

            # 拼接完整路径
            path = directory + '/' + filename
            # 调用之前的加载文件函数，和清洗文件函数
            line = doc_to_line(path)

            # 添加到list型的lines中
            lines.append(line)

        print(lines[0])

        return lines

    def save_list(lines, path):
        # 以\n分开list中的每个评论
        data = '\n'.join(lines)

        file = open(path, 'w', encoding='utf-8')  # 'w'改成'a'可分别处理train and test set至一个文件中

        # 将一类中的所有评论写入到一个文件中
        file.write(data)
        file.close()

    negative_lines = process_docs(path_neg_row)
    positive_lines = process_docs(path_pos_row)

    # saved
    save_list(negative_lines, path_neg_processed)
    save_list(positive_lines, path_pos_processed)
