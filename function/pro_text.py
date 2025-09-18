from tqdm import trange


def to_text():
    """
    从数据集中提取出该数据集所有标签
    Args:
        task_name: NER/weibo, NER/note4 and so on
        mode: train.
    """
    in_file = "../data/dataset/NER/weibo/test.char.bmes"
    out_file = "../data/dataset/NER/weibo/test.bmes"
    file_old = open(in_file, 'r', encoding='utf-8')
    file_new = open(out_file, 'w', encoding='utf-8')

    # with open(in_file, 'r', encoding='utf-8') as f:
    lines = file_old.readlines()  # 读取所有行
    # new_lines = [line.strip() for line in lines if line.strip()]
    for line in lines:
        if line == '\n':
            line = line.strip("\n")
        # with open(out_file, 'w', encoding='utf-8') as file:  # 重新写入文件
        file_new.write(line)


def set_n():
    in_file = "../data/dataset/NER/weibo/test.bmes"
    out_file = "../data/dataset/NER/weibo/test.bmes"
    file_old = open(in_file, 'r', encoding='utf-8')
    file_new = open(out_file, 'w', encoding='utf-8')
    lines = file_old.readlines()
    length = len(lines)


if __name__ == "__main__":
    to_text()
    set_n()
