from tqdm import trange


def to_labels(task_name, mode="train"):
    """
    从数据集中提取出该数据集所有标签
    Args:
        task_name: NER/weibo, NER/note4 and so on
        mode: train.
    """
    in_file = "../data/dataset/NER/msra/train.char.bio"
    out_file = "../data/dataset/NER/msra/labels.txt"

    with open(in_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        line_iter = trange(len(lines))
        labels = []
        for idx in line_iter:
            line = lines[idx].split("\t")
            # print(line)
            if len(line) == 2:
                label = line[1].split("\\")[0]
                labels.append(label)
        labels = list(set(labels))
        # print(labels)

    with open(out_file, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write("%s" % label)


if __name__ == "__main__":
    task_name = "NER/msra"
    to_labels(task_name, "train")
