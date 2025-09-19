# Multi-feature fusion method based on nonlinear spiking neural convolutional model for Chinese named entity recognition

Code for the MF-BERT model

# Requirement

* Python 3.8.18
* Transformer 3.5.1
* Numpy 1.22.4
* Packaging 23.2
* skicit-learn 0.24.2
* torch 2.2.1+cu121
* tqdm 4.66.2
* multiprocess 0.70.10
* tensorflow 2.3.1
* tensorboard 2.11.2
* seqeval 1.2.1
* FastNLP 0.5.0

# Download link

* Chinese BERT: https://huggingface.co/bert-base-chinese/tree/main <!--https://cdn.huggingface.co/bert-base-chinese-pytorch_model.bin-->

* Word Embedding: https://ai.tencent.com/ailab/nlp/en/data/tencent-ailab-embedding-zh-d200-v0.2.0.tar.gz,
More info refers to: [Tencent AI Lab Word Embedding](https://ai.tencent.com/ailab/nlp/en/embedding.html)

* tencent_vocab.txt, the vocab of pre-trained word embedding table, downlaod from [here](https://drive.google.com/file/d/1UmtbCSPVrXBX_y4KcovCknJFu9bXXp12/view?usp=sharing).

* chaizi.txt、ctb.50d.vec、word_char_mix.txt, download from [here](https://github.com/kfcd/chaizi)

# Run

* 1.Convert .char.bmes file to `.json file`, `python3 to_json.py`

* 2.Convert .char.bmes file to `mid_data/.json file`, run the `process.py` file in each dataset folder.

* 2.run the shell, `sh run_demo.sh`
