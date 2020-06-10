# TextING

The code and dataset for the ACL2020 paper [Every Document Owns Its Structure: Inductive Text Classification via Graph Neural Networks](https://arxiv.org/abs/2004.13826), implemented in Tensorflow.

Some functions are based on [Text GCN](https://github.com/yao8839836/text_gcn). Thank for their work.

## Requirements

* Python 3.6
* Tensorflow 1.12.0

## Usage

Download pre-trained word embeddings `glove.6B.300d.txt` from [here](http://nlp.stanford.edu/data/glove.6B.zip) and unzip to the repository.

Build graphs from the datasets in `data/corpus/` as:

    python build_graph.py [DATASET] [WINSIZE]

Provided datasets include `mr`,`ohsumed`,`R8`and`R52`. The default sliding window size is 3.

To use your own dataset, put the text file under `data/corpus/` and the label file under `data/` as other datasets do. Preprocess the text by running `remove_words.py` before building the graphs.

Start training and inference as:

    python train.py [--dataset DATASET] [--learning_rate LR]
                    [--epochs EPOCHS] [--batch_size BATCHSIZE]
                    [--hidden HIDDEN] [--steps STEPS]
                    [--dropout DROPOUT] [--weight_decay WD]

To reproduce the result, large hidden size and batch size are suggested as long as your memory allows. We report our result based on 96 hidden size with 1 batch. For the sake of memory efficiency, you may change according to your hardware.

## Citation

    @article{zhang2020every,
      title={Every Document Owns Its Structure: Inductive Text Classification via Graph Neural Networks},
      author={Zhang, Yufeng and Yu, Xueli and Cui, Zeyu and Wu, Shu and Wen, Zhongzhen and Wang, Liang},
      journal={arXiv preprint arXiv:2004.13826},
      year={2020}
    }
