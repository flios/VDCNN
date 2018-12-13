# Very Deep Convolutional Networks for Text

Pytorch implementation of very deep convolutional networks.

[VDCNN : Very Deep Convolutional Networks for Text Classification](https://arxiv.org/abs/1606.01781)

## Requirements
- Python 3.7+
- [PyTorch 0.4](http://pytorch.org/)
- [gensim 3.2](https://github.com/RaRe-Technologies/gensim)

## Usage

### Train
```python main.py -h```

You will get:

```
usage: main.py [-h]
​               [--dataset {MR,SST-1,SST-2,ag_news,sogou_news,dbpedia,yelp_review_full,yelp_review_polarity,yahoo_answers,amazon_review_full,amazon_review_polarity}]
​               [--use_gpu] [--batch_size BATCH_SIZE] [--initial_lr INITIAL_LR]
​               [--lr_schedule] [--optimizer OPTIMIZER]
​               [--load_model LOAD_MODEL]
​               [--dictionary {CharCNNDictionary,VDCNNDictionary,AllCharDictionary}]
​               [--min_length MIN_LENGTH] [--max_length MAX_LENGTH]
​               [--epochs EPOCHS] [--depth {vdcnn9,vdcnn17,vdcnn29,vdcnn49}]
​               [--embed_size EMBED_SIZE] [--optional_shortcut]
​               [--kernel_size KERNEL_SIZE] [--sort_dataset] [--kmax KMAX]
​               [--pooling {conv,kmaxpool,maxpool}] [--num_workers NUM_WORKERS]
```
## References
- [Deep Text Classification in PyTorch](https://github.com/dreamgonfly/deep-text-classification-pytorch)
