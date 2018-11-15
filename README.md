# VDCNN BW version

## How to run:
1. Change path in ```.pbs``` files


2. Download data using
```console
traxxx@h2ologin1:~/<yourpath>/Proj> qsub download_dataset.pbs
```
- This downloads all 5 dataset we want to use; see python file for details


3. Run job using
```console
traxxx@h2ologin1:~/<yourpath>/Proj> qsub run.pbs
```
- Remeber to change the parameters:
```shell
main.py --dataset yelp_review_full --epochs 100 --batch_size 64 --initial_lr 0.0001 --depth vdcnn49 --pooling maxpool
```
- Especially:
  - dataset: choose from [ag_news, dbpedia, yelp_review_full, yelp_review_polarity, yahoo_answers]

  - depth: choose from [vdcnn17, vdcnn29, vdcnn49]
            
