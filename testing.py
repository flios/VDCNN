import argparse
from os.path import dirname, abspath, join, exists, isdir,isfile
import os

import torch
from torch.optim import Adadelta, Adam, lr_scheduler
from torch import nn
import numpy as np

from download_dataset import DATASETS
from preprocessors import DATASET_TO_PREPROCESSOR
import dictionaries
from dataloaders import TextDataset, TextDataLoader
import trainers
from trainers import Trainer
from evaluators import Evaluator

import model.VDCNN as vdcnn_model
from model.VDCNN import VDCNN

import utils
import sys

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

# Arguments parser
parser = argparse.ArgumentParser(description="Deep NLP Models for Text Classification")
parser.add_argument('--dataset', type=str, choices=DATASETS, default='yelp_review_polarity')
parser.add_argument('--use_gpu', type=bool, default=torch.cuda.is_available())
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--load_model', type=str, default ='yelp_review_polarity')

parser.set_defaults(preprocess_level='char')
parser.add_argument('--dictionary', type=str, default='VDCNNDictionary', choices=['CharCNNDictionary', 'VDCNNDictionary', 'AllCharDictionary'])
parser.add_argument('--min_length', type=int, default=1024)
parser.add_argument('--max_length', type=int, default=1024)
parser.add_argument('--depth', type=str, choices=['vdcnn9', 'vdcnn17', 'vdcnn29', 'vdcnn49'], default='vdcnn49')
parser.add_argument('--embed_size', type=int, default=16)
parser.add_argument('--optional_shortcut', type=bool, default=False)
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--sort_dataset', type=bool, default=False)
parser.add_argument('--kmax', type=int, default=8)
parser.add_argument('--pooling',type=str, choices=['conv','kmaxpool','maxpool'], default='maxpool')
parser.set_defaults(model=VDCNN)

if is_interactive():
    params = []
else:
    params = sys.argv[1:]

args = vars(parser.parse_args(params))
# Logging
model_name = 'TESTING ' + args.get('model').__name__+'_'+args.get('depth')
logger = utils.get_logger(model_name)

logger.info('Arguments: {}'.format(args))

logger.info("Preprocessing...")
Preprocessor = DATASET_TO_PREPROCESSOR[args.get('dataset')]
preprocessor = Preprocessor(args.get('dataset'))
train_data, val_data, test_data = preprocessor.preprocess(level=args.get('preprocess_level'))

logger.info("Building dictionary...")
Dictionary = getattr(dictionaries, args.get('dictionary'))
dictionary = Dictionary(args)
dictionary.build_dictionary(train_data)

logger.info("Making dataset & dataloader...")
test_dataset = TextDataset(test_data, dictionary, args.get('sort_dataset'), args.get('min_length'), args.get('max_length'))
test_dataloader = TextDataLoader(dataset=test_dataset, dictionary=dictionary, batch_size=args.get('batch_size'), shuffle = not args.get('sort_dataset'))

logger.info("Constructing model...")
model_name = getattr(vdcnn_model, args.get('depth'))
model = model_name(n_classes=preprocessor.n_classes, vocabulary_size=dictionary.vocabulary_size, **args)
logger.info("Loading model...")
base_dir = dirname(abspath(trainers.__file__))
checkpoint_dir = join(base_dir, 'checkpoints')
model_load_name = args.get('load_model')
checkpoint_filepath = join(checkpoint_dir, model_load_name)
model_name_list = []
if isdir(checkpoint_filepath):
    model_name_list = [join(checkpoint_filepath, f) for f in os.listdir(checkpoint_filepath) if f.endswith(".ckpt")]
else:
    model_name_list = [checkpoint_filepath]

for checkpoint in model_name_list:
    logger.info(checkpoint)
    model.load_state_dict(torch.load(checkpoint))
    if args.get('use_gpu'):
        model = model.cuda()
    evaluator = Evaluator(model, test_dataloader, use_gpu=args.get('use_gpu'), logger=logger)
    evaluator.evaluate()
