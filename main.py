import argparse
from os.path import dirname, abspath, join, exists
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

# Random seed
np.random.seed(0)
torch.manual_seed(0)

# Arguments parser
parser = argparse.ArgumentParser(description="Deep NLP Models for Text Classification")
parser.add_argument('--dataset', type=str, choices=DATASETS, default='dbpedia')
parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
parser.set_defaults(use_gpu=torch.cuda.is_available())
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--initial_lr', type=float, default=0.0001)
parser.add_argument('--lr_schedule', action='store_true')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--load_model', type=str, default=None)

parser.set_defaults(preprocess_level='char')
parser.add_argument('--dictionary', type=str, default='VDCNNDictionary', choices=['CharCNNDictionary', 'VDCNNDictionary', 'AllCharDictionary'])
parser.add_argument('--min_length', type=int, default=1024)
parser.add_argument('--max_length', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--depth', type=str, choices=['vdcnn9', 'vdcnn17', 'vdcnn29', 'vdcnn49'], default='vdcnn49')
parser.add_argument('--embed_size', type=int, default=16)
parser.add_argument('--optional_shortcut', action='store_true')
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--sort_dataset', action='store_true')
parser.add_argument('--kmax', type=int, default=8)
parser.add_argument('--pooling',type=str, choices=['conv','kmaxpool','maxpool'], default='maxpool')
parser.add_argument('--num_workers', type=int, default=0)
parser.set_defaults(model=VDCNN)

if is_interactive():
    params = []
else:
    params = sys.argv[1:]

args = vars(parser.parse_args(params))

# Logging
model_name = args.get('model').__name__+'_'+args.get('depth')
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

logger.info("Constructing model...")
model_name = getattr(vdcnn_model, args.get('depth'))
model = model_name(n_classes=preprocessor.n_classes, vocabulary_size=dictionary.vocabulary_size, **args)

# load exit model
if args.get('load_model') is not None:
    logger.info("Loading exit model...")
    base_dir = dirname(abspath(trainers.__file__))
    checkpoint_dir = join(base_dir, 'checkpoints')
    model_name = args.get('load_model')
    checkpoint_filepath = join(checkpoint_dir, model_name)
    model.load_state_dict(torch.load(checkpoint_filepath))
    logger.info(checkpoint_filepath)

if args.get('use_gpu'):
    model = model.cuda()

logger.info("Making dataset & dataloader...")
train_dataset = TextDataset(train_data, dictionary, args.get('sort_dataset'), args.get('min_length'), args.get('max_length'))
train_dataloader = TextDataLoader(dataset=train_dataset, dictionary=dictionary, batch_size=args.get('batch_size'), shuffle = not args.get('sort_dataset'), num_workers = args.get('num_workers'))
val_dataset = TextDataset(val_data, dictionary, args.get('sort_dataset'), args.get('min_length'), args.get('max_length'))
val_dataloader = TextDataLoader(dataset=val_dataset, dictionary=dictionary, batch_size=args.get('batch_size'), shuffle = not args.get('sort_dataset'), num_workers = args.get('num_workers'))
# test_dataset = TextDataset(test_data, dictionary, args.get('sort_dataset'), args.get('min_length'), args.get('max_length'))
# test_dataloader = TextDataLoader(dataset=test_dataset, dictionary=dictionary, batch_size=args.get('batch_size'), shuffle = not args.get('sort_dataset'))

logger.info("Training...")
# trainable_params = [p for p in model.parameters() if p.requires_grad]
if args.get('optimizer') == 'Adam':
    optimizer = Adam(model.parameters(), lr=args.get('initial_lr'))
elif args.get('optimizer') == 'Adadelta':
    optimizer = Adadelta(params=trainable_params, lr=args.get('initial_lr'), weight_decay=0.95)
else:
    raise NotImplementedError()

lr_plateau = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5)
criterion = nn.CrossEntropyLoss
trainer = Trainer(model, train_dataloader, val_dataloader,
                  criterion=criterion, optimizer=optimizer,
                  lr_schedule=args.get('lr_schedule'), lr_scheduler=lr_plateau,
                  use_gpu=args.get('use_gpu'), logger=logger)
trainer.run(epochs=args.get('epochs'))
logger.info("Evaluating...")
logger.info('Best Model: {}'.format(trainer.best_checkpoint_filepath))
