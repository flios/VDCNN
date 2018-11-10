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
from trainers import Trainer
from evaluators import Evaluator

import model.VDCNN
from model.VDCNN import VDCNN, vdcnn9, vdcnn17, vdcnn29, vdcnn49

import utils

# Random seed
np.random.seed(0)
torch.manual_seed(0)

# Arguments parser
parser = argparse.ArgumentParser(description="Deep NLP Models for Text Classification")
parser.add_argument('--dataset', type=str, default='MR', choices=DATASETS)
parser.add_argument('--use_gpu', type=bool, default=torch.cuda.is_available())
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--initial_lr', type=float, default=0.01)
parser.add_argument('--lr_schedule', action='store_true')
parser.add_argument('--optimizer', type=str, default='Adam')

subparsers = parser.add_subparsers(help='NLP Model')


## VDCNN
VDCNN_parser = subparsers.add_parser('VDCNN')
VDCNN_parser.set_defaults(preprocess_level='char')
VDCNN_parser.add_argument('--dictionary', type=str, default='VDCNNDictionary', choices=['CharCNNDictionary', 'VDCNNDictionary', 'AllCharDictionary'])
VDCNN_parser.add_argument('--min_length', type=int, default=1014)
VDCNN_parser.add_argument('--max_length', type=int, default=1014)
VDCNN_parser.add_argument('--epochs', type=int, default=3)
VDCNN_parser.add_argument('--depth', type=str, default=29, choices=['vdcnn9', 'vdcnn17', 'vdcnn29', 'vdcnn49'])
VDCNN_parser.add_argument('--embed_size', type=int, default=16)
VDCNN_parser.add_argument('--optional_shortcut', type=bool, default=True)
VDCNN_parser.add_argument('--k', type=int, default=10)
VDCNN_parser.add_argument('--sort_dataset', action='store_true')
VDCNN_parser.set_defaults(model=VDCNN)

args = parser.parse_args()

# Logging
model_name = args.model.__name__
logger = utils.get_logger(model_name)

logger.info('Arguments: {}'.format(args))

logger.info("Preprocessing...")
Preprocessor = DATASET_TO_PREPROCESSOR[args.dataset]
preprocessor = Preprocessor(args.dataset)
train_data, val_data, test_data = preprocessor.preprocess(level=args.preprocess_level)

logger.info("Building dictionary...")
Dictionary = getattr(dictionaries, args.dictionary)
dictionary = Dictionary(args)
dictionary.build_dictionary(train_data)

logger.info("Making dataset & dataloader...")
train_dataset = TextDataset(train_data, dictionary, args.sort_dataset, args.min_length, args.max_length)
train_dataloader = TextDataLoader(dataset=train_dataset, dictionary=dictionary, batch_size=args.batch_size)
val_dataset = TextDataset(val_data, dictionary, args.sort_dataset, args.min_length, args.max_length)
val_dataloader = TextDataLoader(dataset=val_dataset, dictionary=dictionary, batch_size=args.batch_size)
test_dataset = TextDataset(test_data, dictionary, args.sort_dataset, args.min_length, args.max_length)
test_dataloader = TextDataLoader(dataset=test_dataset, dictionary=dictionary, batch_size=args.batch_size)

logger.info("Constructing model...")
model_name = getattr(model.VDCNN, args.depth)
model = model_name(n_classes=preprocessor.n_classes, vocabulary_size=dictionary.vocabulary_size,optional_shortcut=args.optional_shortcut)
if args.use_gpu:
    model = model.cuda()

logger.info("Training...")
trainable_params = [p for p in model.parameters() if p.requires_grad]
if args.optimizer == 'Adam':
    optimizer = Adam(params=trainable_params, lr=args.initial_lr)
if args.optimizer == 'Adadelta':
    optimizer = Adadelta(params=trainable_params, lr=args.initial_lr, weight_decay=0.95)
lr_plateau = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=5, min_lr=0.0001)
criterion = nn.CrossEntropyLoss
trainer = Trainer(model, train_dataloader, val_dataloader,
                  criterion=criterion, optimizer=optimizer,
                  lr_schedule=args.lr_schedule, lr_scheduler=lr_plateau,
                  use_gpu=args.use_gpu, logger=logger)
trainer.run(epochs=args.epochs)

logger.info("Evaluating...")
logger.info('Best Model: {}'.format(trainer.best_checkpoint_filepath))
model.load_state_dict(torch.load(trainer.best_checkpoint_filepath)) # load best model
evaluator = Evaluator(model, test_dataloader, use_gpu=args.use_gpu, logger=logger)
evaluator.evaluate()
