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

# import model.VDCNN
# from model.VDCNN import VDCNN, vdcnn9, vdcnn17, vdcnn29, vdcnn49
import model.VDCNN3 as vdcnn_model
# from model.VDCNN3 import VDCNN, vdcnn9, vdcnn17, vdcnn29, vdcnn49

from model.VDCNN2 import VDCNN2


import utils

# Random seed
np.random.seed(0)
torch.manual_seed(0)



train_args={
    'dataset':'ag_news',
    'dictionary':'VDCNNDictionary',
    'use_gpu':True,
    'batch_size':100,
    'initial_lr':0.01,
    'optimizer':'Adam',
    'lr_schedule':True,
    'model_name':'vdcnn17',
    'min_length':1014,
    'max_length':1014,
    'epochs':100,
    'embed_size':16,
    'k':10,
    'sort_dataset':True,
    'preprocess_level':'char',
    'depth':17,
    'optional_shortcut':True
    }




# Logging
model_name = train_args.get('model_name')
logger = utils.get_logger(model_name)

logger.info('Arguments: {}'.format(train_args))

logger.info("Preprocessing...")
Preprocessor = DATASET_TO_PREPROCESSOR[train_args.get('dataset')]
preprocessor = Preprocessor(train_args.get('dataset'))
train_data, val_data, test_data = preprocessor.preprocess(level=train_args.get('preprocess_level'))

logger.info("Building dictionary...")
Dictionary = getattr(dictionaries, train_args.get('dictionary'))
dictionary = Dictionary()
dictionary.build_dictionary(train_data)

logger.info("Making dataset & dataloader...")
train_dataset = TextDataset(train_data, dictionary, train_args.get('sort_dataset'), train_args.get('min_length'), train_args.get('max_length'))
train_dataloader = TextDataLoader(dataset=train_dataset, dictionary=dictionary, batch_size=train_args.get('batch_size'))
val_dataset = TextDataset(val_data, dictionary, train_args.get('sort_dataset'), train_args.get('min_length'), train_args.get('max_length'))
val_dataloader = TextDataLoader(dataset=val_dataset, dictionary=dictionary, batch_size=train_args.get('batch_size'))
test_dataset = TextDataset(test_data, dictionary, train_args.get('sort_dataset'), train_args.get('min_length'), train_args.get('max_length'))
test_dataloader = TextDataLoader(dataset=test_dataset, dictionary=dictionary, batch_size=train_args.get('batch_size'))

logger.info("Constructing model...")
model_name = getattr(vdcnn_model, train_args.get('model_name'))
model = model_name(n_classes=preprocessor.n_classes, vocabulary_size=dictionary.vocabulary_size,optional_shortcut=train_args.get('optional_shortcut'), k=train_args.get('k'))

# model = VDCNN2(depth = train_args.get('depth'),n_classes=preprocessor.n_classes, dictionary=dictionary,optional_shortcut=train_args.get('optional_shortcut'),k=train_args.get('k'))
if train_args.get('use_gpu'):
    model = model.cuda()

logger.info("Training...")
trainable_params = [p for p in model.parameters() if p.requires_grad]
if train_args.get('optimizer') == 'Adam':
    optimizer = Adam(params=trainable_params, lr=train_args.get('initial_lr'))
if train_args.get('optimizer') == 'Adadelta':
    optimizer = Adadelta(params=trainable_params, lr=train_args.get('initial_lr'), weight_decay=0.95)
lr_plateau = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=5, min_lr=0.0001)
criterion = nn.CrossEntropyLoss
trainer = Trainer(model, train_dataloader, val_dataloader,
                  criterion=criterion, optimizer=optimizer,
                  lr_schedule=train_args.get('lr_schedule'), lr_scheduler=lr_plateau,
                  use_gpu=train_args.get('use_gpu'), logger=logger)
trainer.run(epochs=train_args.get('epochs'))

logger.info("Evaluating...")
logger.info('Best Model: {}'.format(trainer.best_checkpoint_filepath))
model.load_state_dict(torch.load(trainer.best_checkpoint_filepath)) # load best model
evaluator = Evaluator(model, test_dataloader, use_gpu=train_args.get('use_gpu'), logger=logger)
evaluator.evaluate()
