from os.path import dirname, abspath, join, exists
import os
from datetime import datetime
import torch
from torch.autograd import Variable
import numpy as np

class Evaluator():

    def __init__(self, model, test_dataloader, criterion, use_gpu=False, logger=None):

        self.model = model
        self.test_dataloader = test_dataloader
        self.use_gpu = use_gpu
        self.criterion = criterion()
        if logger is not None:
            self.output = logger.info
        else:
            self.output = print

        self.base_message = "Test Loss: {test_loss:<.6}, Test Accuracy: {test_metric:<.1%}"

    def evaluate(self):
        self.output('Evaluating...')
        self.model.eval()

        # validation
        self.test_batch_metrics = []
        self.test_batch_losses = []
        for test_inputs, test_targets in self.test_dataloader:
            if self.use_gpu:
                self.test_inputs, self.test_targets = Variable(test_inputs.cuda()), Variable(test_targets.cuda())
            else:
                self.test_inputs, self.test_targets = Variable(test_inputs), Variable(test_targets)
            self.test_outputs = self.model(self.test_inputs)
            test_batch_metric = self.accuracy(self.test_outputs, self.test_targets)
            test_batch_loss = self.criterion(self.test_outputs, self.test_targets)
            self.test_batch_metrics.append(test_batch_metric.data.item())
            self.test_batch_losses.append(test_batch_loss.data.item())

        test_data_size = len(self.test_dataloader.dataset)
        test_metric = np.sum(self.test_batch_metrics) / test_data_size
        test_loss = np.sum(self.test_batch_losses) / test_data_size

        message = self.base_message.format(test_metric=test_metric, test_loss=test_loss)
        self.output(message)

    def accuracy(self, outputs, labels):
        maximum, argmax = outputs.max(dim=1)
        corrects = argmax == labels # ByteTensor
        n_corrects = corrects.float().sum() # FloatTensor
        return n_corrects
