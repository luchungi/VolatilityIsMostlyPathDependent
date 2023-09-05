import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer:
    '''
    Trainer class for training a pytorch model
    params:
        model: pytorch model
        optimizer: pytorch optimizer
        loss_fn: loss function which must take in y and model output as arguments where model output can be a tuple
        train_dataloader: pytorch dataloader for training data
        scheduler: pytorch scheduler
        valid_dataloader: pytorch dataloader for validation data
        test_dataloader: pytorch dataloader for test data
        device: device to train on
        dtype: convert each batch loaded from dataloader to dtype before passing to model
        path: path to save model weights
        **kwargs: additional arguments to be passed to loss_fn
    '''
    def __init__(self,
                 model,
                 optimizer,
                 loss_fn,
                 train_dataloader,
                 scheduler=None,
                 valid_dataloader=None,
                 test_dataloader=None,
                 device='cpu',
                 dtype=torch.float32,
                 path='./weights',
                 **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.device = device
        if scheduler is not None:
            self.scheduler = scheduler
        if valid_dataloader is not None:
            self.valid_dataloader = valid_dataloader
        if test_dataloader is not None:
            self.test_dataloader = test_dataloader
        self.dtype = dtype
        self.path = path
        self.kwargs = kwargs

        self.batch_log = {}
        self.epoch_log = {}

    def log_loss(self, key, loss):
        '''
        Log loss to self.batch_log
        '''
        # if key not in self.log.keys() then create a new key
        if self.batch_log.get(key) is None:
            self.batch_log[key] = []
        else:
            self.batch_log[key].append(loss)

    def train_epoch_end(self, epoch):
        '''
        Function to be called at the end of each epoch
        '''
        print(f'Epoch {epoch+1}')
        print('Training phase:', end=' ')
        for i, key in enumerate(self.batch_log.keys()):
            if i > 0:
                print(' | ', end=' ')
            if self.epoch_log.get(key) is None:
                self.epoch_log[key] = []
            epoch_loss = np.mean(self.batch_log[key])
            self.epoch_log[key].append(epoch_loss)
            print(f'{key}: {epoch_loss:.5f}', end=' ')
        self.batch_log = {}
        print()

    def valid_epoch_end(self):
        '''
        Function to be called at the end of each epoch
        '''
        print('Validation phase:', end=' ')
        for i, key in enumerate(self.batch_log.keys()):
            if i > 0:
                print(' | ', end=' ')
            if self.epoch_log.get(key) is None:
                self.epoch_log[key] = []
            epoch_loss = np.mean(self.batch_log[key])
            self.epoch_log[key].append(epoch_loss)
            print(f'{key}: {epoch_loss:.5f}', end=' ')
        self.batch_log = {}
        print()

    def test_epoch_end(self):
        '''
        Function to be called at the end of each epoch
        '''
        print('Test phase:', end=' ')
        for i, key in enumerate(self.batch_log.keys()):
            if i > 0:
                print(' | ', end=' ')
            if self.epoch_log.get(key) is None:
                self.epoch_log[key] = []
            epoch_loss = np.mean(self.batch_log[key])
            self.epoch_log[key].append(epoch_loss)
            print(f'{key}: {epoch_loss:.5f}', end=' ')
        self.batch_log = {}
        print()

    def run_scheduler(self, loss_key):
        '''
        Function to run scheduler on loss_key if scheduler is provided
        '''
        if hasattr(self, 'scheduler'):
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(self.epoch_log[loss_key][-1])
            else:
                self.scheduler.step()

    def save_model(self, model_name=None):
        '''
        Function to save the model to self.path with model_name as the filename
        '''
        if model_name is None:
            model_name = ''
        else:
            model_name = '_' + model_name
        torch.save(self.model.state_dict(), f'{self.path}/model{model_name}.pt')

    def check_best(self, loss_key):
        '''
        Function to check if the current epoch is the best epoch based on the loss_key
        '''
        if self.epoch_log[loss_key][-1] == min(self.epoch_log[loss_key]):
            return True
        else:
            return False

    def fit(self,
            n_epochs,
            model_name=None):
        '''
        Function to train the model
        '''
        for epoch in range(n_epochs):
            # Training phase
            self.model.train()
            for x, y in self.train_dataloader:
                x, y = x.to(self.dtype).to(self.device), y.to(self.dtype).to(self.device)
                self.train_batch(x, y)
            self.train_epoch_end(epoch)

            # Validation phase if valid_dataloader is provided
            if hasattr(self, 'valid_dataloader'):
                self.model.eval()
                for x, y in self.valid_dataloader:
                    x, y = x.to(self.dtype).to(self.device), y.to(self.dtype).to(self.device)
                    self.validate_batch(x, y)
                self.valid_epoch_end()

                # Save model if it is the best model based on validation loss
                if model_name is not None and self.check_best('valid_loss'):
                    print('Saving model')
                    self.save_model(model_name)

                # Run scheduler on validation loss if provided
                self.run_scheduler('valid_loss')
            else:
                # Save model if it is the best model based on training loss
                if model_name is not None and self.check_best('train_loss'):
                    print('Saving model')
                    self.save_model(model_name)

                # Run scheduler on training loss if provided
                self.run_scheduler('train_loss')

    def test(self):
        '''
        Function to evaluate the model on test data
        '''
        if hasattr(self, 'test_dataloader'):
            self.model.eval()
            for x, y in self.test_dataloader:
                x, y = x.to(self.dtype).to(self.device), y.to(self.dtype).to(self.device)
                self.test_batch(x, y)
            self.test_epoch_end()
        else:
            raise ValueError('Test dataloader not provided')

    def plot_loss(self, loss_keys, figsize=(10, 5)):
        '''
        Function to plot the loss for each epoch
        '''
        df = pd.DataFrame(self.epoch_log)
        df[loss_keys].plot(figsize=figsize)
        plt.legend()
        plt.show()

class MSETrainer(Trainer):
    '''
    Specific train/valid/test steps for AbsValue model
    '''

    def __init__(self,
                 model,
                 optimizer,
                 loss_fn,
                 train_dataloader,
                 scheduler=None,
                 valid_dataloader=None,
                 test_dataloader=None,
                 device='cpu',
                 dtype=torch.float32,
                 path='./weights',
                 **kwargs):
        super().__init__(model,
                         optimizer,
                         loss_fn,
                         train_dataloader,
                         scheduler,
                         valid_dataloader,
                         test_dataloader,
                         device,
                         dtype,
                         path,
                         **kwargs)

    def train_batch(self, x, y):
        '''
        Function to define a training step
        '''
        loss = self.loss_fn(y, self.model(x).squeeze())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.log_loss('train_loss', loss.item())

    def validate_batch(self, x, y):
        '''
        Function to define a validation step
        '''
        with torch.no_grad():
            loss = self.loss_fn(y, self.model(x).squeeze())
        self.log_loss('valid_loss', loss.item())

    def test_batch(self, x, y):
        '''
        Function to define a test step
        '''
        with torch.no_grad():
            loss = self.loss_fn(y, self.model(x).squeeze())
        self.log_loss('test_loss', loss.item())
