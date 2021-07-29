import numpy as np
import torch
from torch.optim import lr_scheduler
import random
import time
import logging

from bst import BSTransformer

log = logging.getLogger(__name__)


class Trainer():
    def __init__(self, bst, loss_fn, optimizer, batch_fn, epochs, scheduler, scheduler_type='', is_cuda=False):
        self.dtype = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
        self.bst = bst.cuda() if is_cuda else bst
        self.scheduler = False
        self.grad_clipping = False
        self.batch_num = 0
        self.epoch_num = 1
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_fn = batch_fn
        self.total_epochs = epochs
        self.is_cuda = is_cuda
        self.scheduler = scheduler  
        self.scheduler_type = scheduler_type
    
    def epoch(self, user_seq, context_features, batch_size, max_seq_len):
        self.training_start = time.time()
        self.bst.train()
        self.train_loss = 0
        step = 0
        for user_seq_batch, context_batch in self.batch_fn(user_seq, context_features, batch_size, max_seq_len):
            #try:
            pred, target = self.bst(torch.tensor(user_seq_batch).to(device), torch.tensor(context_batch).to(device))
            #except:
             #   return user_seq_batch, context_batch
            loss = self.loss_fn(pred.view(-1, pred.size(-1)), target.view(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.bst.parameters(), 
                                              self.clipper.clip)
            self.optimizer.step()
            self.train_loss += loss.data
            self.batch_num += 1
            if int(self.batch_num * batch_size) >= save_states[step]:
                print(f'Loss after {self.batch_num * batch_size} sequences: '
                         f'{self.train_loss.cpu().data.numpy() / self.batch_num}'
                         f'\nTraining time: {time.time() - self.training_start}')

                #self.save_state(save_loc, int(self.batch_num * batch_size))
                step += 1
            if self.scheduler_type == 'cyclic':
                self.scheduler.step()
        
        self.train_loss = self.train_loss.cpu().data.numpy()/self.batch_num
        
    def save_state(self, path, batch_num, save_grads=False):
        torch.save(self.encoder.state_dict(), f'{path}_{batch_num}')
        np.save(f'{path}_grads', self.clipper.total_grads)
