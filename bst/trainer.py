import numpy as np
import torch
import time

from bst.transformer import BSTransformer
from bst.grad_clipping import GradientClipping


class Trainer:
    def __init__(self, config, loss_fn, batch_fn, device, grad_clipping=True):
        self.config = config
        self.bst = self.init_bst_encoder()
        self.optimizer = torch.optim.AdamW(self.bst.parameters(), lr=config['lr'])
        self.loss_fn = loss_fn
        self.batch_fn = batch_fn
        self.training_start = None
        self.device = device
        self.train_loss = 0
        self.best_loss = np.inf
        self.batch_num = 0
        self.epoch_num = 0
        self.scheduler = None
        try:
            if grad_clipping:
                self.clipper = GradientClipping(config['clip_value'])
                self.clipper.register_hook(self.bst)
        except KeyError:
            print("Gradient Clipping not available! Pass clip value in config!")

    def epoch(self, user_seq, context_features, batch_size, max_seq_len):
        self.training_start = time.time()
        self.bst.train()
        self.train_loss = 0

        # Iterate through batch.
        for user_seq_batch, context_batch in self.batch_fn(user_seq, context_features, batch_size, max_seq_len):
            pred, target = self.bst(torch.tensor(user_seq_batch).to(self.device),
                                    torch.tensor(context_batch).to(self.device))
            loss = self.loss_fn(pred.view(-1, pred.size(-1)), target.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.bst.parameters(),
                                           self.clipper.clip)
            self.optimizer.step()
            self.train_loss += loss.data
            self.batch_num += 1

            self.scheduler.step()  # set_to_none=True
        self.train_loss = self.train_loss.cpu().data.numpy() / self.batch_num

        # Log
        print(f'Loss after {self.batch_num * batch_size} sequences: '
              f'{self.train_loss}'
              f'\nTraining time: {time.time() - self.training_start}')

        # Save best weights
        if self.train_loss < self.best_loss:
            self.save_state('best', save_grads=False)
            self.best_loss = self.train_loss

    def init_bst_encoder(self):
        # Init Behaviour Seq Transformer model.
        bst = BSTransformer(self.config)
        bst = bst.cuda() if self.config['cuda'] else bst
        return bst

    def save_state(self, path, save_grads=False):
        # Save state to path.
        torch.save(self.bst.state_dict(), path)
        if save_grads:
            np.save(f'{path}_grads', self.clipper.total_grads)

    def set_lr_scheduler(self, milestones, gamma, last_epoch):
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
                                                              gamma=gamma, last_epoch=last_epoch)
