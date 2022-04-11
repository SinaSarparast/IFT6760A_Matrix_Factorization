import torch 
import os
import math
import copy
import time
from torch import nn, Tensor
from utils import write_to_log
from dataloader import *
from transformer_model import generate_square_subsequent_mask
from transformer_model import TransformerModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer():
    def __init__(self,  model: nn.Module, train_data, val_data, test_data, lr, ntokens, bptt, train_from,model_save_dir='./models', logs_dir='./logs') -> None:
        self.model = model.train()  # turn on train mode
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.log_interval = 200
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1) #reduction="sum", 
        # self.criterion = cal_loss
        self.lr = lr
        self.ntokens = ntokens
        self.bptt = bptt
        self.num_batches = len(self.train_data) // bptt
        self.train_from = train_from
        self.model_save_dir = model_save_dir
        self.logs_dir = logs_dir

        if train_from != None:
            checkpoint = torch.load(train_from)
            # model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout, atten_type='multilinear').to(device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.scheduler = checkpoint['lr_sched']
            # loss = checkpoint['loss']
            print('>>> successfully loaded the model from checkpoint.')
        else:
            self.current_epoch = 1

        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,amsgrad=False)
        # what sort of scheduling should we use, given that we can't train for very long? 
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 4000.0, gamma=0.95, verbose=True)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.1, verbose=True)

        pass

    def step(self, epoch) -> None:
        total_loss = 0.
        cur_loss = 0
        start_time = time.time()
        src_mask = generate_square_subsequent_mask(self.bptt).to(device)

        for batch, i in enumerate(range(0, self.train_data.size(0) - 1, self.bptt)):
            data, targets = get_batch(self.train_data, i, self.bptt)
            batch_size = data.size(0)
            if batch_size != self.bptt:  # only on last batch
                src_mask = src_mask[:batch_size, :batch_size]
            output = self.model(data, src_mask)
            loss = self.criterion(output.view(-1, self.ntokens), targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()
            if batch % self.log_interval == 0 and batch > 0:
                lr = self.scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / self.log_interval
                cur_loss = total_loss / self.log_interval
                ppl = math.exp(cur_loss)
                st = f'| epoch {epoch:3d} | {batch:5d}/{self.num_batches:5d} batches | 'f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | 'f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}'
                print(st)
                write_to_log(self.logs_dir,'training_log.log',st)
                total_loss = 0
                start_time = time.time()
            
        return cur_loss

    def evaluate(self, model, eval_data: Tensor) -> float:
        model.eval()  # turn on evaluation mode
        total_loss = 0.
        src_mask = generate_square_subsequent_mask(self.bptt).to(device)
        with torch.no_grad():
            for i in range(0, eval_data.size(0) - 1, self.bptt):
                data, targets = get_batch(eval_data, i, self.bptt)
                batch_size = data.size(0)
                if batch_size != self.bptt:
                    src_mask = src_mask[:batch_size, :batch_size]
                output = model(data, src_mask)
                output_flat = output.view(-1, self.ntokens)
                total_loss += batch_size * self.criterion(output_flat, targets).item()
        return total_loss / (len(eval_data) - 1)

    def end_of_training(self, best_model):
        # for evaluation
        test_loss = self.evaluate(best_model,self.test_data)
        test_ppl = math.exp(test_loss)
        st = '=' * 89+f'| End of training | test loss {test_loss:5.2f} | '+f'test ppl {test_ppl:8.2f}'+'=' * 89
        write_to_log(self.logs_dir,'training_log.log',st)
        print(st)

    def train(self, epochs):
        best_val_loss = float('inf')
        best_model = None
        for epoch in range(self.current_epoch, epochs + 1):
            epoch_start_time = time.time()
            loss = self.step(epoch)
            val_loss = self.evaluate(self.model, self.val_data)
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - epoch_start_time

            # More information about handling model weights
            # https://pytorch.org/tutorials/beginner/saving_loading_models.html
            if not os.path.exists(self.model_save_dir):
                os.mkdir(self.model_save_dir)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    'lr_sched': self.scheduler
                    }, os.path.join(self.model_save_dir,'multilinear_epoch{}.pt'.format(epoch)) )
            print('>>> successfully saved the model to checkpoint.')
            

            st = '-'*89+'\n'+f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '+f'valid loss {val_loss:5.2f} | valid ppl{val_ppl:8.2f}'+'\n'+'-'*89
            write_to_log(self.logs_dir,'training_log.log',st)
            print(st)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model)

            self.scheduler.step()
        
        self.end_of_training(best_model)
        return best_model