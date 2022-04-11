import os, torch
from trainer import *
from dataloader import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# from argparse import ArgumentParser

# to do: a more scaleable argument passing
# if __name__ == "__main__":
#     parser.add_argument("--experiment_name", type=str, help="name of the experiment")
#     args = parser.parse_args()

data_dir = './processed_data'
train_data = torch.load(os.path.join(data_dir,'train_data.pt'), map_location=device )
val_data = torch.load(os.path.join(data_dir,'val_data.pt'), map_location=device )
test_data = torch.load(os.path.join(data_dir,'test_data.pt'), map_location=device )
vocab = torch.load(os.path.join(data_dir,'vocab.pt'), map_location=device )

ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
lr = 5.0  # learning rate
model_save_dir = './models' # checkpoints are save here
logs_dir='./training_logs' # logs are save here
epochs = 10 # number of training epochs
bptt = 35 # ?
train_from = None#os.path.join(model_save_dir,'multilinear_epoch4.pt')
atten_type='multilinear' # attention type multihead (vanila transformer) and multilinear

# warmup_steps = 4000
# label_smoothing 

#build the model
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout, atten_type).to(device)
#
trainer = Trainer(model, train_data, val_data, test_data, lr, ntokens, bptt, train_from,model_save_dir, logs_dir)
trainer.train(epochs)

