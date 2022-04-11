from train import *
from dataloader import *
from torchtext import datasets
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import dataset
from torchtext.datasets import WikiText2, WikiText103

# download and prepare wikitext dataset for language modeling
# use wikitext 2 for debugging it's faster.
# train_iter = WikiText2(split='train')
train_iter = WikiText103(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# train_iter was "consumed" by the process of building the vocab,
# so we have to create it again
train_iter, val_iter, test_iter = WikiText103(root = './data', split = ('train', 'valid', 'test'))

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

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
