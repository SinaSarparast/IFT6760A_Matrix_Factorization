import os
import torch
from typing import Tuple
from torch.utils.data import dataset
from torch import Tensor
from torchtext import datasets
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import dataset
from torchtext.datasets import WikiText2, WikiText103

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_batch(source: Tensor, i: int, bptt: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def data_process(raw_text_iter: dataset.IterableDataset, vocab, tokenizer) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    # reduce data size only for debugging
    # data = []
    # i = 0
    # for item in raw_text_iter:
    #   if i == 500:
    #     break
    #   data.append(torch.tensor(vocab(tokenizer(item)), dtype=torch.long))
    #   i+=1
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler="resolve")
    parser.add_argument("--raw_data_dir", default='./raw_data', type=str, help="raw data directory ")
    parser.add_argument("--prc_data_dir", default='./processed_data', type=str, help="processed data directory ")
    parser.add_argument("--train_batch_size", default=20, type=str, help="train set batch size")
    parser.add_argument("--eval_batch_size", default=10, type=str, help="evaluation set batch size")
    parser.add_argument("--test_batch_size", default=10, type=str, help="test set batch size")

    args = parser.parse_args()

    # download and prepare wikitext dataset for language modeling
    # use wikitext 2 for debugging it's faster.
    train_iter = WikiText2(split='train')
    # train_iter = WikiText103(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    # train_iter, val_iter, test_iter = WikiText103(root = args.raw_data_dir, split = ('train', 'valid', 'test'))
    train_iter, val_iter, test_iter = WikiText2(root = args.raw_data_dir, split = ('train', 'valid', 'test'))

    train_data = data_process(train_iter,vocab,tokenizer)
    val_data = data_process(val_iter,vocab,tokenizer)
    test_data = data_process(test_iter,vocab,tokenizer)

    train_data = batchify(train_data, args.train_batch_size)  # shape [seq_len, batch_size]
    val_data = batchify(val_data, args.eval_batch_size)
    test_data = batchify(test_data, args.test_batch_size)

    if not os.path.exists(args.prc_data_dir):
        os.mkdir(args.prc_data_dir)
    torch.save(train_data, os.path.join(args.prc_data_dir,'train_data.pt') )
    torch.save(val_data, os.path.join(args.prc_data_dir,'val_data.pt') )
    torch.save(test_data, os.path.join(args.prc_data_dir,'test_data.pt') )
    torch.save(vocab,os.path.join(args.prc_data_dir,'vocab.pt') )