import torch
from trainer import *
from dataloader import *
from argparse import ArgumentParser

data_dir = './processed_data'
train_data = torch.load(os.path.join(data_dir,'train_data.pt'), map_location=device )
val_data = torch.load(os.path.join(data_dir,'val_data.pt'), map_location=device )
test_data = torch.load(os.path.join(data_dir,'test_data.pt'), map_location=device )
vocab = torch.load(os.path.join(data_dir,'vocab.pt'), map_location=device )

# only use for debugging
# train_data = torch.randint(0, 100, (150,50))
# val_data = torch.randint(0, 100, (100,50))
# test_data = torch.randint(0, 100, (100,50))


if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler="resolve")

    parser.add_argument("-emsize", type=int, default=50, help="multilabel classification")
    parser.add_argument("-d_hid", type=int, default=50, help="multilabel classification")
    parser.add_argument("-d_model", type=int, default=50, help="multilabel classification")
    parser.add_argument("-dim_feedforward", type=int, default=2048, help="multilabel classification")
    parser.add_argument("-nlayers", type=int, default=2, help="multilabel classification")
    parser.add_argument("-nhead", type=int, default=2, help="multilabel classification")
    parser.add_argument("-dropout", type=float, default=0.2, help="multilabel classification")
    parser.add_argument("-lr", type=float, default=5.0, help="multilabel classification")
    parser.add_argument("-model_save_dir", type=str, default='./models', help="multilabel classification")
    parser.add_argument("-logs_dir", type=str, default='./training_logs', help="multilabel classification")
    parser.add_argument("-epochs", type=int, default=10, help="multilabel classification")
    parser.add_argument("-bptt", type=int, default=35, help="multilabel classification")
    parser.add_argument("-atten_type", type=str, default='multilinear', help="multilabel classification")
    parser.add_argument("-log_interval", type=int, default=2000, help="multilabel classification")
    parser.add_argument("-train_from", type=str, default=None, help="multilabel classification")

    args = parser.parse_args()
    args.ntokens = len(vocab)  # size of vocabulary

    # only for debugging
    # args.ntokens = 100

    # warmup_steps = 4000
    # label_smoothing 

    #build the model
    model = TransformerModel(args).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} parameters.'.format(pytorch_total_params))

    trainer = Trainer(model, train_data, val_data, test_data, args)
    trainer.train(args.epochs)

