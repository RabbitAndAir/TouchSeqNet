import argparse
import torch
parser = argparse.ArgumentParser()
# dataset and dataloader args
parser.add_argument('--file_path', type=str, choices=['data/regulation', 'data/touchalytics/data.csv', 'data/biodent/rawdata.csv'], default='data/regulation')
parser.add_argument('--save_path', type=str, default='exp/model')
parser.add_argument('--save_path_figure', type=str, default='out/figure')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser.add_argument('--device', type=str, default=device)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--val_batch_size', type=int, default=256)

# model args
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--attn_heads', type=int, default=4)
parser.add_argument('--eval_per_steps', type=int, default=16)
parser.add_argument('--enable_res_parameter', type=int, default=1)
parser.add_argument('--layers', type=int, default=8)
parser.add_argument('--alpha', type=float, default=5.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--num_channels', type=int, nargs='+', default=[32, 64], help="List of channel sizes")
parser.add_argument('--num_inputs', type=int, default=4)
parser.add_argument('--kernel_size', type=int, default=7)
parser.add_argument('--num_heads', type=int, default=4)

parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--vocab_size', type=int, default=192)
parser.add_argument('--wave_length', type=int, default=4)
parser.add_argument('--max_train_len', type=int, default=128)
parser.add_argument('--max_val_len', type=int, default=128)
parser.add_argument('--mask_ratio', type=float, default=0.5)
parser.add_argument('--reg_layers', type=int, default=4)

# train args
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_rate', type=float, default=1.)
parser.add_argument('--lr_decay_steps', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--num_epoch_pretrain', type=int, default=1)
parser.add_argument('--num_epoch', type=int, default=10)
parser.add_argument('--load_pretrained_model', type=int, default=1)


parser.add_argument('--num_epoch_contrastive', type=int, default=10)
parser.add_argument('--num_epoch_classifier', type=int, default=10)
parser.add_argument('--m', type=float, default=10)
parser.add_argument('--n', type=float, default=1)


args = parser.parse_args()

