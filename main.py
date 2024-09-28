# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model
import metrics
import random

from datetime import datetime
from fast_transformers.attention_registry import AttentionRegistry

now = datetime.now()

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Transformer Language Model')
parser.add_argument('--exp_name', type=str, default=now.strftime("%d-%m-%yT%H-%M-%S"),
                    help="Name of the experiment")
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--data-clean', action='store_true',
                    help='Apply data cleaning strategies')
parser.add_argument('--shuffle', action='store_true', default=False,
                    help='shuffle data')
parser.add_argument('--emsize', type=int, default=256,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=256,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.01,
                    help='weight decay')
parser.add_argument('--optim', type=str, default='adam',
                    help='type of optimizer', choices=['adam', 'sgd', 'adamw'])
parser.add_argument('--gamma', type=float, default=0.9,
                    help='multiplicative factor of learning rate decay')
parser.add_argument('--clip', type=float, default=5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                    help='enables macOS GPU training')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--valid-interval', type=int, default=1000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default=None,
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--attention-type', default="full",
                    choices=AttentionRegistry.keys,
                    help='Typ of attention mechanism, ("full" is equivalent to scaled dot product attention)')
args = parser.parse_args()
args.exp_name = f"{args.attention_type}_{args.exp_name}"

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    if not args.mps:
        print("WARNING: You have mps device, to enable macOS GPU run with --mps.")

use_mps = args.mps and torch.backends.mps.is_available()
if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

###############################################################################
# Load data
###############################################################################
corpus = data.Corpus(args.data, clean=args.data_clean)


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
train_idxs = list(range(0, train_data.size(0) - args.bptt - 1))
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
atype = args.attention_type
model = model.CustomTransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout, atype).to(device)

criterion = nn.NLLLoss()
if args.optim == 'adam':
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == 'adamw':
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
elif args.optim == 'sgd':
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
else:
    raise ValueError("Unknow optimizer: " + args.optim)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=4, factor=args.gamma, min_lr=1e-5)
print(model)
print(criterion)
print("Model Size:\t", end='')
print(f"{round(sum([p.numel() for p in model.parameters()])/1000000, 3)}M")
print("Attention size:\t", end='')
print(f"{round(sum([p.numel() for k,p in model.named_parameters() if 'inner_attention' in k])/1000000, 3)}M")
print([k for k,_ in model.named_parameters() if 'layers.0.attention.inner_attention' in k])

###############################################################################
# Training code
###############################################################################


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output = model(data)
            output = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train(train_step):
    # Turn on training mode which enables dropout.
    model.train()
    idx = train_step % len(train_idxs)
    if args.shuffle and idx == 0:
        random.shuffle(train_idxs)

    data, targets = get_batch(train_data, train_idxs[idx])
    # Starting each batch, we detach the hidden state from how it was previously produced.
    # If we didn't, the model would try backpropagating all the way to start of the dataset.
    optim.zero_grad()
    output = model(data)
    output = output.view(-1, ntokens)
    loss = criterion(output, targets)
    loss.backward()

    grads = metrics.get_grads(model.parameters())
    grad_norm = grads.norm()

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optim.step()
    lr = lr_scheduler.get_last_lr()[-1]

    metric_writer.log_metrics(loss.item(), train=True)
    if train_step % args.log_interval == 0 and train_step > 0 or args.dry_run:
        print('| step {:3d} / {:3d} | lr {:02.5f} | '
              'grad_norm {:.3f} | '
              'loss {:5.2f} | ppl {:8.2f}'.format(
            train_step, max_steps, lr, grad_norm,
            loss.item(), math.exp(loss.item()))
        )


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
best_val_loss = None

# Log metrics to tensorboard
metric_writer = metrics.LMMetricWritter(model, args.exp_name)

if args.save is None:
    if not os.path.isdir("models"):
        os.mkdir("models")
    args.save = os.path.join("models", f"{args.exp_name}_model.pt")

# At any point you can hit Ctrl + C to break out of training early.
try:
    max_steps = args.epochs * train_data.size(0)
    epoch_start_time = time.time()
    for step in range(max_steps):
        train(step)
        if step % args.valid_interval == 0 and step > 0 or step == max_steps - 1 or args.dry_run:
            val_loss = evaluate(val_data)
            lr_scheduler.step(val_loss)
            metric_writer.log_metrics(val_loss, step, train=False)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(step, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)
            epoch_start_time = time.time()
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
        if args.dry_run:
            break
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
metric_writer.log_hparams(vars(args), {"hparam/loss": test_loss, "hparam/PPL": math.exp(test_loss)})
metric_writer.close()
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
print("Peak Memory: {:.3f} GB".format(torch.cuda.max_memory_allocated() / 1e9))

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
