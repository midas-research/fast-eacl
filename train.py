from __future__ import division
from __future__ import print_function
import os
import copy
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from model import FAST
import torch.optim as optim
from evaluator import evaluate
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
seed = 123456789
np.random.seed(seed)
device = 'cuda'
writer = SummaryWriter('logs/')

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=123456789, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-2, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.20, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda=True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_text_path = "./processed_data/train_text/"
train_time_path = "./processed_data/train_timestamps/"
train_price_path = "./processed_data/train_price/"
train_mask_path = "./processed_data/train_mask/"
train_gt_path = "./processed_data/train_gt/"
no_of_tr_samples = len(os.listdir(train_text_path))

val_text_path = "./processed_data/val_text/"
val_time_path = "./processed_data/val_timestamps/"
val_price_path = "./processed_data/val_price/"
val_mask_path = "./processed_data/val_mask/"
val_gt_path = "./processed_data/val_gt/"
no_of_val_samples = len(os.listdir(val_text_path))

test_text_path = "./processed_data/test_text/"
test_time_path = "./processed_data/test_timestamps/"
test_price_path = "./processed_data/test_price/"
test_mask_path = "./processed_data/test_mask/"
test_gt_path = "./processed_data/test_gt/"
no_of_test_samples = len(os.listdir(test_text_path))

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

def loss_rank(pred, base_price, ground_truth, mask, alpha, no_stocks):
    return_ratio = torch.div((pred - base_price), base_price)
    reg_loss = weighted_mse_loss(return_ratio, ground_truth, mask)
    all_ones = torch.ones(no_stocks,1).to(device)
    pre_pw_dif =  (torch.matmul(return_ratio, torch.transpose(all_ones, 0, 1)) - torch.matmul(all_ones, torch.transpose(return_ratio,0,1)))
    gt_pw_dif = (torch.matmul(all_ones, torch.transpose(ground_truth,0,1)) - torch.matmul(ground_truth, torch.transpose(all_ones,0,1)))
    mask_pw = torch.matmul(mask, torch.transpose(mask,0,1))
    rank_loss = torch.mean(F.relu(((pre_pw_dif*gt_pw_dif)*mask_pw)))
    loss = reg_loss + alpha*rank_loss
    del mask_pw, gt_pw_dif, pre_pw_dif, all_ones
    return loss, reg_loss, rank_loss, return_ratio

def train(epoch):
    tra_loss = 0.0
    tra_reg_loss = 0.0
    tra_rank_loss = 0.0
    model.train()
    optimizer.zero_grad()
    for i in range(no_of_tr_samples):
      train_text = torch.tensor(np.load(train_text_path+str(i).zfill(10)+'.npy'), dtype=torch.float32).cuda()
      train_timestamps = torch.tensor(np.load(train_time_path+str(i).zfill(10)+'.npy'), dtype=torch.float32).cuda()
      output = model(train_text, train_timestamps, no_stocks)
      mask_batch = (np.load(train_mask_path+str(i).zfill(10)+'.npy'))#.cuda()
      price_batch = (np.load(train_price_path+str(i).zfill(10)+'.npy'))#.cuda()
      gt_batch = (np.load(train_gt_path+str(i).zfill(10)+'.npy'))#.cuda()
      cur_loss, cur_reg_loss, cur_rank_loss, curr_rr_train = loss_rank(output, torch.FloatTensor(price_batch).to(device), 
                                                                                  torch.FloatTensor(gt_batch).to(device), 
                                                                                  torch.FloatTensor(mask_batch).to(device), 
                                                                                  float(0.2), int(no_stocks))
      cur_loss.backward()
      # print('[INFO] Training: loss: ', cur_loss)
      optimizer.step()
      tra_loss += cur_loss.detach().cpu().item()
      tra_reg_loss += cur_reg_loss.detach().cpu().item()
      tra_rank_loss += cur_rank_loss.detach().cpu().item()
      # print('[INFO] METRICS -- Training Loss:',
            # tra_loss / (no_of_tr_samples),
            # tra_reg_loss / (no_of_tr_samples),
            # tra_rank_loss / (no_of_tr_samples))
    del price_batch
    del gt_batch
    del mask_batch 

def test_dict():
    with torch.no_grad():
        cur_valid_pred = np.zeros(
            [no_stocks, no_of_val_samples],
            dtype=float)
        cur_valid_gt = np.zeros(
            [no_stocks, no_of_val_samples],
            dtype=float)
        cur_valid_mask = np.zeros(
            [no_stocks, no_of_val_samples],
            dtype=float)
        val_loss = 0.0
        val_reg_loss = 0.0
        val_rank_loss = 0.0
        
        model.eval()
        for i in range(no_of_val_samples):
            val_text = torch.tensor(np.load(val_text_path+str(i).zfill(10)+'.npy'), dtype=torch.float32).cuda()
            val_timestamps = torch.tensor(np.load(val_time_path+str(i).zfill(10)+'.npy'), dtype=torch.float32).cuda()
            output_val = model(val_text, val_timestamps, no_stocks)
            mask_batch = (np.load(val_mask_path+str(i).zfill(10)+'.npy'))#.cuda()
            price_batch = (np.load(val_price_path+str(i).zfill(10)+'.npy'))#.cuda()
            gt_batch = (np.load(val_gt_path+str(i).zfill(10)+'.npy'))#.cuda()
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = loss_rank(output_val, torch.FloatTensor(price_batch).to(device), 
                                                                                        torch.FloatTensor(gt_batch).to(device), 
                                                                                        torch.FloatTensor(mask_batch).to(device), 
                                                                                        float(0.2), int(1.0))
            cur_rr = cur_rr.detach().cpu().numpy().reshape((no_stocks,1))
            val_loss += cur_loss.detach().cpu().item()
            val_reg_loss += cur_reg_loss.detach().cpu().item()
            val_rank_loss += cur_rank_loss.detach().cpu().item()

            cur_valid_pred[:, i] = \
                copy.copy(cur_rr[:, 0])
            cur_valid_gt[:, i] = \
                copy.copy(gt_batch[:, 0])
            cur_valid_mask[:, i] = \
                copy.copy(mask_batch[:, 0])

        # print('[INFO] METRICS -- Validation MSE:',
        #     val_loss / (no_of_val_samples),
        #     val_reg_loss / (no_of_val_samples),
        #     val_rank_loss / (no_of_val_samples))
        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
        # print('\t [INFO] METRICS -- Validation preformance:', cur_valid_perf)
        del price_batch
        del gt_batch
        del mask_batch

        cur_test_pred = np.zeros(
            [no_stocks, no_of_test_samples],
            dtype=float)
        cur_test_gt = np.zeros(
            [no_stocks, no_of_test_samples],
            dtype=float)
        cur_test_mask = np.zeros(
            [no_stocks, no_of_test_samples],
            dtype=float
        )
        test_loss = 0.0
        test_reg_loss = 0.0
        test_rank_loss = 0.0
        
        model.eval()
        for i in range(no_of_test_samples):
            test_text = torch.tensor(np.load(test_text_path+str(i).zfill(10)+'.npy'), dtype=torch.float32).cuda()
            test_timestamps = torch.tensor(np.load(test_time_path+str(i).zfill(10)+'.npy'), dtype=torch.float32).cuda()
            output_test = model(test_text, test_timestamps, no_stocks)
            mask_batch = (np.load(test_mask_path+str(i).zfill(10)+'.npy'))#.cuda()
            price_batch = (np.load(test_price_path+str(i).zfill(10)+'.npy'))#.cuda()
            gt_batch = (np.load(test_gt_path+str(i).zfill(10)+'.npy'))#.cuda()
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = loss_rank(output_test, torch.FloatTensor(price_batch).to(device), 
                                                                                        torch.FloatTensor(gt_batch).to(device), 
                                                                                        torch.FloatTensor(mask_batch).to(device), 
                                                                                        float(0.2), int(1.0))
            cur_rr = cur_rr.detach().cpu().numpy().reshape((no_stocks,1))
            test_loss += cur_loss.detach().cpu().item()
            test_reg_loss += cur_reg_loss.detach().cpu().item()
            test_rank_loss += cur_rank_loss.detach().cpu().item()

            cur_test_pred[:, i] = \
                copy.copy(cur_rr[:, 0])
            cur_test_gt[:, i] = \
                copy.copy(gt_batch[:, 0])
            cur_test_mask[:, i] = \
                copy.copy(mask_batch[:, 0])

        # print('[INFO] METRICS -- Test:',
        #     test_loss / (no_of_test_samples),
        #     test_reg_loss / (no_of_test_samples),
        #     test_rank_loss / (no_of_test_samples))
        cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
        print('\t[INFO] METRICS -- Test performance:', cur_test_perf)
        df = pd.DataFrame(columns=['mrr','irr','sr','ndcg'])
        df = df.append(cur_test_perf,ignore_index=True)
        df.to_csv('fast_test_results.csv')
        del price_batch
        del gt_batch
        del mask_batch

model = FAST().to('cuda')
lr_list = [1e-3, 5e-4, 3e-5]
for lr in lr_list:
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=l_r, weight_decay=args.weight_decay)
    for epoch in tqdm(range(args.epochs)):
        train(epoch)
    results = test_dict()