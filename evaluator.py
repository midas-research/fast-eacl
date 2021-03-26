import math
import numpy as np
from sklearn.metrics import ndcg_score

def evaluate(prediction, ground_truth, mask, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    # performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask)**2/ np.sum(mask)
    mrr_top = 0.0
    all_miss_days_top = 0
    bt_long = 1.0
    bt_long5 = 1.0
    bt_long10 = 1.0
    top_1_ground_truth = []
    top_5_ground_truth = []
    top_10_ground_truth = []
    sharpe_li = []
    sharpe_li5 = []
    for i in range(prediction.shape[1]):
        rank_gt = np.argsort(ground_truth[:, i])  
        gt_top1 = set()
        gt_top5 = set()
        gt_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top1) < 1:
                gt_top1.add(cur_rank)
            if len(gt_top5) < 5:
                gt_top5.add(cur_rank)
            if len(gt_top10) < 10:
                gt_top10.add(cur_rank)
        rank_pre = np.argsort(prediction[:, i])
        pre_top1 = set()
        pre_top5 = set()
        pre_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top1) < 1:
                pre_top1.add(cur_rank)
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
            if len(pre_top10) < 10:
                pre_top10.add(cur_rank)
        performance['ndcg'] = ndcg_score(np.array(list(gt_top5)).reshape(1,-1), np.array(list(pre_top5)).reshape(1,-1))
        # performance['ndcg10'] = ndcg_score(np.array(list(gt_top10)).reshape(1,-1), np.array(list(pre_top10)).reshape(1,-1))
        top1_pos_in_gt = 0
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            else:
                top1_pos_in_gt += 1
                if cur_rank in pre_top1:
                    break
        if top1_pos_in_gt == 0:
            all_miss_days_top += 1
        else:
            mrr_top += 1.0 / top1_pos_in_gt
        aaabbb = ground_truth[list(gt_top1)[0]][i]
        top_1_ground_truth.append(aaabbb)
        real_ret_rat_top = ground_truth[list(pre_top1)[0]][i]
        bt_long += real_ret_rat_top
        sharpe_li.append(real_ret_rat_top)
        real_ret_rat_top5 = 0
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        real_ret_rat_top5 /= 5
        bt_long5 += real_ret_rat_top5
        sharpe_li5.append(real_ret_rat_top)
        real_ret_rat_top5_gt = 0
        for pre in gt_top5:
            real_ret_rat_top5_gt += ground_truth[pre][i]
        real_ret_rat_top5_gt /= 5
        top_5_ground_truth.append(real_ret_rat_top5_gt)
        real_ret_rat_top10_gt = 0
        real_ret_rat_top10 = 0
        for pre in gt_top10:
            real_ret_rat_top10_gt += ground_truth[pre][i]
        real_ret_rat_top10_gt /= 10
        for pre in pre_top10:
            real_ret_rat_top10 += ground_truth[pre][i]
        real_ret_rat_top10 /= 10
        bt_long10 += real_ret_rat_top10
    performance['mrr'] = mrr_top / (prediction.shape[1] - all_miss_days_top)
    # performance['irr'] = bt_long
    performance['irr'] = bt_long5
    # performance['irr10'] = bt_long10
    sharpe_li = np.array(sharpe_li)
    performance['sr'] = (np.mean(sharpe_li)/np.std(sharpe_li))*15.87
    return performance