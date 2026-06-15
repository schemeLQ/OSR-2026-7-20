import torch
import numpy as np
from misc.util import *
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve

MAX_NUM = 999999


def compute_score(logit_list, softmax_list, score_wgts, branch_opt, fts=None):
    msp = softmax_list[branch_opt].max(1)[0]
    mls = logit_list[branch_opt].max(1)[0]
    if score_wgts[2] != 0:
        ftl = fts.mean(dim=[2, 3]).norm(dim=1, p=2)
        temp = (score_wgts[0] * msp + score_wgts[1] * mls + score_wgts[2] * ftl)
    else:
        temp = (score_wgts[0] * msp + score_wgts[1] * mls)
    return temp


def evaluation(model, test_loader, out_loader, **options):
    model.eval()
    torch.cuda.empty_cache()

    correct = 0
    total = 0
    n = 0

    pred_close = []
    pred_open = []
    labels_close = []
    labels_open = []
    score_close = []
    score_open = []

    open_labels = torch.zeros(MAX_NUM)
    probs = torch.zeros(MAX_NUM)

    # 智能开关特征图计算
    need_ft = (options.get('score_wgts', [0, 1, 0])[2] != 0)

    print(f"Start Evaluation... (Feature Calc: {'ON' if need_ft else 'OFF'})")

    with torch.no_grad():
        # --- 1. 已知类测试 ---
        for batch_idx, (data, labels) in enumerate(test_loader):
            if batch_idx % 100 == 0:
                print(f"Testing Known Classes: [{batch_idx}/{len(test_loader)}]")

            data, labels = data.cuda(), labels.cuda()
            batch_size = labels.size(0)

            with torch.set_grad_enabled(False):
                output_dict = model(data, return_ft=need_ft)
                logits_list = output_dict['logits']
                softmax_list = torch.stack(logits_list)
                softmax_list = torch.softmax(softmax_list / options.get('lgs_temp', 1.0), dim=2)

                if need_ft:
                    fts = output_dict['fts']
                    score_temp = compute_score(logits_list, softmax_list, options['score_wgts'], options['branch_opt'],
                                               fts=fts)
                else:
                    score_temp = compute_score(logits_list, softmax_list, options['score_wgts'], options['branch_opt'])

                score_close.append(score_temp.data.cpu().numpy())
                cpu_scores = score_temp.detach().cpu()
                probs[n: n + batch_size] = cpu_scores
                open_labels[n: n + batch_size] = 1
                n += batch_size

                pred_label = softmax_list[options['branch_opt']].data.max(1)[1]
                total += labels.size(0)
                correct += (pred_label == labels.data).sum()
                pred_close.append(softmax_list[options['branch_opt']].data.cpu().numpy())
                labels_close.append(labels.data.cpu().numpy())

        # --- 2. 未知类测试 ---
        for batch_idx, (data, labels) in enumerate(out_loader):
            if batch_idx % 100 == 0:
                print(f"Testing Unknown Classes: [{batch_idx}/{len(out_loader)}]")

            data, labels = data.cuda(), labels.cuda()
            batch_size = labels.size(0)
            ood_label = torch.zeros_like(labels) - 1

            with torch.set_grad_enabled(False):
                output_dict = model(data, return_ft=need_ft)
                logits_list = output_dict['logits']
                softmax_list = torch.stack(logits_list)
                softmax_list = torch.softmax(softmax_list / options.get('lgs_temp', 1.0), dim=2)

                if need_ft:
                    fts = output_dict['fts']
                    score_temp = compute_score(logits_list, softmax_list, options['score_wgts'], options['branch_opt'],
                                               fts=fts)
                else:
                    score_temp = compute_score(logits_list, softmax_list, options['score_wgts'], options['branch_opt'])

                score_open.append(score_temp.data.cpu().numpy())
                cpu_scores = score_temp.detach().cpu()
                probs[n: n + batch_size] = cpu_scores
                open_labels[n: n + batch_size] = 0  # 0 表示 Unknown (OOD)
                n += batch_size

                pred_open.append(softmax_list[options['branch_opt']].data.cpu().numpy())
                labels_open.append(ood_label.data.cpu().numpy())

    # --- 3. 指标计算 ---
    acc = float(correct) * 100. / float(total)

    pred_close = np.concatenate(pred_close, 0)
    pred_open = np.concatenate(pred_open, 0)
    labels_close = np.concatenate(labels_close, 0)
    labels_open = np.concatenate(labels_open, 0)
    score_close = np.concatenate(score_close, 0)
    score_open = np.concatenate(score_open, 0)

    # 这里的 labels 逻辑：已知类=1, 未知类=0
    open_labels = open_labels[:n].cpu().numpy()
    prob = probs[:n].reshape(-1, 1)

    # ROC 计算
    fpr, tpr, thresholds = roc_curve(open_labels, prob)
    auroc = auc(fpr, tpr)

    # 🚨🚨🚨 新增指标计算核心区域 🚨🚨🚨

    # [1] TNR @ TPR 95%
    # 也就是当 95% 的已知类被接受时，有多少未知类被拒绝
    # open_labels: 1=Known, 0=Unknown
    # prob: Knownness Score (越高越像已知类)

    # 找到 TPR 最接近 0.95 的阈值索引
    thresh_idx_95 = np.abs(np.array(tpr) - 0.95).argmin()
    tnr_at_tpr95 = 1. - fpr[thresh_idx_95]

    # [2] DTACC (Detection Accuracy)
    # 最佳分类阈值下的准确率：0.5 * (TPR + TNR)
    dtacc = 0.5 * (tpr + (1. - fpr)).max()

    # Macro F1 (保持原逻辑)
    pred1, pred2 = np.argmax(pred_close, axis=1), np.argmax(pred_open, axis=1)
    total_pred_label = np.concatenate([pred1, pred2], axis=0)
    total_label = np.concatenate([labels_close, labels_open], axis=0)
    total_pred = np.concatenate([score_close, score_open], axis=0)

    threshold_f1 = thresholds[thresh_idx_95]
    open_pred = (total_pred > threshold_f1).astype(np.float32)
    macro_f1 = f1_score(total_label, ((total_pred_label + 1) * open_pred) - 1, average='macro')

    # AUPR 计算
    precision, recall, _ = precision_recall_curve(open_labels, prob)
    aupr_in = auc(recall, precision)
    precision, recall, _ = precision_recall_curve(np.bitwise_not((open_labels).astype(bool)), -prob)
    aupr_out = auc(recall, precision)

    # 🚨🚨🚨 打印更新 (这就是你之前没看到指标的原因) 🚨🚨🚨
    print('=' * 50)
    print(f'📊 Evaluation Results:')
    print(f'   Accuracy: {acc:.3f}%')
    print(f'   AUROC:    {auroc:.5f}')
    print(f'   TNR@95:   {tnr_at_tpr95:.5f}  <-- [新增]')
    print(f'   DTACC:    {dtacc:.5f}       <-- [新增]')
    print(f'   AUPR_IN:  {aupr_in:.5f}')
    print(f'   AUPR_OUT: {aupr_out:.5f}')
    print(f'   Macro F1: {macro_f1:.5f}')
    print('=' * 50)

    # 返回完整列表
    return [acc, auroc, aupr_in, aupr_out, macro_f1, tnr_at_tpr95, dtacc]