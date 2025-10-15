import torch
import torch.nn.functional as F
import numpy as np
import math
from torch_topological.nn import CubicalComplex
from skimage.morphology import skeletonize, skeletonize_3d

getPersistentInfo = CubicalComplex(dim=2)

def pixel_accuracy(pred, target):
    assert pred.size() == target.size()

    correct = torch.sum(pred == target)
    total = pred.numel()

    acc = correct / total

    return [acc.item()]

def dice_score(pred, target):
    assert pred.size() == target.size()

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = torch.sum(pred * target)
    segmentation = torch.sum(pred)
    ground_truth = torch.sum(target)

    dice = (2.0 * intersection) / (segmentation + ground_truth + 1e-6)

    return [dice.item()]

def dice_score3d(pred, target):
    assert pred.size() == target.size()

    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = torch.sum(pred * target)
    segmentation = torch.sum(pred)
    ground_truth = torch.sum(target)

    dice = (2.0 * intersection) / (segmentation + ground_truth + 1e-6)

    return [dice.item()]


def pixel_accuracy_item(pred, target):
    assert pred.size() == target.size()
    N, _, _, _ = pred.size()
    acc_book = []

    for i in range(N):
        correct = torch.sum(pred[i,:,:,:] == target[i,:,:,:])
        total = pred[i,:,:,:].numel()

        acc = correct / total
        acc_book.append(acc.item())

    return acc_book


def dice_score_item(pred, target):
    assert pred.size() == target.size()
    N, _, _, _ = pred.size()
    dice_book = []

    for i in range(N):
        pred_i = pred[i,:,:,:].view(-1)
        target_i = target[i,:,:,:].view(-1)

        intersection = torch.sum(pred_i * target_i)
        segmentation = torch.sum(pred_i)
        ground_truth = torch.sum(target_i)

        dice = (2.0 * intersection) / (segmentation + ground_truth + 1e-6)

        dice_book.append(dice.item())

    return dice_book


def BettiError(pred, target):
    assert pred.size() == target.size()
    N, C, H, W = target.size()
    assert C == 1

    # pad to square (Must-do, otherwise computation of persistent diagrams incorrect)
    if H != W:
        margin = abs(H - W)
        pad1, pad2 = margin // 2, margin - margin // 2

        if H > W:
            paddings = (pad1, pad2, 0, 0)
        else:
            paddings = (0, 0, pad1, pad2)

        if pred is not None:
            pred = F.pad(pred, paddings, "constant", 0.0)
        if target is not None:
            target = F.pad(target, paddings, "constant", 0.0)

    # Invert color and clamp
    pred = 1 - pred
    target = 1 - target
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)

    p_pred = getPersistentInfo(pred)
    p_tgt = getPersistentInfo(target)

    B0E = []
    B1E = []

    for b_idx in range(N):
        Betti_p_0 = len(p_pred[b_idx][0][0].diagram)
        Betti_p_1 = len(p_pred[b_idx][0][1].diagram)
        Betti_t_0 = len(p_tgt[b_idx][0][0].diagram)
        Betti_t_1 = len(p_tgt[b_idx][0][1].diagram)

        B0E.append(np.abs(Betti_p_0 - Betti_t_0))
        B1E.append(np.abs(Betti_p_1 - Betti_t_1))

    return (B0E, B1E)

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice_ins(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))

    if tprec == 0 and tsens == 0:
        return 0.0
    elif math.isnan(tsens) and math.isnan(tprec):
        return 1.0
    elif not math.isnan(tsens) and not math.isnan(tprec):
        return 2 * tprec * tsens / (tprec + tsens)
    else:
        return 0.0


def clDice(x, y):
    assert x.size() == y.size()
    N, C, H, W = x.size()
    assert C == 1

    x = x.squeeze(1)
    y = y.squeeze(1)

    x = x.cpu().numpy()
    y = y.cpu().numpy()

    result = []

    for i in range(N):
        score = clDice_ins(x[i], y[i])
        if score == np.nan:
            score = 1.0
        result.append(score)

    return result


def precision(x, y):
    assert x.size() == y.size()
    N = x.size(0)
    pr = []

    for i in range(N):
        tp = (torch.mul(y[i], (x[i]==y[i]))).sum().item()
        fp = ((x[i]-y[i])==1).sum().item()
        if tp == 0 and fp == 0:
            pr.append(1.0)
        else:
            pr.append(tp / (tp + fp))

    return pr

def recall(x, y):
    assert x.size() == y.size()
    N = x.size(0)
    re = []

    for i in range(N):
        tp = (torch.mul(y[i], (x[i]==y[i]))).sum().item()
        fn = ((y[i]-x[i])==1).sum().item()
        if tp == 0 and fn == 0:
            re.append(1.0)
        else:
            re.append(tp / (tp + fn))

    return re

def f1score(x, y):
    assert x.size() == y.size()
    N = x.size(0)
    f1 = []

    for i in range(N):
        tp = (torch.mul(y[i], (x[i] == y[i]))).sum().item()
        fp = ((x[i] - y[i]) == 1).sum().item()
        fn = ((y[i] - x[i]) == 1).sum().item()

        if tp == 0 and fp == 0 and fn == 0:
            f1.append(1.0)
        else:
            f1.append((2 * tp) / (2 * tp + fp + fn))

    return f1

