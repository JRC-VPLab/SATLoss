import os
from os.path import exists
import torch
import numpy as np
import cv2 as cv

from utils.dataset import Topo_dataloader
from utils.metrics import pixel_accuracy_item, dice_score_item, BettiError, clDice, precision, recall, f1score

def tester(args, model):
    accmeter = []
    dicemeter = []
    B0meter = []
    B1meter = []
    cldicemeter = []
    premeter = []
    recmeter = []
    f1meter = []

    test_dataset = Topo_dataloader(os.path.join(args.dataroot, args.dataset), 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    save_pred_dir = os.path.join(args.exp_output_dir, 'pred')
    save_pred_g_dir = os.path.join(args.exp_output_dir, 'pred_g')
    save_overlay_dir = os.path.join(args.exp_output_dir, 'overlay')
    if not exists(save_pred_dir):
        os.makedirs(save_pred_dir)
    if not exists(save_pred_g_dir):
        os.makedirs(save_pred_g_dir)
    if not exists(save_overlay_dir):
        os.makedirs(save_overlay_dir)

    model = model.eval()
    with torch.no_grad():
        for step, (samples, target, img_names) in enumerate(test_loader):
            # print('{}/{}'.format((step+1), len(test_loader)))
            samples = samples.permute(0, 3, 1, 2).to(torch.float32).to(args.device)
            target = target.permute(0, 3, 1, 2).to(torch.float32).to(args.device)
            target = target[:, 0, :, :].unsqueeze(1)
            N, C, H, W = samples.size()

            # forward
            pred = model(samples)

            pred_g = pred.clone()
            pred[pred <= 0.5] = 0
            pred[pred > 0.5] = 1

            # metrics
            acc = pixel_accuracy_item(pred, target)
            dice = dice_score_item(pred, target)
            Berror = BettiError(pred.cpu(), target.cpu())
            cldice = clDice(pred, target)
            prec = precision(pred, target)
            recl = recall(pred, target)
            f1 = f1score(pred, target)

            accmeter.extend(acc)
            dicemeter.extend(dice)
            B0meter.extend(Berror[0])
            B1meter.extend(Berror[1])
            cldicemeter.extend(cldice)
            premeter.extend(prec)
            recmeter.extend(recl)
            f1meter.extend(f1)

            pred_save = pred.permute(0, 2, 3, 1).squeeze(-1).cpu().numpy()  # binary prediction
            pred_g_save = pred_g.permute(0, 2, 3, 1).squeeze(-1).cpu().numpy()  # raw grayscale prediction


            for b_idx in range(N):
                save_name = img_names[b_idx].split('.')[0]
                cv.imwrite(os.path.join(save_pred_dir, '{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}.png'.format(save_name, acc[b_idx],
                                                                                             dice[b_idx], Berror[0][b_idx], Berror[1][b_idx], cldice[b_idx], prec[b_idx], recl[b_idx], f1[b_idx])),
                                                                                            (pred_save[b_idx,:,:]*255).astype(np.uint8))
                cv.imwrite(os.path.join(save_pred_g_dir, '{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}.png'.format(save_name, acc[b_idx],
                                                                                             dice[b_idx], Berror[0][b_idx], Berror[1][b_idx], cldice[b_idx], prec[b_idx], recl[b_idx], f1[b_idx])),
                                                                                            (pred_g_save[b_idx,:,:]*255).astype(np.uint8))

        acc_mean = np.mean(accmeter)
        dice_mean = np.mean(dicemeter)
        B0_mean = np.mean(B0meter)
        B1_mean = np.mean(B1meter)
        cldicemean = np.mean(cldicemeter)
        precmean = np.mean(premeter)
        reclmean = np.mean(recmeter)
        f1mean = np.mean(f1meter)

        print('Acc:{}, Dice:{}, B0 Error:{}, B1 Error:{}, clDice:{}, precision:{}, recall:{}, F1:{}'.format(acc_mean, dice_mean, B0_mean, B1_mean, cldicemean, precmean, reclmean, f1mean))
