import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torch.nn as nn


class TMSE(nn.Module):
    """
        Temporal MSE Loss Function
        Proposed in Y. A. Farha et al. MS-TCN: Multi-Stage Temporal Convolutional Network for ActionSegmentation in CVPR2019
        arXiv: https://arxiv.org/pdf/1903.01945.pdf
    """

    def __init__(self, threshold=4, ignore_index=255):
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, preds, gts):

        total_loss = 0.
        batch_size = preds.shape[0]
        for pred, gt in zip(preds, gts):
            pred = pred[:, torch.where(gt != self.ignore_index)[0]]

            loss = self.mse(
                F.log_softmax(pred[:, 1:], dim=1),
                F.log_softmax(pred[:, :-1], dim=1)
            )

            loss = torch.clamp(loss, min=0, max=self.threshold**2)
            total_loss += torch.mean(loss)

        return total_loss / batch_size


class GaussianSimilarityTMSE(nn.Module):
    """
        Temporal MSE Loss Function with Gaussian Similarity Weighting
    """

    def __init__(self, threshold=4, sigma=1.0, ignore_index=255):
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.mse = nn.MSELoss(reduction='none')
        self.sigma = sigma

    def forward(self, preds, gts, feats):

        total_loss = 0.
        batch_size = preds.shape[0]
        for pred, gt, feat in zip(preds, gts, feats):
            pred = pred[:, torch.where(gt != self.ignore_index)[0]]
            feat = feat[:, torch.where(gt != self.ignore_index)[0]]

            # calculate gaussian similarity
            diff = feat[:, 1:] - feat[:, :-1]
            similarity = torch.exp(
                - torch.norm(diff, dim=0) / (2 * self.sigma**2)
            )

            # calculate temporal mse
            loss = self.mse(
                F.log_softmax(pred[:, 1:], dim=1),
                F.log_softmax(pred[:, :-1], dim=1)
            )
            loss = torch.clamp(loss, min=0, max=self.threshold**2)

            # gaussian similarity weighting
            loss = similarity * loss

            total_loss += torch.mean(loss)

        return total_loss / batch_size


class ActionSegmentationLoss(nn.Module):
    """
        Loss Function for Action Segmentation
        You can choose the below loss functions and combine them.
            - Cross Entropy Loss (CE)
            - Temporal MSE (TMSE)
    """

    def __init__(
        self, ce=True, tmse=False, weight=None, threshold=4, ignore_index=255,
        ce_weight=1.0, tmse_weight=0.15,
    ):
        super().__init__()
        self.criterions = []
        self.weights = []

        if ce:
            self.criterions.append(
                nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index))
            self.weights.append(ce_weight)

        if tmse:
            self.criterions.append(
                TMSE(threshold=threshold, ignore_index=ignore_index))
            self.weights.append(tmse_weight)

        if len(self.criterions) == 0:
            print("You have to choose at least one loss function.")
            sys.exit(1)

    def forward(self, preds, gts, feats):
        loss = 0.
        for criterion, weight in zip(self.criterions, self.weights):
            loss += weight * criterion(preds, gts)

        return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)