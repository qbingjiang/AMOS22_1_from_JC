import torch
import torch.nn as nn
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def dice_loss(self, target, predictive, ep=1e-8):
        intersection = 2. * torch.sum(predictive * target) + ep
        union = torch.sum(predictive) + torch.sum(target) + ep
        loss = 1 - intersection / union
        return loss

    def forward(self, target, predictive):
        return self.dice_loss(target, predictive)


class Generalized_Dice_loss(nn.Module):
    def __init__(self, class_weight):
        super(Generalized_Dice_loss, self).__init__()
        self.class_weight = class_weight

    def Generalized_Dice_Loss(self, y_pred, y_true, class_weights, smooth=1.0):
        '''
        inputs:
            y_pred [batch, n_classes, x, y, z] probability
            y_true [batch, n_classes, x, y, z] one-hot code
            class_weights
            smooth = 1.0
        '''
        smooth = 1.
        loss = 0.
        n_classes = y_pred.shape[1]
        class_weights = np.asarray(class_weights, dtype=float)
        for c in range(0, n_classes):  # pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()

            # with weight
            w = class_weights[c] / class_weights.sum()
            loss += w * (1 - ((2. * intersection + smooth) /
                              (pred_flat.sum() + true_flat.sum() + smooth)))

        return loss

    def forward(self, y_pred, y_true):
        return self.Generalized_Dice_Loss(y_pred, y_true, self.class_weight)
