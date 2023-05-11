from __future__ import print_function

import torch


def mse_loss_va(x, y, clamp=False):

    if clamp:
        val_pred, val_true = x[:, 0].clamp(-1, 1), y[:, 0].clamp(-1, 1)
        arsl_pred, arsl_true = x[:, 1].clamp(-1, 1), y[:, 1].clamp(-1, 1)
    else:
        val_pred, val_true = x[:, 0], y[:, 0]
        arsl_pred, arsl_true = x[:, 1], y[:, 1]

    mse_v = torch.nn.functional.mse_loss(val_pred, val_true)

    mse_a = torch.nn.functional.mse_loss(arsl_pred, arsl_true)

    loss = mse_v + mse_a

    return loss


def ccc(x, y):
    pcc = torch.corrcoef(torch.stack((x, y), dim=0))[0, 1]
    num = 2 * pcc * x.std() * y.std()
    den = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    ccc = num / den
    return torch.nan_to_num(ccc, nan=0)


def ccc_loss_va(x, y, clamp=False):
    # x and y shape: (bs, 2)
    # first dimension for valence, second for arousal

    if clamp:
        val_pred, val_true = x[:, 0].clamp(-1, 1), y[:, 0].clamp(-1, 1)
        arsl_pred, arsl_true = x[:, 1].clamp(-1, 1), y[:, 1].clamp(-1, 1)
    else:
        val_pred, val_true = x[:, 0], y[:, 0]
        arsl_pred, arsl_true = x[:, 1], y[:, 1]

    ccc_v = ccc(val_pred, val_true)
    ccc_a = ccc(arsl_pred, arsl_true)

    loss = 1 - 0.5 * (ccc_v + ccc_a)

    return loss


def mse_ccc_loss_va(x, y, weights=(1, 1), clamp=False):
    loss = (weights[0] * mse_loss_va(x, y, clamp)) +\
           (weights[1] * ccc_loss_va(x, y, clamp))
    return loss


def dyn_wt_mse_ccc_loss_va(x, y, epoch, max_epochs, alpha=1, weight_exponent=2, clamp=False):

    weights = (alpha * ((epoch/max_epochs)**weight_exponent), 1.0 - ((epoch/max_epochs)**weight_exponent))
    loss = (weights[0] * mse_loss_va(x, y, clamp)) +\
           (weights[1] * ccc_loss_va(x, y, clamp))

    return loss


def dyn_wt_mse_ccc_loss(x, y, epoch, max_epochs, alpha=1, weight_exponent=2, clamp=False):

    weights = (alpha * ((epoch/max_epochs)**weight_exponent), 1.0 - ((epoch/max_epochs)**weight_exponent))

    if clamp:
        x, y = x.clamp(-1, 1), y.clamp(-1, 1)

    loss = (weights[0] * torch.nn.functional.mse_loss(x, y)) +\
           (weights[1] * (1.0 - ccc(x, y)))

    return loss
