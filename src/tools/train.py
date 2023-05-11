import numpy as np
import torch
from torch import nn

from src.utils import AverageMeter, threshold_cos_embed_loss
from src.tools import ACC, RMSE, PCC, CCC, SAGR
from .loss import dyn_wt_mse_ccc_loss_va, dyn_wt_mse_ccc_loss

np.random.seed(5)
torch.manual_seed(5)


def train(model, device, train_loader, optimizer, epoch, logger, log_interval,
          plotter, cfg, scheduler, affect=None):
    model.train()
    running_train_loss = 0.0
    losses = AverageMeter()
    losses_1 = AverageMeter()
    losses_2 = AverageMeter()
    losses_3 = AverageMeter()
    losses_4 = AverageMeter()
    keys = ['Accuracy']

    metrics = {key: AverageMeter() for key in keys}

    criterion_sim = nn.CrossEntropyLoss()
    criterion_cos_embed = nn.CosineEmbeddingLoss(margin=cfg['MARGIN'])


    for batch_idx, data_dict in enumerate(train_loader):
        # get images from data loader
        images1, images2 = data_dict['image1'].to(device), data_dict['image2'].to(device)

        # get targets
        target = data_dict['target'].to(device)
        # delta = data_dict[f'delta_{affect}'].to(device)
        delta_valence = data_dict['delta_valence'].to(device)
        delta_arousal = data_dict['delta_arousal'].to(device)

        # Initialise optimiser
        optimizer.zero_grad()

        # get predictions
        output_dict = model(images1, images2)

        # ------- Loss -------- #

        loss_1 = dyn_wt_mse_ccc_loss(2 * torch.tanh(output_dict['target0'][:, 0]), delta_valence, epoch=epoch,
                                     max_epochs=cfg['NUM_EPOCHS'], clamp=False)

        loss_2 = criterion_sim(output_dict['target'], target)

        loss_3 = dyn_wt_mse_ccc_loss(2 * torch.tanh(output_dict['target2'][:, 0]), delta_arousal, epoch=epoch,
                                     max_epochs=cfg['NUM_EPOCHS'], clamp=False)

        # Cosine embedding loss
        # Convert zeros to ones
        target_temp = target.clone()
        target_temp[target_temp == 0] = -1
        # Noramlise to [0,1]
        loss_4 = criterion_cos_embed(output_dict['output1'], output_dict['output2'], target_temp)

        # Total loss
        # loss = loss_cat + loss_delta + loss_cosine
        a, b, c, d = np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)
        weights = (a, b, c, d)
        weights = tuple(w / sum(weights) for w in weights)
        # Add weights
        loss = weights[0] * loss_1 + weights[1] * loss_2 + weights[2] * loss_3 + weights[3] * loss_4

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

        # Log
        losses.update(loss.item(), target.size(0))
        losses_1.update(loss_1.item(), target.size(0))
        losses_2.update(loss_2.item(), target.size(0))
        losses_3.update(loss_3.item(), target.size(0))
        losses_4.update(loss_4.item(), target.size(0))

        # Metrics
        _, predicted = torch.max(output_dict['target'].data, 1)

        acc = ACC(predicted.clone().detach().cpu().numpy(), target.clone().detach().cpu().numpy())

        metrics['Accuracy'].update(acc, target.size(0))

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_train_loss / log_interval
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(images1), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader),
                avg_loss))
            running_train_loss = 0.0

    logger.info(
        'Train set ({:d} samples): Average Loss: {:.4f}\tAccuracy: {:.4f}'.format(
            len(train_loader.dataset), losses.avg, metrics['Accuracy'].avg))

    plotter.plot('Loss (sum)', 'train', 'Total Loss', epoch, losses.avg)
    plotter.plot('Loss (Valence)', 'train', 'Reg Loss', epoch, losses_1.avg)
    plotter.plot('Loss (Similarity)', 'train', 'CCE Loss', epoch, losses_2.avg)
    plotter.plot('Loss (Arousal)', 'train', 'Reg Loss', epoch, losses_3.avg)
    plotter.plot('Loss (Contrastive)', 'train', 'Contrastive Loss', epoch, losses_4.avg)
    plotter.plot('Learning rate', 'train', 'Learning rate', epoch, optimizer.param_groups[0]["lr"])
    plotter.plot('Accuracy', 'train', 'Accuracy', epoch, metrics['Accuracy'].avg)

    return {'Loss': losses.avg, 'metrics': metrics}


def train3(model, device, train_loader, optimizer, epoch, logger, log_interval, plotter, cfg, clamp):

    model.train()
    running_train_loss = 0.0
    losses = AverageMeter()

    valence_true = []
    valence_pred = []

    arousal_true = []
    arousal_pred = []

    for batch_idx, data_dict in enumerate(train_loader):
        inputs, valence, arousal = data_dict['feats'].to(device), data_dict['valence'].to(device),\
                                   data_dict['arousal'].to(device)

        labels = torch.stack((valence, arousal), dim=1)

        # Initialise optimiser
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = dyn_wt_mse_ccc_loss_va(outputs, labels, epoch=epoch,
                                     max_epochs=cfg['NUM_EPOCHS'], weight_exponent=cfg["WEIGHTS_EXPONENT"], clamp=False)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        # Log
        losses.update(loss.item(), labels.size(0))

        if clamp:
            val_pred, val_true = outputs[:, 0].clamp(-1, 1).cpu().data, labels[:, 0].clamp(-1, 1).cpu().data
            arsl_pred, arsl_true = outputs[:, 1].clamp(-1, 1).cpu().data, labels[:, 1].clamp(-1, 1).cpu().data
        else:
            val_pred, val_true = outputs[:, 0].cpu().data, labels[:, 0].cpu().data
            arsl_pred, arsl_true = outputs[:, 1].cpu().data, labels[:, 1].cpu().data

        valence_true.append(val_true)
        valence_pred.append(val_pred)

        arousal_true.append(arsl_true)
        arousal_pred.append(arsl_pred)

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_train_loss / log_interval
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(inputs), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader),
                avg_loss))
            running_train_loss = 0.0

    valence_true = torch.stack(valence_true, dim=1).view(-1, ).numpy()
    valence_pred = torch.stack(valence_pred, dim=1).view(-1, ).numpy()
    arousal_true = torch.stack(arousal_true, dim=1).view(-1, ).numpy()
    arousal_pred = torch.stack(arousal_pred, dim=1).view(-1, ).numpy()

    rmse_v = RMSE(valence_true, valence_pred)
    pcc_v = PCC(valence_true, valence_pred)
    ccc_v = CCC(valence_true, valence_pred)
    sagr_v = SAGR(valence_true, valence_pred)

    rmse_a = RMSE(arousal_true, arousal_pred)
    pcc_a = PCC(arousal_true, arousal_pred)
    ccc_a = CCC(arousal_true, arousal_pred)
    sagr_a = SAGR(arousal_true, arousal_pred)


    logger.info(
        'Train set ({:d} samples): Average Loss: {:.4f}'.format(
            len(train_loader.dataset), losses.avg))
    logger.info(
        '****** Valence *****\tRMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}'.format(
            rmse_v, pcc_v, ccc_v, sagr_v))
    logger.info(
        '****** Arousal *****\tRMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}'.format(
            rmse_a, pcc_a, ccc_a, sagr_a))


    plotter.plot('Loss', 'train', 'Total Loss', epoch, losses.avg)
    plotter.plot('RMSE (Valence)', 'train', 'RMSE', epoch, rmse_v)
    plotter.plot('RMSE (Arousal)', 'train', 'RMSE', epoch, rmse_a)
    plotter.plot('PCC (Valence)', 'train', 'PCC', epoch, pcc_v)
    plotter.plot('PCC (Arousal)', 'train', 'PCC', epoch, pcc_a)
    plotter.plot('CCC (Valence)', 'train', 'CCC', epoch, ccc_v)
    plotter.plot('CCC (Arousal)', 'train', 'CCC', epoch, ccc_a)
    plotter.plot('SAGR (Valence)', 'train', 'SAGR', epoch, sagr_v)
    plotter.plot('SAGR (Arousal)', 'train', 'SAGR', epoch, sagr_a)
    plotter.plot('Learning rate', 'train', 'Learning rate', epoch, optimizer.param_groups[0]["lr"])

    keys = ['RMSE_valence', 'PCC_valence', 'CCC_valence', 'SAGR_valence',
            'RMSE_arousal', 'PCC_arousal', 'CCC_arousal', 'SAGR_arousal']
    #rmse_a, pcc_a, ccc_a, sagr_a = 0, 0, 0, 0
    values = [rmse_v, pcc_v, ccc_v, sagr_v, rmse_a, pcc_a, ccc_a, sagr_a]
    metrics = dict(zip(keys, values))

    return {'Loss': losses.avg, 'metrics': metrics}


def train3a(model, device, train_loader, optimizer, epoch, logger, log_interval, plotter, cfg, clamp):

    model.train()
    running_train_loss = 0.0
    losses = AverageMeter()
    keys = ['Accuracy']

    metrics = {key: AverageMeter() for key in keys}

    criterion = nn.CrossEntropyLoss()

    for batch_idx, data_dict in enumerate(train_loader):
        inputs, labels = data_dict['feats'].to(device), data_dict['categories'].to(device)

        # Initialise optimiser
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        # Log
        losses.update(loss.item(), labels.size(0))

        _, predicted = torch.max(outputs.data, 1)

        acc = ACC(predicted.clone().detach().cpu().numpy(), labels.clone().detach().cpu().numpy())

        metrics['Accuracy'].update(acc, labels.size(0))

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_train_loss / log_interval
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(inputs), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader),
                avg_loss))
            running_train_loss = 0.0

    logger.info(
        'Train set ({:d} samples): Average Loss: {:.4f}\tAccuracy: {:.4f}'.format(
            len(train_loader.dataset), losses.avg, metrics['Accuracy'].avg))

    plotter.plot('Learning rate', 'train', 'Learning rate', epoch, optimizer.param_groups[0]["lr"])
    plotter.plot('Accuracy', 'train', 'Accuracy', epoch, metrics['Accuracy'].avg)

    plotter.plot('Loss', 'train', 'Total Loss', epoch, losses.avg)

    return {'Loss': losses.avg, 'metrics': metrics}
