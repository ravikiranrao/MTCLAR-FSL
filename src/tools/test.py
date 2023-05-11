import numpy as np
from tqdm import tqdm
import torch
from torch import nn

from src.utils import AverageMeter, threshold_cos_embed_loss
from src.tools import ACC, RMSE, PCC, CCC, SAGR
from .loss import ccc_loss_va, mse_ccc_loss_va, mse_pcc_ccc_loss_va, mse_pcc_ccc_loss, mse_loss_va,\
    dyn_wt_mse_ccc_loss, dyn_wt_mse_ccc_loss_va


def test(model, mode, device, test_loader, logger, batch_size, plotter=None, cfg=None, epoch=-1,
         affect=None):
    model.eval()

    target_true = np.zeros(shape=(len(test_loader), batch_size))
    target_pred = np.zeros(shape=(len(test_loader), batch_size))
    delta_valence_true = np.zeros(shape=(len(test_loader), batch_size))
    delta_valence_pred = np.zeros(shape=(len(test_loader), batch_size))
    delta_arousal_true = np.zeros(shape=(len(test_loader), batch_size))
    delta_arousal_pred = np.zeros(shape=(len(test_loader), batch_size))
    valence_true = np.zeros(shape=(len(test_loader), batch_size))
    arousal_true = np.zeros(shape=(len(test_loader), batch_size))
    valence_anchor = np.zeros(shape=(len(test_loader), batch_size))
    arousal_anchor = np.zeros(shape=(len(test_loader), batch_size))

    losses = AverageMeter()
    losses_1 = AverageMeter()
    losses_2 = AverageMeter()
    losses_3 = AverageMeter()
    losses_4 = AverageMeter()
    keys = ['Accuracy', 'RMSE_valence', 'PCC_valence', 'CCC_valence', 'SAGR_valence',
            'RMSE_arousal', 'PCC_arousal', 'CCC_arousal', 'SAGR_arousal']

    metrics = {key: 0 for key in keys}

    criterion_sim = nn.CrossEntropyLoss()
    criterion_cos_embed = nn.CosineEmbeddingLoss(margin=cfg['MARGIN'])

    all_pairs = []

    with torch.no_grad():
        for batch_idx, data_dict in enumerate(tqdm(test_loader)):
            # get images from data loader
            images1, images2 = data_dict['image1'].to(device), data_dict['image2'].to(device)

            target = data_dict['target'].flatten().to(device)
            # delta = data_dict[f'delta_{affect}'].to(device)
            delta_valence = data_dict['delta_valence'].to(device)
            delta_arousal = data_dict['delta_arousal'].to(device)

            # get Pairs
            pairs = data_dict['categories']

            all_pairs.append(pairs)

            # get predictions
            output_dict = model(images1, images2)

            # ------- Loss -------- #

            loss_1 = dyn_wt_mse_ccc_loss(2 * torch.tanh(output_dict['target0'][:, 0]), delta_valence, epoch=1,
                                         max_epochs=2, weight_exponent=1, clamp=False)

            loss_2 = criterion_sim(output_dict['target'], target)

            loss_3 = dyn_wt_mse_ccc_loss(2 * torch.tanh(output_dict['target2'][:, 0]), delta_arousal, epoch=1,
                                         max_epochs=2, weight_exponent=1, clamp=False)

            # Cosine embedding loss
            # Convert zeros to ones
            target_temp = target.clone()
            target_temp[target_temp == 0] = -1
            # Noramlise to [0,1]
            loss_4 = criterion_cos_embed(output_dict['output1'], output_dict['output2'], target_temp)

            # Total loss
            # loss = loss_cat + loss_delta + loss_cosine
            a, b, c, d = 0.5, 0.5, 0.5, 0.5
            weights = (a, b, c, d)
            weights = tuple(w / sum(weights) for w in weights)
            # Add weights
            loss = weights[0] * loss_1 + weights[1] * loss_2 + weights[2] * loss_3 + weights[3] * loss_4

            losses.update(loss.item(), target.size(0))
            losses_1.update(loss_1.item(), target.size(0))
            losses_2.update(loss_1.item(), target.size(0))
            losses_3.update(loss_3.item(), target.size(0))
            losses_4.update(loss_1.item(), target.size(0))

            target_true[batch_idx, :] = target.clone().detach().cpu().numpy()
            _, target_pred[batch_idx, :] = torch.max(output_dict['target'].cpu().data, 1)

            # Valence
            delta_valence_true[batch_idx, :] = delta_valence.clone().detach().cpu().numpy()
            delta_valence_pred[batch_idx, :] = (2 * torch.tanh(output_dict['target0'][:, 0])).clone().detach().cpu().numpy()

            # Arousal
            delta_arousal_true[batch_idx, :] = delta_arousal.clone().detach().cpu().numpy()
            delta_arousal_pred[batch_idx, :] = (2 * torch.tanh(output_dict['target2'][:, 0])).clone().detach().cpu().numpy()

        target_true = np.squeeze(np.asarray(target_true)).flatten()
        target_pred = np.squeeze(np.asarray(target_pred)).flatten()

    all_pairs = torch.cat(all_pairs, 0)
    categories_pair = all_pairs.numpy()

    metrics['Accuracy'] = ACC(target_true, target_pred)

    logger.info(
        '{} set ({:d} samples): Average Loss: {:.4f}\tAccuracy: {:.4f}'.format(
            mode.capitalize(), len(test_loader.dataset), losses.avg, metrics['Accuracy']))

    if mode == "validation":
        assert epoch > 0
        assert plotter is not None

        plotter.plot('Loss (sum)', 'val', 'Total Loss', epoch, losses.avg)
        plotter.plot('Loss (Valence)', 'val', 'MSE Loss', epoch, losses_1.avg)
        plotter.plot('Loss (Similarity)', 'val', 'CCE Loss', epoch, losses_2.avg)
        plotter.plot('Loss (Arousal)', 'val', 'MSE Loss', epoch, losses_3.avg)
        plotter.plot('Loss (Contrastive)', 'val', 'Contrastive Loss', epoch, losses_4.avg)
        plotter.plot('RMSE (Valence)', 'val', 'RMSE', epoch, metrics['RMSE_valence'])
        plotter.plot('RMSE (Arousal)', 'val', 'RMSE', epoch, metrics['RMSE_arousal'])
        plotter.plot('PCC (Valence)', 'val', 'PCC', epoch, metrics['PCC_valence'])
        plotter.plot('PCC (Arousal)', 'val', 'PCC', epoch, metrics['PCC_arousal'])
        plotter.plot('CCC (Valence)', 'val', 'CCC', epoch, metrics['CCC_valence'])
        plotter.plot('CCC (Arousal)', 'val', 'CCC', epoch, metrics['CCC_arousal'])
        plotter.plot('SAGR (Valence)', 'val', 'SAGR', epoch, metrics['SAGR_valence'])
        plotter.plot('SAGR (Arousal)', 'val', 'SAGR', epoch, metrics['SAGR_arousal'])
        plotter.plot('Accuracy', 'val', 'Accuracy', epoch, metrics['Accuracy'])
        plotter.conf_mat(target_true, target_pred, labels=[1, 0],
                         display_labels=["Similar", "Dissimilar"], epoch=epoch,
                         title=f'Accuracy')

    return {'Loss': losses.avg, 'metrics': metrics,
            'target_true': target_true, 'target_pred': target_pred,
            'categories_pair': categories_pair,
            'delta_valence_true':delta_valence_true, 'delta_valence_pred': delta_valence_pred,
            'delta_arousal_true': delta_arousal_true, 'delta_arousal_pred': delta_arousal_pred
            }


def test3(model, mode, device, test_loader,logger, cfg, clamp, epoch=-1, plotter=None):
    model.eval()

    losses = AverageMeter()

    valence_true = []
    valence_pred = []

    arousal_true = []
    arousal_pred = []

    with torch.no_grad():
        for batch_idx, data_dict in enumerate(tqdm(test_loader)):
            inputs, valence, arousal = data_dict['feats'].to(device), data_dict['valence'].to(device), \
                                                   data_dict['arousal'].to(device)

            labels = torch.stack((valence, arousal), dim=1)

            # forward
            outputs = model(inputs)

            loss = dyn_wt_mse_ccc_loss_va(outputs, labels, epoch=1,
                                         max_epochs=2, weight_exponent=2, clamp=False)

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
        '{} set ({:d} samples): Average Loss: {:.4f}'.format(mode.capitalize(),
            len(test_loader.dataset), losses.avg))
    logger.info(
        '****** Valence *****\tRMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}'.format(
            rmse_v, pcc_v, ccc_v, sagr_v))
    logger.info(
        '****** Arousal *****\tRMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}'.format(
            rmse_a, pcc_a, ccc_a, sagr_a))
    if mode == "validation":
        assert epoch > 0
        assert plotter is not None
        plotter.plot('Loss', 'val', 'Total Loss', epoch, losses.avg)
        plotter.plot('RMSE (Valence)', 'val', 'RMSE', epoch, rmse_v)
        plotter.plot('RMSE (Arousal)', 'val', 'RMSE', epoch, rmse_a)
        plotter.plot('PCC (Valence)', 'val', 'PCC', epoch, pcc_v)
        plotter.plot('PCC (Arousal)', 'val', 'PCC', epoch, pcc_a)
        plotter.plot('CCC (Valence)', 'val', 'CCC', epoch, ccc_v)
        plotter.plot('CCC (Arousal)', 'val', 'CCC', epoch, ccc_a)
        plotter.plot('SAGR (Valence)', 'val', 'SAGR', epoch, sagr_v)
        plotter.plot('SAGR (Arousal)', 'val', 'SAGR', epoch, sagr_a)
        plotter.scatter(valence_true, valence_pred, "Valence", "valence", epoch)
        plotter.scatter(arousal_true, arousal_pred, "Arousal", "arousal", epoch)

    keys = ['RMSE_valence', 'PCC_valence', 'CCC_valence', 'SAGR_valence',
            'RMSE_arousal', 'PCC_arousal', 'CCC_arousal', 'SAGR_arousal']

    values = [rmse_v, pcc_v, ccc_v, sagr_v, rmse_a, pcc_a, ccc_a, sagr_a]
    metrics = dict(zip(keys, values))

    return {'Loss': losses.avg, 'metrics': metrics, 'valence_true': valence_true, 'valence_pred': valence_pred,
            'arousal_true': arousal_true, 'arousal_pred': arousal_pred}


def test3a(model, mode, device, test_loader,logger, cfg, clamp, epoch=-1, plotter=None):
    model.eval()

    losses = AverageMeter()

    keys = ['Accuracy']

    metrics = {key: AverageMeter() for key in keys}

    target_true = np.zeros(shape=(len(test_loader), cfg["BATCH_SIZE"]))
    target_pred = np.zeros(shape=(len(test_loader), cfg["BATCH_SIZE"]))

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, data_dict in enumerate(tqdm(test_loader)):
            inputs, labels = data_dict['feats'].to(device), data_dict['categories'].to(device)

            # forward
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            # Log
            losses.update(loss.item(), labels.size(0))

            target_true[batch_idx, :] = labels.clone().detach().cpu().numpy()

            _, target_pred[batch_idx, :] = torch.max(outputs.cpu().data, 1)

    target_true = np.squeeze(np.asarray(target_true)).flatten()
    target_pred = np.squeeze(np.asarray(target_pred)).flatten()

    metrics['Accuracy'] = ACC(target_true, target_pred)

    logger.info(
        '{} set ({:d} samples): Average Loss: {:.4f}\tAccuracy: {:.4f}'.format(
            mode.capitalize(), len(test_loader.dataset), losses.avg, metrics['Accuracy']))


    if mode == "validation":
        assert epoch > 0
        assert plotter is not None
        plotter.plot('Loss', 'val', 'Total Loss', epoch, losses.avg)
        plotter.plot('Accuracy', 'val', 'Accuracy', epoch, metrics['Accuracy'])
        plotter.conf_mat(target_true, target_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7],
                         display_labels=["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"],
                         epoch=epoch,
                         title=f'Accuracy')

    return {'Loss': losses.avg, 'metrics': metrics, 'target_true': target_true, 'target_pred': target_pred}


def test4(model, mode, device, test_loader, logger, batch_size, plotter=None, cfg=None, epoch=-1,
         affect=None):
    model.eval()

    target_true = np.zeros(shape=(len(test_loader), batch_size))
    target_pred = np.zeros(shape=(len(test_loader), batch_size))
    delta_valence_true = np.zeros(shape=(len(test_loader), batch_size))
    delta_valence_pred = np.zeros(shape=(len(test_loader), batch_size))
    delta_arousal_true = np.zeros(shape=(len(test_loader), batch_size))
    delta_arousal_pred = np.zeros(shape=(len(test_loader), batch_size))
    valence_true = np.zeros(shape=(len(test_loader), batch_size))
    arousal_true = np.zeros(shape=(len(test_loader), batch_size))
    valence_anchor = np.zeros(shape=(len(test_loader), batch_size))
    arousal_anchor = np.zeros(shape=(len(test_loader), batch_size))

    losses = AverageMeter()
    losses_1 = AverageMeter()
    losses_2 = AverageMeter()
    losses_3 = AverageMeter()
    losses_4 = AverageMeter()
    keys = ['Accuracy', 'RMSE_valence', 'PCC_valence', 'CCC_valence', 'SAGR_valence',
            'RMSE_arousal', 'PCC_arousal', 'CCC_arousal', 'SAGR_arousal']

    metrics = {key: 0 for key in keys}

    all_pairs = []

    with torch.no_grad():
        for batch_idx, data_dict in enumerate(tqdm(test_loader)):
            # get images from data loader
            images1, images2 = data_dict['image1'].to(device), data_dict['image2'].to(device)

            target = data_dict['target'].flatten().to(device)
            # delta = data_dict[f'delta_{affect}'].to(device)
            delta_valence = data_dict['delta_valence'].to(device)
            delta_arousal = data_dict['delta_arousal'].to(device)

            # get Pairs
            pairs = data_dict['categories']

            all_pairs.append(pairs)

            # get predictions
            output_dict = model(images1, images2)

            # Valence
            delta_valence_true[batch_idx, :] = delta_valence.clone().detach().cpu().numpy()
            delta_valence_pred[batch_idx, :] = (2 * torch.tanh(output_dict['target0'][:, 0])).clone().detach().cpu().numpy()

            # Arousal
            delta_arousal_true[batch_idx, :] = delta_arousal.clone().detach().cpu().numpy()
            delta_arousal_pred[batch_idx, :] = (2 * torch.tanh(output_dict['target2'][:, 0])).clone().detach().cpu().numpy()

            valence_true[batch_idx, :] = data_dict['valence'][:, 0].numpy()
            arousal_true[batch_idx, :] = data_dict['arousal'][:, 0].numpy()

            valence_anchor[batch_idx, :] = data_dict['valence'][:, 1].numpy()
            arousal_anchor[batch_idx, :] = data_dict['arousal'][:, 1].numpy()

        target_true = np.squeeze(np.asarray(target_true)).flatten()
        target_pred = np.squeeze(np.asarray(target_pred)).flatten()

        valence_true = np.nan_to_num(np.squeeze(np.asarray(valence_true)).flatten())
        arousal_true = np.nan_to_num(np.squeeze(np.asarray(arousal_true)).flatten())

        valence_pred = np.clip(np.squeeze(np.asarray(delta_valence_pred)).flatten() + np.squeeze(np.asarray(valence_anchor)).flatten(), -1, 1)
        arousal_pred = np.clip(np.squeeze(np.asarray(delta_arousal_pred)).flatten() + np.squeeze(np.asarray(arousal_anchor)).flatten() , -1, 1)

        rmse_v = RMSE(valence_true, valence_pred)
        pcc_v = PCC(valence_true, valence_pred)
        ccc_v = CCC(valence_true, valence_pred)
        sagr_v = SAGR(valence_true, valence_pred)

        rmse_a = RMSE(arousal_true, arousal_pred)
        pcc_a = PCC(arousal_true, arousal_pred)
        ccc_a = CCC(arousal_true, arousal_pred)
        sagr_a = SAGR(arousal_true, arousal_pred)

        logger.info(
            '****** Valence from Anchor*****\tRMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}'.format(
                rmse_v, pcc_v, ccc_v, sagr_v))
        logger.info(
            '****** Arousal from Anchor *****\tRMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}'.format(
                rmse_a, pcc_a, ccc_a, sagr_a))

    all_pairs = torch.cat(all_pairs, 0)
    categories_pair = all_pairs.numpy()

    metrics['RMSE_valence'] = RMSE(delta_valence_true, delta_valence_pred)
    metrics['RMSE_arousal'] = RMSE(delta_arousal_true, delta_arousal_pred)

    metrics['PCC_valence'] = PCC(delta_valence_true, delta_valence_pred)
    metrics['PCC_arousal'] = PCC(delta_arousal_true, delta_arousal_pred)

    metrics['CCC_valence'] = CCC(delta_valence_true, delta_valence_pred)
    metrics['CCC_arousal'] = CCC(delta_arousal_true, delta_arousal_pred)

    metrics['SAGR_valence'] = SAGR(delta_valence_true, delta_valence_pred)
    metrics['SAGR_arousal'] = SAGR(delta_arousal_true, delta_arousal_pred)


    logger.info(
        '****** Delta Valence *****\tRMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}'.format(
            metrics['RMSE_valence'], metrics['PCC_valence'], metrics['CCC_valence'],
            metrics['SAGR_valence']))
    logger.info(
        '****** Delta Arousal *****\tRMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}'.format(
            metrics['RMSE_arousal'], metrics['PCC_arousal'], metrics['CCC_arousal'],
            metrics['SAGR_arousal']))


    if mode == "validation":
        assert epoch > 0
        assert plotter is not None

        plotter.plot('Loss (sum)', 'val', 'Total Loss', epoch, losses.avg)
        plotter.plot('Loss (Valence)', 'val', 'MSE Loss', epoch, losses_1.avg)
        plotter.plot('Loss (Similarity)', 'val', 'CCE Loss', epoch, losses_2.avg)
        plotter.plot('Loss (Arousal)', 'val', 'MSE Loss', epoch, losses_3.avg)
        plotter.plot('Loss (Contrastive)', 'val', 'Contrastive Loss', epoch, losses_4.avg)
        plotter.plot('RMSE (Valence)', 'val', 'RMSE', epoch, metrics['RMSE_valence'])
        plotter.plot('RMSE (Arousal)', 'val', 'RMSE', epoch, metrics['RMSE_arousal'])
        plotter.plot('PCC (Valence)', 'val', 'PCC', epoch, metrics['PCC_valence'])
        plotter.plot('PCC (Arousal)', 'val', 'PCC', epoch, metrics['PCC_arousal'])
        plotter.plot('CCC (Valence)', 'val', 'CCC', epoch, metrics['CCC_valence'])
        plotter.plot('CCC (Arousal)', 'val', 'CCC', epoch, metrics['CCC_arousal'])
        plotter.plot('SAGR (Valence)', 'val', 'SAGR', epoch, metrics['SAGR_valence'])
        plotter.plot('SAGR (Arousal)', 'val', 'SAGR', epoch, metrics['SAGR_arousal'])
        plotter.plot('Accuracy', 'val', 'Accuracy', epoch, metrics['Accuracy'])
        plotter.conf_mat(target_true, target_pred, labels=[1, 0],
                         display_labels=["Similar", "Dissimilar"], epoch=epoch,
                         title=f'Accuracy')

    return {'Loss': losses.avg, 'metrics': metrics,
            'target_true': target_true, 'target_pred': target_pred,
            'categories_pair': categories_pair,
            'delta_valence_true':delta_valence_true, 'delta_valence_pred': delta_valence_pred,
            'delta_arousal_true': delta_arousal_true, 'delta_arousal_pred': delta_arousal_pred,
            'valence_true': valence_true, 'valence_pred': valence_pred, 'arousal_true': arousal_true,
            'arousal_pred': arousal_pred
            }


def test5(model, mode, device, test_loader, logger, batch_size, plotter=None, cfg=None, epoch=-1,
         affect=None):
    model.eval()
    #batch_size=1       # set batch size = 1 for testing "mean" of all the anchors.
    target_true = np.zeros(shape=(len(test_loader), batch_size))
    target_pred = np.zeros(shape=(len(test_loader), batch_size))
    delta_valence_true = np.zeros(shape=(len(test_loader), batch_size))
    delta_valence_pred = np.zeros(shape=(len(test_loader), batch_size))
    delta_arousal_true = np.zeros(shape=(len(test_loader), batch_size))
    delta_arousal_pred = np.zeros(shape=(len(test_loader), batch_size))
    valence_true = np.zeros(shape=(len(test_loader), batch_size))
    arousal_true = np.zeros(shape=(len(test_loader), batch_size))
    valence_anchor = np.zeros(shape=(len(test_loader), batch_size))
    arousal_anchor = np.zeros(shape=(len(test_loader), batch_size))

    losses = AverageMeter()

    keys = ['Accuracy', 'RMSE_valence', 'PCC_valence', 'CCC_valence', 'SAGR_valence',
            'RMSE_arousal', 'PCC_arousal', 'CCC_arousal', 'SAGR_arousal']

    metrics = {key: 0 for key in keys}

    with torch.no_grad():
        for batch_idx, data_dict in enumerate(tqdm(test_loader)):
            # get images from data loader
            images1, images2 = data_dict['image1'].to(device), data_dict['image2'].to(device)

            # get predictions
            output_dict = model(images1, images2)

            # Valence

            delta_valence_pred[batch_idx, :] = np.mean((2 * torch.tanh(output_dict['target0'][:, 0])).clone().detach().cpu().numpy())

            # Arousal
            delta_arousal_pred[batch_idx, :] = np.mean((2 * torch.tanh(output_dict['target2'][:, 0])).clone().detach().cpu().numpy())

            valence_true[batch_idx, :] = data_dict['valence'][:, 0].numpy()
            arousal_true[batch_idx, :] = data_dict['arousal'][:, 0].numpy()

            valence_anchor[batch_idx, :] = data_dict['valence'][:, 1].numpy()
            arousal_anchor[batch_idx, :] = data_dict['arousal'][:, 1].numpy()

        valence_true = np.nan_to_num(np.squeeze(np.asarray(valence_true)).flatten())
        arousal_true = np.nan_to_num(np.squeeze(np.asarray(arousal_true)).flatten())

        valence_pred = np.clip(np.squeeze(np.asarray(delta_valence_pred)).flatten() + np.squeeze(np.asarray(valence_anchor)).flatten(), -1, 1)
        arousal_pred = np.clip(np.squeeze(np.asarray(delta_arousal_pred)).flatten() + np.squeeze(np.asarray(arousal_anchor)).flatten() , -1, 1)

        rmse_v = RMSE(valence_true, valence_pred)
        pcc_v = PCC(valence_true, valence_pred)
        ccc_v = CCC(valence_true, valence_pred)
        sagr_v = SAGR(valence_true, valence_pred)

        rmse_a = RMSE(arousal_true, arousal_pred)
        pcc_a = PCC(arousal_true, arousal_pred)
        ccc_a = CCC(arousal_true, arousal_pred)
        sagr_a = SAGR(arousal_true, arousal_pred)

        logger.info(
            '****** Valence from Anchor*****\tRMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}'.format(
                rmse_v, pcc_v, ccc_v, sagr_v))
        logger.info(
            '****** Arousal from Anchor *****\tRMSE: {:.4f}\tPCC: {:.4f}\tCCC: {:.4f}\tSAGR: {:.4f}'.format(
                rmse_a, pcc_a, ccc_a, sagr_a))


    return {'Loss': losses.avg, 'metrics': metrics,
            'target_true': target_true, 'target_pred': target_pred,
            #'categories_pair': categories_pair,
            'delta_valence_true':delta_valence_true, 'delta_valence_pred': delta_valence_pred,
            'delta_arousal_true': delta_arousal_true, 'delta_arousal_pred': delta_arousal_pred,
            'valence_true': valence_true, 'valence_pred': valence_pred, 'arousal_true': arousal_true,
            'arousal_pred': arousal_pred
            }
