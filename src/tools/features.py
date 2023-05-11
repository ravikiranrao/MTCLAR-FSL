from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor

from src.data import AffectnetIdenticalPair, AfewvaIdenticalPair


def get_features(model, data, data_root, path_to_csv, cfg, augment, get_more_data_num, device, logger, save_path=None):

    # new_model = create_feature_extractor(model, {'proj.4': 'feat1', 'fc.2': 'feat2', 'fc.4': 'feat3'})
    # new_model = create_feature_extractor(model, {'emo_net.view_1': 'feat1', 'fc.16': 'feat2', 'fc.17': 'feat3'})
    new_model = create_feature_extractor(model, {'emo_net.view_1': 'feat1', 'fc.4': 'feat2', 'fc.7': 'feat3'})
    new_model.eval()

    if data == "affectnet":
        dataset = AffectnetIdenticalPair(data_root, path_to_csv,
                                     image_shape=(3, cfg['IMAGE_SHAPE'], cfg['IMAGE_SHAPE']),
                                     augment=False)
    elif data == "afewva":
        dataset = AfewvaIdenticalPair(data_root, path_to_csv, image_shape=(3, cfg['IMAGE_SHAPE'], cfg['IMAGE_SHAPE']),
                                     augment=augment)
    else:
        raise ValueError('Unsupported value for data. Got {}'.format(data))

    dataloader = DataLoader(dataset, batch_size=32, num_workers=20, shuffle=False, drop_last=True)

    feat_dict_keys = ['features', 'categories', 'valence', 'arousal', 'predictions']
    feat_dict = {key: [] for key in feat_dict_keys}

    for epoch in range(0, get_more_data_num):
        # loop through batches
        for idx, data_dict in enumerate(tqdm(dataloader)):
            # get images from data loader
            images1, images2 = data_dict['image1'].to(device), data_dict['image2'].to(device)

            feat_dict['categories'].append(data_dict['categories'][:, 0].numpy())
            # since both images have same labels, take only one

            feat_dict['valence'].append(data_dict['valence'][:, 0].numpy())

            feat_dict['arousal'].append(data_dict['arousal'][:, 0].numpy())

            with torch.no_grad():
                # forward pass [with feature extraction]
                preds = new_model(images1, images2)
                # logger.info(f'Features shape\n{[(k, v.shape) for k, v in preds.items()]}')

                # add feats and preds to lists
                feat_dict['predictions'].append(preds['feat3'].detach().cpu().numpy())
                feat_dict['features'].append(preds['feat1'].detach().cpu().numpy())

    # flatten features and inspect
    for key in feat_dict.keys():
        feat_dict[key] = np.concatenate(feat_dict[key])
        logger.info(f'-- {key} shape: {feat_dict[key].shape}')

    if save_path is not None:
        import pickle
        logger.info(f'Saving features to {save_path}')
        with open(save_path, 'wb') as f:
            pickle.dump(feat_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return feat_dict
