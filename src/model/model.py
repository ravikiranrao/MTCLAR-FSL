from .siamese_network import SiameseNetEmoNetMLP
from .linear_net import LinearNet


def get_model(name, cfg, logger):
    if name == "SiameseNetEmoNetMLPMultitask":
        num_neurons = (1, 2, 1)
        emofan_n_emotions = 8
        model = SiameseNetEmoNetMLP(emofan_n_emotions, cfg["IS_PRETRAINED"], cfg["FEAT_FUSION"], num_neurons,
                                    cfg["DROPOUT_RATE"],
                                    is_multi_task=True)
        logger.info(f'Building model: \nSiameseNet EmoFAN MultiTask, Feat fusion: {cfg["FEAT_FUSION"]},'
                    f'Num neurons: {num_neurons}, Dropout rate: {cfg["DROPOUT_RATE"]},'
                    f' emofan_n_emotions: {emofan_n_emotions}\n\n')
        logger.info(f'{model}')
        # print the number of parameters in the model
        logger.info(f'{sum(p.numel() for p in model.parameters()) / 1e6} M parameters')

    elif name == "SiameseNetEmoNetMLP":
        num_neurons = 2
        emofan_n_emotions = 8
        model = SiameseNetEmoNetMLP(emofan_n_emotions, cfg["IS_PRETRAINED"], cfg["FEAT_FUSION"], num_neurons,
                                    cfg["DROPOUT_RATE"],
                                    is_multi_task=False)
        logger.info(f'Building model: \nSiameseNet EmoFAN, Feat fusion: {cfg["FEAT_FUSION"]},'
                    f'Num neurons: {num_neurons}, Dropout rate: {cfg["DROPOUT_RATE"]},'
                    f' emofan_n_emotions: {emofan_n_emotions}\n\n')
        logger.info(f'{model}')
        # print the number of parameters in the model
        logger.info(f'{sum(p.numel() for p in model.parameters()) / 1e6} M parameters')

    elif name == "LinearNet":
        n_input = 256
        #n_input = 128
        model = LinearNet(n_input, cfg["DROPOUT_RATE"], cfg["N_OUT"], activation=cfg["IS_ACTIVATION"])
        logger.info(f'Building model: \nLinearNet'
                    f'Num neurons: {cfg["N_OUT"]}, Dropout rate: {cfg["DROPOUT_RATE"]}\n\n')
        logger.info(f'{model}')
        # print the number of parameters in the model
        logger.info(f'{sum(p.numel() for p in model.parameters()) / 1e6} M parameters')

    else:
        raise NameError('Please specify current argument for name.')

    if cfg["IS_DATA_PARALLEL"]:
        from torch.nn import DataParallel
        model = DataParallel(model, device_ids=cfg["GPU_IDS"])

    return model
