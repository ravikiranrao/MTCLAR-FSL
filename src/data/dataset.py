from .affectnet import AffectnetPairExpression
from .afewva import AfewvaPairExpression


def get_dataset(name, mode, cfg):
    assert mode in ["train", "validation", "test"]

    if mode == "train":
        dataset_len = cfg["TRAIN_LEN"]
        path_to_csv = cfg['PATH_TO_CSV_TRAIN']
    elif mode == "validation":
        dataset_len = cfg["VAL_LEN"]
        path_to_csv = cfg['PATH_TO_CSV_VALIDATION']
    else:
        dataset_len = cfg["TEST_LEN"]
        path_to_csv = cfg['PATH_TO_CSV_TEST']

    if name == "AffectnetPairExpression":
        dataset = AffectnetPairExpression(cfg['DATA_ROOT'], path_to_csv, n_expression=cfg['N_EXPRESSION'],
                                          image_shape=(3, cfg['IMAGE_SHAPE'], cfg['IMAGE_SHAPE']),
                                          augment=(mode == "train" and cfg['AUGMENT']), dataset_len=dataset_len)

    elif name == "AfewvaPairExpression":
        dataset = AfewvaPairExpression(cfg['DATA_ROOT'], path_to_csv, n_expression=cfg['N_EXPRESSION'],
                                       image_shape=(3, cfg['IMAGE_SHAPE'], cfg['IMAGE_SHAPE']),
                                       augment=(mode == "train" and cfg['AUGMENT']), dataset_len=dataset_len)
    else:
        raise ValueError('Input the right name for the dataset.')

    return dataset
