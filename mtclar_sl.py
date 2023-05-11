"""
Supervised learning from the features learnt
"""
# imports
import os
import datetime
from pathlib import Path
import pickle
import numpy as np


import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.utils import load_config, SaveBestModel, save_model, dump_config, EarlyStopping,\
    get_logger, VisdomLinePlotter, conf_mat
from src.data import AffectnetFeature, AfewvaFeature
from src.tools import get_features, get_scheduler, train3, test3, train3a, test3a
from src.model import get_model

torch.manual_seed(5)


# Load config file
cfg_main = load_config(str(Path(__file__).parent.joinpath('mtclar_sl_config.yaml')))

# config file from the saved directory
cfg = load_config(os.path.join(cfg_main["SAVED_MODEL_DIR"], "config.yaml"))

save_dir = os.path.join(cfg_main["SAVED_MODEL_DIR"],
                        f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_' + cfg_main['EXPERIMENT_NAME'])

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Create logger and save everything that's displayed
logger = get_logger(os.path.join(save_dir, 'everything_SL.log'))

global plotter
plotter = VisdomLinePlotter(env_name=
                            f'{os.path.basename(os.path.normpath(cfg_main["SAVED_MODEL_DIR"]))}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_Features',
                            save_dir=save_dir)

# get features
if cfg_main["IS_GENERATE_FEATURES"]:
    logger.info('Generating features')

    # Get model
    cfg["IS_DATA_PARALLEL"] = False
    cfg["IS_PRETRAINED"] = True
    feat_model = get_model(cfg["MODEL_NAME"], cfg, logger).to(cfg_main["DEVICE"])

    # Load best weights
    checkpoint = torch.load(os.path.join(cfg_main["SAVED_MODEL_DIR"], 'best_model.pth'),
                            map_location=cfg_main["DEVICE"])
    # For emonet weights
    checkpoint['model_state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    logger.info(f'Loading best model from epoch {checkpoint["epoch"]}')
    feat_model.load_state_dict(checkpoint['model_state_dict'])
    feat_model.eval()

    train_feat_dict = get_features(feat_model, cfg_main["DATA"], cfg_main["DATA_ROOT"],cfg_main["PATH_TO_CSV_TRAIN"],
                                   cfg, cfg_main["IS_AUGMENT"], cfg_main["GET_MORE_DATA_NUM"], cfg_main["DEVICE"], logger,
                                   save_path=os.path.join(cfg_main["SAVED_MODEL_DIR"], cfg_main["TRAIN_PICKLE_PATH"]))

    val_feat_dict = get_features(feat_model, cfg_main["DATA"], cfg_main["DATA_ROOT"], cfg_main["PATH_TO_CSV_VALIDATION"],
                                 cfg, False, 1, cfg_main["DEVICE"], logger,
                                 save_path=os.path.join(cfg_main["SAVED_MODEL_DIR"], cfg_main["VAL_PICKLE_PATH"]))

    test_feat_dict = get_features(feat_model, cfg_main["DATA"], cfg_main["DATA_ROOT"], cfg_main["PATH_TO_CSV_TEST"],
                                  cfg, False, 1, cfg_main["DEVICE"], logger,
                                  save_path=os.path.join(cfg_main["SAVED_MODEL_DIR"], cfg_main["TEST_PICKLE_PATH"]))
else:
    logger.info('Loading Features')
    # load features from already extracted
    with open(os.path.join(cfg_main["SAVED_MODEL_DIR"], cfg_main["TRAIN_PICKLE_PATH"]), 'rb') as f:
        train_feat_dict = pickle.load(f)
    with open(os.path.join(cfg_main["SAVED_MODEL_DIR"], cfg_main["VAL_PICKLE_PATH"]), 'rb') as f:
        val_feat_dict = pickle.load(f)
    with open(os.path.join(cfg_main["SAVED_MODEL_DIR"], cfg_main["TEST_PICKLE_PATH"]), 'rb') as f:
        test_feat_dict = pickle.load(f)


# upsampler
if cfg_main["IS_SAMPLER"]:
    # class weighting

    labels_unique, counts = np.unique(train_feat_dict['categories'], return_counts=True)
    logger.info(f"Unique lables: {labels_unique}")
    logger.info(f"Counts: {counts}")

    # class weights
    class_weights = [sum(counts) / c for c in counts]
    logger.info(f"Class weights: {class_weights}")

    # Assign weights to each input sample from training
    example_weights = [class_weights[int(e)] for e in train_feat_dict['categories']]

    sampler = WeightedRandomSampler(example_weights, len(train_feat_dict['categories']))

else:
    sampler = None

if cfg_main["DATA"] == "affectnet":
    train_dataset = AffectnetFeature(train_feat_dict['features'], train_feat_dict['categories'],
                                  train_feat_dict['valence'], train_feat_dict['arousal'])

    validation_dataset = AffectnetFeature(val_feat_dict['features'], val_feat_dict['categories'], val_feat_dict['valence'],
                                       val_feat_dict['arousal'])
elif cfg_main["DATA"] == "afewva":
    train_dataset = AfewvaFeature(train_feat_dict['features'], train_feat_dict['categories'],
                                  train_feat_dict['valence'], train_feat_dict['arousal'])

    validation_dataset = AfewvaFeature(val_feat_dict['features'], val_feat_dict['categories'], val_feat_dict['valence'],
                                       val_feat_dict['arousal'])
else:
    raise ValueError

# dataloader

train_loader = DataLoader(train_dataset, batch_size=cfg_main["BATCH_SIZE"], sampler=sampler,
                          num_workers=cfg_main['NUM_WORKERS'], drop_last=True)

validation_loader = DataLoader(validation_dataset, batch_size=cfg_main["BATCH_SIZE"],
                               num_workers=cfg_main['NUM_WORKERS'], drop_last=True)

# MODEL
model = get_model(cfg_main["MODEL_NAME"], cfg_main, logger).to(cfg_main["DEVICE"])

optimizer = torch.optim.Adam(params=model.parameters(), lr=float(cfg_main['LEARNING_RATE']),
                             weight_decay=float(cfg_main['WEIGHT_DECAY']))

scheduler = get_scheduler(optimizer, cfg_main["SCHEDULER"], cfg_main,
                          steps_per_epoch=len(train_dataset)//cfg_main['BATCH_SIZE'],
                          epochs=cfg_main['NUM_EPOCHS'])


# Main training loop
# initialise SaveBestModel class
save_best_model = SaveBestModel()

if cfg_main["IS_EARLY_STOPPING"]:
    # Initialize Early stopping
    early_stopping = EarlyStopping(tolerance=cfg_main["TOLERANCE_EARLY_STOP"], min_delta=cfg_main["DELTA_EARLY_STOP"])

logger.info('Training started..')

for epoch in range(1, cfg_main['NUM_EPOCHS'] + 1):
    train_dict = train3a(model, cfg_main["DEVICE"], train_loader, optimizer, epoch, logger, cfg_main["LOG_INTERVAL"],
                        plotter, cfg_main, cfg_main["IS_CLAMP"])
    val_dict = test3a(model, "validation", cfg_main["DEVICE"], validation_loader, logger, cfg_main,
                     cfg_main["IS_CLAMP"], epoch=epoch, plotter=plotter)

    save_best_model(val_dict['Loss'], epoch, model, optimizer, None, os.path.join(save_dir), logger)

    if cfg_main["IS_EARLY_STOPPING"]:
        # early stopping
        early_stopping(train_dict['Loss'], val_dict['Loss'])
        logger.info(f"EARLY STOPPING TOLERANCE: {early_stopping.counter}")
        if early_stopping.early_stop:
            logger.info(f"Training stopped at epoch: {epoch}")
            break

    if cfg_main['SCHEDULER'] != 'OneCycleLR':
        # All other scheduler except OneCycleLR needs to be updated every epoch

        if cfg_main["SCHEDULER"] == "ReduceLROnPlateau":
            scheduler.step(val_dict["Loss"])
        else:
            scheduler.step()

save_model(epoch, model, optimizer, None, os.path.join(save_dir), logger)

del model

# Testing
# Create Test dataset
if cfg_main["DATA"] == "affectnet":
    test_dataset = AffectnetFeature(test_feat_dict['features'], test_feat_dict['categories'], test_feat_dict['valence'],
                                 test_feat_dict['arousal'])
elif cfg_main["DATA"] == "afewva":
    test_dataset = AfewvaFeature(test_feat_dict['features'], test_feat_dict['categories'], test_feat_dict['valence'],
                                 test_feat_dict['arousal'])
else:
    raise ValueError

test_loader = DataLoader(test_dataset, batch_size=cfg_main["BATCH_SIZE"], num_workers=1, shuffle=False, drop_last=True)

# MODEL
test_model = get_model(cfg_main["MODEL_NAME"], cfg_main, logger).to(cfg_main['DEVICE'])

checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
logger.info(f'Loading best model from epoch {checkpoint["epoch"]}')
test_model.load_state_dict(checkpoint['model_state_dict'])

test_model.eval()
test_dict = test3a(test_model, "test", cfg_main["DEVICE"], test_loader, logger, cfg_main, cfg_main["IS_CLAMP"],
                  epoch=-1, plotter=None)

#plot_results(test_dict['valence_true'], test_dict['valence_pred'],
#             savefig_path=os.path.join(save_dir, f'valence.png'), title="Valence")
#plot_results(test_dict['arousal_true'], test_dict['arousal_pred'],
#             savefig_path=os.path.join(save_dir, f'arousal.png'), title="Arousal")
# Save Confusion matrix
conf_mat(test_dict['target_true'], test_dict['target_pred'], labels=[0, 1, 2, 3, 4, 5, 6, 7],
         display_labels=["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"],
         savefig_path=os.path.join(save_dir, f'conf_mat.png'))

pickle.dump(test_dict, open(os.path.join(save_dir, 'objects.p'), 'wb'))
logger.info(f'Test dict dumped at {os.path.join(save_dir, "objects.p")}')

dump_config(cfg_main, save_dir)

# save visdom environment as json file to save_dir
plotter.save_json()

logger.info(f'All files are stored in the directory {save_dir}. DONE!')
