import os
import datetime
from pathlib import Path
import pickle

import torch
from torch.utils.data import DataLoader

from src.utils import load_config, SaveBestModel, save_model, dump_config, EarlyStopping,\
    conf_mat, get_logger, VisdomLinePlotter
from src.data import get_dataset
from src.model import get_model
from src.tools import train, test, get_scheduler

torch.manual_seed(5)

# Load config file
cfg = load_config(str(Path(__file__).parent.joinpath('mtclar_config.yaml')))

save_dir = os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + cfg['EXPERIMENT_NAME'])

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Create logger and save everything that's displayed
logger = get_logger(os.path.join(save_dir, 'everything.log'))

global plotter
plotter = VisdomLinePlotter(env_name=os.path.basename(os.path.normpath(save_dir)), save_dir=save_dir)


# Create Train dataset
train_dataset = get_dataset(cfg["DATASET_NAME"], "train", cfg)

train_loader = DataLoader(train_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=False, num_workers=cfg['NUM_WORKERS'],
                          drop_last=True)
# shuffle False still shuffles the data. Check dataset for more info.

# Create Validation dataset
validation_dataset = get_dataset(cfg["DATASET_NAME"], "validation", cfg)

validation_loader = DataLoader(validation_dataset, batch_size=cfg['BATCH_SIZE'], num_workers=1,
                               shuffle=False, drop_last=True)

model = get_model(cfg["MODEL_NAME"], cfg, logger).to(cfg['DEVICE'])


# Load best weights
checkpoint = torch.load(os.path.join(cfg["SAVED_MODEL_DIR"], 'final_model.pth'),
                        map_location=cfg["DEVICE"])
# For emonet weights
#checkpoint['model_state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
logger.info(f'Loading best model from epoch {checkpoint["epoch"]}')
model.load_state_dict(checkpoint['model_state_dict'])
model.train()


parameters_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        parameters_to_update.append(param)

optimizer = torch.optim.Adam(params=model.parameters(), lr=float(cfg['LEARNING_RATE']),
                             weight_decay=float(cfg['WEIGHT_DECAY']))

# optimizer = torch.optim.SGD(params=parameters_to_update, lr=float(cfg['LEARNING_RATE']), momentum=0.9)

scheduler = get_scheduler(optimizer, cfg["SCHEDULER"], cfg, steps_per_epoch=len(train_dataset)//cfg['BATCH_SIZE'],
                          epochs=cfg['NUM_EPOCHS'])


# Main training loop
# initialise SaveBestModel class
save_best_model = SaveBestModel()

if cfg["IS_EARLY_STOPPING"]:
    # Initialize Early stopping
    early_stopping = EarlyStopping(tolerance=cfg["TOLERANCE_EARLY_STOP"], min_delta=cfg["DELTA_EARLY_STOP"])

logger.info('Training started..')

for epoch in range(1, cfg['NUM_EPOCHS'] + 1):
    train_dict = train(model, cfg['DEVICE'], train_loader, optimizer, epoch, logger,
                       cfg['LOG_INTERVAL'], plotter, cfg, scheduler)

    val_dict = test(model, "validation", cfg['DEVICE'], validation_loader, logger,
                    cfg['BATCH_SIZE'], plotter, epoch=epoch, cfg=cfg)

    save_best_model(val_dict['Loss'], epoch, model, optimizer, None,
                    os.path.join(save_dir), logger)
    if cfg["IS_EARLY_STOPPING"]:
        # early stopping
        early_stopping(train_dict['Loss'], val_dict['Loss'])
        logger.info(f"EARLY STOPPING TOLERANCE: {early_stopping.counter}")
        if early_stopping.early_stop:
            logger.info(f"Training stopped at epoch: {epoch}")
            break

    if cfg['SCHEDULER'] != 'OneCycleLR':
        # All other scheduler except OneCycleLR needs to be updated every epoch

        if cfg["SCHEDULER"] == "ReduceLROnPlateau":
            scheduler.step(val_dict["Loss"])
        else:
            scheduler.step()

save_model(epoch, model, optimizer, None, os.path.join(save_dir), logger)

del model

# Testing
# Create Test dataset
test_dataset = get_dataset(cfg["DATASET_NAME"], "test", cfg)

test_loader = DataLoader(test_dataset, batch_size=cfg['BATCH_SIZE'], num_workers=1, shuffle=False,
                         drop_last=True)

test_model = get_model(cfg["MODEL_NAME"], cfg, logger).to(cfg['DEVICE'])

checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
logger.info(f'Loading best model from epoch {checkpoint["epoch"]}')
test_model.load_state_dict(checkpoint['model_state_dict'])

test_model.eval()
test_dict = test(test_model, "test", cfg['DEVICE'], test_loader,
                 logger, cfg['BATCH_SIZE'], cfg=cfg)

# Save Confusion matrix
conf_mat(test_dict['target_true'], test_dict['target_pred'], labels=[1, 0], display_labels=["Similar", "Dissimilar"],
         savefig_path=os.path.join(save_dir, f'conf_mat.png'))

pickle.dump(test_dict, open(os.path.join(save_dir, 'objects.p'), 'wb'))
logger.info(f'Test dict dumped at {os.path.join(save_dir, "objects.p")}')

dump_config(cfg, save_dir)

# save visdom environment as json file to save_dir
plotter.save_json()

logger.info(f'All files are stored in the directory {save_dir}. DONE!')
