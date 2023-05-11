"""
FSL from MT-CLAR
"""
# imports
import os
import datetime
from pathlib import Path
import pickle
from PIL import Image

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.utils import load_config, dump_config, plot_results,get_logger, VisdomLinePlotter
from src.data import AfewvaAnchorPair
from src.tools import test5
from src.model import get_model

torch.manual_seed(5)


# Load config file
cfg_main = load_config(str(Path(__file__).parent.joinpath('fsl_mtclar_config.yaml')))

# config file from the saved directory
cfg = load_config(os.path.join(cfg_main["SAVED_MODEL_DIR"], "config.yaml"))

save_dir = os.path.join(cfg_main["SAVED_MODEL_DIR"],
                        f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_FSL_' + cfg_main['EXPERIMENT_NAME'])

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Create logger and save everything that's displayed
logger = get_logger(os.path.join(save_dir, 'everything_SL.log'))

global plotter
plotter = VisdomLinePlotter(env_name=
                            f'{os.path.basename(os.path.normpath(cfg_main["SAVED_MODEL_DIR"]))}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_Features',
                            save_dir=save_dir)

# Get model
cfg["IS_DATA_PARALLEL"] = False
cfg["IS_PRETRAINED"] = True
feat_model = get_model(cfg["MODEL_NAME"], cfg, logger).to(cfg_main["DEVICE"])

# Load best weights
checkpoint = torch.load(os.path.join(cfg_main["SAVED_MODEL_DIR"], 'final_model.pth'),
                        map_location=cfg_main["DEVICE"])
# For emonet weights
checkpoint['model_state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
logger.info(f'Loading best model from epoch {checkpoint["epoch"]}')
feat_model.load_state_dict(checkpoint['model_state_dict'])
feat_model.eval()

# Testing
# Create Test dataset
test_dataset = AfewvaAnchorPair(cfg_main["DATA_ROOT"], cfg_main["PATH_TO_CSV_TEST"],
                                image_shape=(3, cfg['IMAGE_SHAPE'], cfg['IMAGE_SHAPE']), augment=False, verbose=0)
logger.info(f"Anchor images: {len(test_dataset.actor_df)} / {len(test_dataset.data_df)}")

# - collate function - #
def paired_collate(batch, anchor_df, data_root, transform):
    # assert batch size is 1
    assert len(batch) == 1
    batch_dict = batch[0]

    #a = anchor_df.loc[anchor_df.video_id == batch_dict["video_id"][0]]
    a = anchor_df
    img1 = batch_dict["image1"]

    # Repeat the image tensor for each frame in the category
    image1 = img1.repeat(len(a), 1, 1, 1)

    # Repeat valence and arousal values
    valence_1 = batch_dict["valence"][0].repeat(len(a),)
    arousal_1 = batch_dict["arousal"][0].repeat(len(a),)

    anchor_images_dict = {}
    anchor_images = []
    valence = []
    arousal =[]

    for i in range(len(a)):
        # Anchor image
        image_2_path = os.path.join(a.video_id.values[i], a.frame_num.values[i])

        # Load both the images
        image2 = Image.open(os.path.join(data_root, image_2_path)).convert("RGB")

        image2 = transform(image2)

        anchor_images.append(image2)
        valence.append(torch.tensor(a.valence.values[i]))
        arousal.append(torch.tensor(a.arousal.values[i]))

    image2 = torch.stack(anchor_images, dim=0)
    valence_anchor = torch.stack(valence, dim=0)
    arousal_anchor = torch.stack(arousal, dim=0)

    valence = torch.stack((valence_1, valence_anchor), dim=1)
    arousal = torch.stack((arousal_1, arousal_anchor), dim=1)

    # return a batch with (Image i, Anchor 1), (Image i, Anchor 2)
    return dict(image1=image1, image2=image2, valence=valence, arousal=arousal)

#
test_loader = DataLoader(test_dataset, batch_size=cfg_main["BATCH_SIZE"], num_workers=1, shuffle=False, drop_last=False)
#anchor_df = test_dataset.actor_df
#data_root = test_dataset.data_root
#transform = test_dataset.transform
#test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=False,
#                         collate_fn=lambda x: paired_collate(x, anchor_df, data_root, transform))

test_dict = test5(feat_model, "test", cfg_main["DEVICE"], test_loader, logger, cfg_main["BATCH_SIZE"], cfg=cfg)

plot_results(test_dict['valence_true'], test_dict['valence_pred'],
             savefig_path=os.path.join(save_dir, f'valence.png'), title="Valence")
plot_results(test_dict['arousal_true'], test_dict['arousal_pred'],
             savefig_path=os.path.join(save_dir, f'arousal.png'), title="Arousal")

pickle.dump(test_dict, open(os.path.join(save_dir, 'objects.p'), 'wb'))
logger.info(f'Test dict dumped at {os.path.join(save_dir, "objects.p")}')

dump_config(cfg_main, save_dir)

# save visdom environment as json file to save_dir
#plotter.save_json()

logger.info(f'All files are stored in the directory {save_dir}. DONE!')
