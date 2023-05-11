import matplotlib.pyplot as plt
import os
import logging
import numpy as np
import yaml
import seaborn as sns
from visdom import Visdom
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch


class AverageMeter(object):
    """Computes ans stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping():
    """
    Adopted from https://stackoverflow.com/a/71999355
    """
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:
                self.early_stop = True


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's validation loss is less than the previous
    least loss, then save the model state.
    """
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion, save_model_dir, logger):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            logger.info(f'Best validation loss: {round(self.best_valid_loss, 4)}')
            logger.info(f'Saving best model for epoch: {epoch}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion},
                os.path.join(save_model_dir, 'best_model.pth'))


def save_model(epoch, model, optimizer, criterion, save_model_dir, logger):
    """
    Function to save the trained model to disk
    """
    logger.info(f'Saving the final model...')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion},
        os.path.join(save_model_dir, 'final_model.pth'))


def calc_metrics_all_folds(list_of_dicts, affect, verbose=1, logger=None):
    """
    Takes input as list of dicts from each fold (train, val or test dicts)
    :param list_of_dicts:
    :return:
    """
    metrics = {}
    # metrics.keys = ['RMSE, PCC', 'CCC', 'SAGR']
    metrics['RMSE'], metrics['PCC'], metrics['CCC'], metrics['SAGR'] = [], [], [], []
    for dict in list_of_dicts:
        for keys in metrics.keys():
            metrics[keys].append(dict[affect + '_metrics'][keys])

    if verbose:
        logger.info('RMSE: {:.3f} +- {:.3f}\n PCC: {:.3f} +- {:.3f}\n CCC: {:.3f} +- {:.3f}\n SAGR: {:.3f} +- {:.3f}'.format(
            np.array(metrics['RMSE']).mean(), np.array(metrics['RMSE']).std(),
            np.array(metrics['PCC']).mean(), np.array(metrics['PCC']).std(),
            np.array(metrics['CCC']).mean(), np.array(metrics['CCC']).std(),
            np.array(metrics['SAGR']).mean(), np.array(metrics['SAGR']).std()
        ))

    return metrics


def load_config(config_file_path):
    """
    Function to load configuration file
    """
    with open(config_file_path) as file:
        config = yaml.safe_load(file)
    return config


def dump_config(config, path_to_save):
    """
    Function to dump configuration file
    """
    with open(os.path.join(path_to_save, 'config.yaml'), 'w', encoding='utf8') as outfile:
        yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)


def eda_afewva(train_df, val_df, test_df):
    print("**** Train _df ****")
    print(train_df.head(10))
    print("**** Val _df ****")
    print(val_df.head(10))
    print("**** Test _df ****")
    print(test_df.head(10))

    print("**** Train df describe ****")
    print(train_df.describe())
    print("**** Val df describe ****")
    print(val_df.describe())
    print("**** Test df describe ****")
    print(test_df.describe())

    sns.set_theme(style="darkgrid")
    sns.lineplot(x="", y="valence", hue="region", style="event")


def get_logger(save_path):

    print('Creating Log')
    # create a logger object instance
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.NOTSET)
    console_handler_format = '%(asctime)s | %(levelname)s: %(message)s'
    console_handler.setFormatter(logging.Formatter(console_handler_format))
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(save_path)
    file_handler.setLevel(logging.INFO)
    file_handler_format = '%(asctime)s | %(levelname)s | %(lineno)d: %(message)s'
    file_handler.setFormatter(logging.Formatter(file_handler_format))
    logger.addHandler(file_handler)
    return logger


# For visualising plots
# Works like tensorboard
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name, save_dir=None):
        self.viz = Visdom()
        self.env = env_name
        self.save_dir = save_dir
        self.plots = {}


    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append')

    def save_json(self):
        #
        self.viz.save([self.env])

        if self.save_dir is not None:
            import shutil
            p = shutil.copy2(os.path.join(os.path.expanduser("~/.visdom"), self.env + ".json"), self.save_dir)

    def conf_mat(self, y_true, y_pred, labels, display_labels, epoch, title):
        assert self.save_dir is not None

        if not os.path.isdir(os.path.join(self.save_dir, "conf_mats")):
            os.makedirs(os.path.join(self.save_dir, "conf_mats"), exist_ok=True)

        plt.close("all")
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot()
        disp.ax_.get_images()[0].set_clim(0, 1)

        save_img_path = os.path.join(self.save_dir, "conf_mats", f"conf_mat_epoch{epoch}.jpeg")
        plt.savefig(save_img_path, bbox_inches='tight')
        img = plt.imread(save_img_path)
        img = np.moveaxis(img, -1, 0)
        self.viz.image(img,
                       win='Confusion Matrix',
                       opts=dict(caption=f'Epoch {epoch}', store_history=True,
                                 title=title),
                       env=self.env
                       )

    def image_pairs_classification_viz(self, y_true, y_pred, categories_pair, display_labels, epoch, title):
        assert self.save_dir is not None

        if not os.path.isdir(os.path.join(self.save_dir, "emotion_pair_plots")):
            os.makedirs(os.path.join(self.save_dir, "emotion_pair_plots"), exist_ok=True)
        if not os.path.isdir(os.path.join(self.save_dir, "emotion_pair_plots_symm")):
            os.makedirs(os.path.join(self.save_dir, "emotion_pair_plots_symm"), exist_ok=True)

        plt.close("all")
        # 1 -> if predicted correctly, 0 otherwise
        # Apply XNOR gate between true and pred
        match = np.logical_not(np.logical_xor(y_true, y_pred)).astype(int)

        # create matrix with the total number of data points for each emotion class
        K = len(np.unique(categories_pair))  # Number of classes

        result_n = np.zeros((K, K))

        for i in range(categories_pair.shape[0]):
            result_n[int(categories_pair[i, :][0])][int(categories_pair[i, :][1])] += 1

        # if you assume that Pair(A, B) is same as pair(B, A), then
        result_n_symm = np.tril(result_n + np.transpose(np.triu(result_n, k=1)))

        # -----  Prediction ---------#
        result_pred = np.zeros((K, K))

        for i in range(y_pred.shape[0]):
            # Fill in 1 or 0 (from match) to the prediction matrix
            result_pred[int(categories_pair[i, :][0])][int(categories_pair[i, :][1])] += match[i]

        # considering Pair(A, B) = pair(B, A)
        result_pred_symm = np.tril(result_pred + np.transpose(np.triu(result_pred, k=1)))

        # ------- Accuracy ----------#
        np.seterr(divide='ignore', invalid='ignore')
        results_acc = np.round((result_pred / result_n) * 100, 3)
        results_acc_symm = np.round(np.nan_to_num(result_pred_symm / result_n_symm) * 100, 3)

        # Have the same colour bap range for both the plots
        # min_value = np.min(results_acc)
        # max_value = np.max(results_acc)
        # not a good idea for comparison. Have same min and max for all the window
        min_value = 0
        max_value = 100

        # -------- Plotting ---------#
        ax = plt.subplot()
        disp = ConfusionMatrixDisplay(confusion_matrix=results_acc, display_labels=display_labels)
        disp.plot(xticks_rotation='vertical', ax=ax)
        ax.set_xlabel('Emotion Category of Image B')
        ax.set_ylabel('Emotion Category of Image A')
        disp.ax_.get_images()[0].set_clim(min_value, max_value)

        save_img_path = os.path.join(self.save_dir, "emotion_pair_plots", f"emotion_pair_plot_epoch{epoch}.jpeg")
        plt.savefig(save_img_path, bbox_inches='tight')
        img = plt.imread(save_img_path)
        img = np.moveaxis(img, -1, 0)
        self.viz.image(img,
                       win='Similarity/Dissimilarity Accuracy for each pair',
                       opts=dict(caption=f'Epoch {epoch}', store_history=True,
                                 title=title),
                       env=self.env
                       )

        ax_symm = plt.subplot()
        disp_symm = ConfusionMatrixDisplay(confusion_matrix=results_acc_symm, display_labels=display_labels)
        disp_symm.plot(xticks_rotation='vertical', ax=ax_symm)
        ax_symm.set_xlabel('Emotion Category of Image B')
        ax_symm.set_ylabel('Emotion Category of Image A')
        disp_symm.ax_.get_images()[0].set_clim(min_value, max_value)

        save_img_path = os.path.join(self.save_dir,
                                     "emotion_pair_plots_symm", f"emotion_pair_plot_symm_epoch{epoch}.jpeg")
        plt.savefig(save_img_path, bbox_inches='tight')
        img = plt.imread(save_img_path)
        img = np.moveaxis(img, -1, 0)
        self.viz.image(img,
                       win='Similarity/Dissimilarity Accuracy for each pair (symmetry assumption)',
                       opts=dict(caption=f'Epoch {epoch}', store_history=True,
                                 title=title),
                       env=self.env
                       )

        results_acc_sim = (100 - results_acc) + 100 * np.eye(K)
        ax_sim = plt.subplot()
        disp_sim = ConfusionMatrixDisplay(confusion_matrix=results_acc_sim, display_labels=display_labels)
        disp_symm.plot(xticks_rotation='vertical', ax=ax_sim)
        ax_symm.set_xlabel('Emotion Category of Image B')
        ax_symm.set_ylabel('Emotion Category of Image A')
        disp_symm.ax_.get_images()[0].set_clim(min_value, max_value)

        save_img_path = os.path.join(self.save_dir,
                                     "emotion_pair_plots_symm", f"emotion_pair_plot_symm_epoch{epoch}.jpeg")
        plt.savefig(save_img_path, bbox_inches='tight')
        img = plt.imread(save_img_path)
        img = np.moveaxis(img, -1, 0)
        self.viz.image(img,
                       win='Similarity Accuracy for each pair',
                       opts=dict(caption=f'Epoch {epoch}', store_history=True,
                                 title=title),
                       env=self.env
                       )

    def scatter(self, y_true, y_pred, title, affect, epoch):
        if not os.path.isdir(os.path.join(self.save_dir, "scatter_plots")):
            os.makedirs(os.path.join(self.save_dir, "scatter_plots"), exist_ok=True)

        plt.close("all")
        plt.scatter(y_true, y_pred)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.xlabel('True values', fontsize=18)
        plt.ylabel('Predicted values', fontsize=16)
        save_img_path = os.path.join(self.save_dir,
                                     "scatter_plots", f"scatter_{affect}_epoch{epoch}.jpeg")

        plt.savefig(save_img_path, bbox_inches='tight')
        img = plt.imread(save_img_path)
        img = np.moveaxis(img, -1, 0)
        self.viz.image(img,
                       win=f'Scatter Plot ({affect.capitalize()})',
                       opts=dict(caption=f'Epoch {epoch}', store_history=True,
                                 title=title),
                       env=self.env
                       )

    def plot_results(y_true, y_pred, savefig_path=None, title=None):
        plt.close("all")
        plt.scatter(y_true, y_pred)
        plt.xlim([y_true.min(), y_true.max()])
        plt.ylim([y_pred.min(), y_pred.max()])
        plt.xlabel('True values', fontsize=18)
        plt.ylabel('Predicted values', fontsize=16)
        if title is not None:
            plt.title(title)
        if savefig_path is not None:
            plt.savefig(savefig_path, bbox_inches='tight')
        plt.show()
        plt.close()
