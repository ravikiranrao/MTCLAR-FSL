import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def conf_mat(y_true, y_pred, labels, display_labels, savefig_path=None):
    """
    Plot confusion matrix for similar/dissimilar predictions.
    """
    plt.close("all")
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot()
    if savefig_path is not None:
        plt.savefig(savefig_path, bbox_inches='tight')
    plt.show()
    plt.close()


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
