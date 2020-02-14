import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(predicted_class, labels, ax, title,
                          normalize=False, cmap=plt.cm.Blues):
    # mostly copied from
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    if normalize:
        title += '\nNormalized entries'

    # Compute confusion matrix
    cm = confusion_matrix(labels, predicted_class)

    # Only use the labels that appear in the data
    classes = np.unique(labels)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="grey" if cm[i, j] > thresh else "black")


def plot_confusion_matrices(pred_train, train_labels, pred_valid, valid_labels,
                            pred_test=None, test_labels=None,
                            subtitle=None, save_dir=None):
    # NOTE: predictions have to be class labels already!
    titles = ["Training", "Validation"]

    data = [
        (pred_train, train_labels), (pred_valid, valid_labels)]
    if pred_test is not None and test_labels is not None:
        titles.append("Test")
        data.append((pred_test, test_labels))

    if subtitle is not None:
        for i in range(len(titles)):
            titles[i] += "\n" + subtitle

    n_plots = len(titles)
    fig, axs = plt.subplots(1, n_plots, sharey=True, figsize=(4*n_plots, 5), dpi=300)

    for i, (title, data) in enumerate(zip(titles, data)):
        pred, labels = data
        plot_confusion_matrix(
            pred, labels,
            ax=axs[i], title=title)

    plt.tight_layout()

    if save_dir is not None:
        #fn = os.path.join(save_dir, "kaplan_meier.png")
        fn = save_dir
        plt.savefig(fn)
        print("saved plot to", fn)

    #plt.show()
    return fig
