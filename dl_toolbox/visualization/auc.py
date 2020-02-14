import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve


def plot_auc_curve(predictions, labels, ax, title):

    auc = roc_auc_score(labels, predictions)

    fpr, tpr, _ = roc_curve(labels, predictions)

    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")


def plot_auc_curves(pred_train, train_labels, pred_valid, valid_labels,
                    pred_test=None, test_labels=None,
                    subtitle=None, save_dir=None):
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
        plot_auc_curve(
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
