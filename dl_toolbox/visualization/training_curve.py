
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_train_loss(hist, key='loss', save_dir=None):
    train_losses = hist.history[key]
    test_losses = hist.history['val_' + key]

    fig, ax = plt.subplots(1, 1, dpi=300)
    ax.plot(train_losses)
    ax.plot(test_losses)
    ax.set_title(key + " during training")
    ax.set_xlabel("epoch")
    ax.set_ylabel(key)
    ax.legend(['training', 'validation'], loc='upper left')

    if save_dir is not None:
        fn = os.path.join(save_dir, "training_history.png")
        plt.savefig(fn)
        print("saved plot to", fn)

    plt.show()


def plot_histories(histories, keys=["loss"], save_dir=None):
    """
    Parameters
    ----------
    histories: list of tuples of (model_name, model_history)
    """

    # need to check which keys are available in all histories
    available_keys = []
    for k in keys:
        is_present_in_all = np.all([k in history.history for (_, history) in histories ])
        if is_present_in_all:
            available_keys.append(k)

    if len(available_keys) < len(keys):
        print("Ignore keys {} since not available in all histories!".format(
            list(set(keys) - set(available_keys))))

    fig, axs = plt.subplots(1, len(available_keys), figsize=(5*len(available_keys), 5), dpi=300)
    if len(available_keys) == 1:
        axs = [axs]

    # one plot for each specified key (a metric or loss)
    for i, ax in enumerate(axs):
        key = available_keys[i]
        for name, history in histories:
            # validation
            val = ax.plot(history.history['val_'+key],
                          '--', label=name.title()+' validation')
            # training
            ax.plot(
                history.history[key], color=val[0].get_color(),
                label=name.title()+' training')

        ax.set_xlabel('Epoch')
        if key == "ci":
            # only for nicer plots
            key = "Concordance index"
            ax.set_ylim((0, 1))

        ax.set_ylabel(key.replace('_', ' ').title())

        ax.legend()
        ax.set_title(key.title())
        ax.set_xlim([0, max(history.epoch)])

    plt.tight_layout()

    if save_dir is not None:
        fn = os.path.join(save_dir, "training_curve.png")
        plt.savefig(fn)
        print("saved plot to", fn)