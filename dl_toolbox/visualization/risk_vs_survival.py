from lifelines.utils import concordance_index
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib.pyplot as plt
import os

import numpy as np


def surv_time_vs_risk(model_predictions, labels, ax, title,
                      time_in_month=True):

    surv_time = labels[:, 0]
    event = labels[:, 1]

    event_idx = np.where(event == 1)[0]
    censor_idx = np.where(event == 0)[0]

    ax.scatter(surv_time[event_idx], model_predictions[event_idx], color="r")
    ax.scatter(surv_time[censor_idx], model_predictions[censor_idx], color="b")

    # also plot a smoothed curve using LOWESS regression
    # for a) all data
    #     b) event data
    #     c) censor data
    filtered = lowess(model_predictions, surv_time)
    ax.plot(filtered[:, 0], filtered[:, 1], color='black', label = "Combined")

    filtered = lowess(model_predictions[event_idx], surv_time[event_idx])
    ax.plot(filtered[:, 0], filtered[:, 1], color='r', label = "Events")

    filtered = lowess(model_predictions[censor_idx], surv_time[censor_idx])
    ax.plot(filtered[:, 0], filtered[:, 1], color='b', label = "Censor")

    try:
        cindex = concordance_index(surv_time, model_predictions, event)
    except Exception as e:
        cindex = np.nan

    ax.legend()
    ax.set_title(title + f": C-Index = {cindex:.2f}", fontsize=16)

    if time_in_month:
        ax.set_xlabel("Time / months", fontsize=14)
        ax.set_xlim((0, 60))
        ax.set_xticks([0, 12, 24, 36, 48, 60])
    else:
        ax.set_xlabel("Time", fontsize=14)


def plot_surv_times_vs_risk(pred_train, train_labels, pred_valid, valid_labels,
                            pred_test=None, test_labels=None,
                            time_in_month=True, subtitle=None, save_dir=None):

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
    fig, axs = plt.subplots(1, n_plots, figsize=(4*n_plots, 5), sharey=True, dpi=300)
    axs[0].set_ylabel("Prediction", fontsize=14)
    # axs[0].set_ylim((-1.2, 1.2))

    for i, (title, data) in enumerate(zip(titles, data)):
        pred, labels = data
        surv_time_vs_risk(
            pred, labels, ax=axs[i], title=title,
            time_in_month=time_in_month)

    plt.tight_layout()
    if save_dir is not None:
        #fn = os.path.join(save_dir, "surv_time_vs_risk.png")
        fn = save_dir
        plt.savefig(fn)
        print("saved plot to", fn)

    #plt.show()
    return fig
