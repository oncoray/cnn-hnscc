from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
# from lifelines.plotting import add_at_risk_counts
from lifelines.plotting import remove_spines, remove_ticks, move_spines,\
    is_latex_enabled

import matplotlib.pyplot as plt
import numpy as np
import os


def add_at_risk_counts(*fitters, **kwargs):
    """
    Copied and slightly adapted from lifelines to work with 3x3 subplots without
    cluttering the risk_counts and the x axis labels!

    Add counts showing how many individuals were at risk at each time point in
    survival/hazard plots.
    Parameters
    ----------
    fitters:
      One or several fitters, for example KaplanMeierFitter,
      NelsonAalenFitter, etc...
    Returns
    --------
      ax:
        The axes which was used.
    Examples
    --------
    >>> # First train some fitters and plot them
    >>> fig = plt.figure()
    >>> ax = plt.subplot(111)
    >>>
    >>> f1 = KaplanMeierFitter()
    >>> f1.fit(data)
    >>> f1.plot(ax=ax)
    >>>
    >>> f2 = KaplanMeierFitter()
    >>> f2.fit(data)
    >>> f2.plot(ax=ax)
    >>>
    >>> # There are equivalent
    >>> add_at_risk_counts(f1, f2)
    >>> add_at_risk_counts(f1, f2, ax=ax, fig=fig)
    >>>
    >>> # This overrides the labels
    >>> add_at_risk_counts(f1, f2, labels=['fitter one', 'fitter two'])
    >>>
    >>> # This hides the labels
    >>> add_at_risk_counts(f1, f2, labels=None)
    """
    add_caption = kwargs.pop("add_caption", True)
    table_below_scaling = kwargs.pop("table_below_scaling", 0.15)

    # Axes and Figure can't be None
    ax = kwargs.pop("ax", None)
    if ax is None:
        ax = plt.gca()

    fig = kwargs.pop("fig", None)
    if fig is None:
        fig = plt.gcf()

    if "labels" not in kwargs:
        labels = [f._label for f in fitters]
    else:
        # Allow None, in which case no labels should be used
        labels = kwargs.pop("labels", None)
        if labels is None:
            labels = [None] * len(fitters)
    # Create another axes where we can put size ticks
    ax2 = plt.twiny(ax=ax)
    # Move the ticks below existing axes
    # Appropriate length scaled for 6 inches. Adjust for figure size.
    ax2_ypos = -1. * table_below_scaling * 2 * 6.0 / fig.get_figheight()
    move_spines(ax2, ["bottom"], [ax2_ypos])
    # Hide all fluff
    remove_spines(ax2, ["top", "right", "bottom", "left"])
    # Set ticks and labels on bottom
    ax2.xaxis.tick_bottom()
    # Set limit
    min_time, max_time = ax.get_xlim()
    ax2.set_xlim(min_time, max_time)
    # Set ticks to kwarg or visible ticks
    xticks = kwargs.pop("xticks", None)
    if xticks is None:
        xticks = [xtick for xtick in ax.get_xticks() if min_time <= xtick <= max_time]
    ax2.set_xticks(xticks)
    # Remove ticks, need to do this AFTER moving the ticks
    remove_ticks(ax2, x=True, y=True)
    # Add population size at times
    ticklabels = []
    for tick in ax2.get_xticks():
        lbl = ""
        # Get counts at tick
        counts = [f.durations[f.durations >= tick].shape[0] for f in fitters]
        # Create tick label
        for l, c in zip(labels, counts):
            # First tick is prepended with the label
            if tick == ax2.get_xticks()[0] and l is not None:
                # Get length of largest count
                max_length = len(str(max(counts)))
                if is_latex_enabled():
                    s = "\n{}\\quad".format(l) + "{{:>{}d}}".format(max_length)
                else:
                    s = "\n{}   ".format(l) + "{{:>{}d}}".format(max_length)
            else:
                s = "\n{}"
            lbl += s.format(c)
        ticklabels.append(lbl.strip())
    # Align labels to the right so numbers can be compared easily
    ax2.set_xticklabels(ticklabels, ha="right", **kwargs)

    # Add a descriptive headline.
    ax2.xaxis.set_label_coords(0, ax2_ypos)

    if add_caption:
        ax2.set_xlabel("At risk")

    plt.tight_layout()
    return ax




def km_curve(times, events, label, ax, show_censors=True, ci_show=False):
    km = KaplanMeierFitter()
    try:
        km.fit(times, event_observed=events, label=label)
        ax = km.plot(ax=ax, show_censors=show_censors, ci_show=ci_show)
        ax.legend(loc="upper right")
    except Exception as e:
        print("WARN: KaplanMeierFitter failed for label={}! {}".format(
            label, e))
        return None

    return km


def two_km_curves_with_pval(times, events, label, ax, show_censors=True,
                            ci_show=False, at_risk_counts=True,
                            annotation=None, print_at_risk_count_label=True,
                            time_in_month=True, table_below_scaling=0.15, size=12):
    """
    Plots Kaplan Meier curves for two different strata and computes p-value
    of logrank test for statistically significant differences between curves.
    """
    assert len(times) == len(events) == len(label) == 2

    if time_in_month:
        # we have to set the coordinate system before calling
        # lifelines to plot into it since otherwise it would
        # choose its own partition of the x axis.
        ax.set_xlim(0, 60)
        ax.set_xticks([0, 12, 24, 36, 48, 60])

    kms = [None] * 2
    for i in range(2):
        # plot group
        try:
            kms[i] = km_curve(times=times[i], events=events[i], label=label[i], ax=ax,
                              show_censors=show_censors, ci_show=ci_show)
        except Exception as e:
            # kms[i] stays None
            pass

    if time_in_month:
        ax.set_xlabel("Time / months", size=size)
    else:
        ax.set_xlabel("Time", size=size)

    # p value
    # logrank testing for distribution differences
    test_res = logrank_test(times[0],
                            times[1],
                            event_observed_A=events[0],
                            event_observed_B=events[1])
    p_val = test_res.p_value
    # print("p_val=", p_val)
    text = "p"
    if p_val < 0.001:
        text += "<0.001"
    elif p_val > 0.1:
        text += f"={p_val:.2f}"
    else:
        # 0.001 <= p <= 0.1
        text += f"={p_val:.3f}"

    if annotation is not None:
        text += ("\n" + annotation)

    ax.text(30, 0.03, text, size=size, multialignment="right")

    kms = [k for k in kms if k is not None]
    if at_risk_counts:
        if print_at_risk_count_label:
            add_at_risk_counts(*kms, ax=ax, table_below_scaling=table_below_scaling)
        else:
            add_at_risk_counts(
                *kms, ax=ax, labels=[None, None], add_caption=False,
                table_below_scaling=table_below_scaling)

    return p_val


def plot_two_km_curves_per_cohort(axs, groups_train, groups_valid=None,
                                  groups_test=None, subtitle=None,
                                  save_dir=None, y_label="Loco-regional tumor control",
                                  strata_labels=["Low risk", "High risk"],
                                  at_risk_counts=True,
                                  annotation=None,
                                  time_in_month=True,
                                  titles=None,
                                  set_titles=True,
                                  table_below_scaling=0.15):
    """
    Creates up to 3 plots of KM curves
    Parameters
    ----------
    groups_train: list of length two
        groups_train[0] -> list of length two with the event times for both strata
        groups_train[1] -> list of length two with the event status for both strata
    groups_valid: see groups_train, if None this will be ignored
    groups_test: see groups_train, if None this will be ignored

    """

    if titles is None:
        titles_used = ["Training"]
    else:
        titles_used = titles

    n_plots = 1

    groups = [g for g in [groups_train, groups_valid, groups_test] if g is not None]
    n_plots = len(groups)
    assert len(axs) == n_plots

    if titles is None:
        titles = ["Training", "Validation", "Test"]
    else:
        assert len(titles) >= n_plots

    titles_used = [titles[i] for i in range(n_plots)]

    if subtitle is not None:
        for i in range(len(titles_used)):
            titles_used[i] += "\n" + subtitle

    # width = 4 * n_plots
    # height = 5
    # fig, axs = plt.subplots(1, n_plots, sharey=True, figsize=(width, height), dpi=300)
    fontsize = 12
    for ax in axs:
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(y_label, size=fontsize)

    p_vals = []
    for i, (title, group) in enumerate(zip(titles_used, groups)):
        p = two_km_curves_with_pval(
            times=group[0],  # times for both strata
            events=group[1],
            label=strata_labels,
            ax=axs[i],
            at_risk_counts=at_risk_counts,
            annotation=annotation,
            print_at_risk_count_label=(i==0),  # risk count labels only for the first cohort
            time_in_month=time_in_month,
            table_below_scaling=table_below_scaling,
            size=fontsize)
        p_vals.append(p)

        if set_titles:
            axs[i].set_title(title, size=fontsize)
        # label only for the first cohort
        if i > 0:
            axs[i].get_legend().remove()

    # plt.tight_layout()

    if save_dir is not None:
        #fn = os.path.join(save_dir, "kaplan_meier.png")
        plt.savefig(save_dir)
        print("saved plot to", save_dir)

    return axs, p_vals


def plot_stratified_cohort_km(axs, condition, pred_train, train_labels,
                              pred_valid, valid_labels,
                              pred_test=None, test_labels=None,
                              subtitle=None, save_dir=None, at_risk_counts=True,
                              strata_labels=["condition", "~condition"],
                              y_label="Probability",
                              annotation=None,
                              time_in_month=True,
                              titles=None,
                              set_titles=True,
                              table_below_scaling=0.15):

    """
    Plot for each cohort a Kaplan Meier plot with two curves. The samples that participate
    in each of the two curves are determined by 'condition'. The first group is the one
    for which condition evaluates to True (elementwise)

    Parameters
    ----------
    condition: callable that returns either True or False for each element
                  of an array.
                  Used to build strata on every cohort
    """

    # stratify each cohort into two groups
    idx_A_train = condition(pred_train)
    idx_B_train = ~idx_A_train

    # groups_train[0] -> the times for each group at idx 0 and 1
    # groups_train[1] -> the event for each group at idx 0 and 1
    groups_train = [
        [train_labels[idx_A_train, 0], train_labels[idx_B_train, 0]],  # the times
        [train_labels[idx_A_train, 1], train_labels[idx_B_train, 1]]   # the events
    ]

    idx_A_valid = condition(pred_valid)
    idx_B_valid = ~idx_A_valid
    groups_valid = [
        [valid_labels[idx_A_valid, 0], valid_labels[idx_B_valid, 0]],
        [valid_labels[idx_A_valid, 1], valid_labels[idx_B_valid, 1]],
    ]

    # then call a method that just plots both groups for each of the cohorts
    # that can be reused

    if pred_test is not None and test_labels is not None:
        idx_A_test = condition(pred_test)
        idx_B_test = ~idx_A_test

        groups_test = [
            [test_labels[idx_A_test, 0], test_labels[idx_B_test, 0]],
            [test_labels[idx_A_test, 1], test_labels[idx_B_test, 1]]
        ]
    else:
        groups_test = None

    return plot_two_km_curves_per_cohort(
        axs,
        groups_train=groups_train,
        groups_valid=groups_valid,
        groups_test=groups_test,
        subtitle=subtitle,
        save_dir=save_dir,
        y_label=y_label,
        strata_labels=strata_labels,
        annotation=annotation,
        time_in_month=time_in_month,
        titles=titles,
        set_titles=set_titles,
        table_below_scaling=table_below_scaling)


def plot_kms(pred_train, train_labels, pred_valid, valid_labels,
             pred_test=None, test_labels=None,
             subtitle=None, save_dir=None, at_risk_counts=True,
             time_in_month=True,
             y_label="Loco-regional tumour control",
             axs=None,
             titles=None,
             set_titles=True,
             table_below_scaling=0.15):
    """
    Create KM Plots for high/low risk groups of training, validation
    and if given, for test cases.
    Group stratification is based on the median value of the training
    cases.
    """
    n_plots = 2 if (pred_test is None or test_labels is None) else 3
    if axs is None:
        # we manually create a new figure
        width = 4 * n_plots
        height = 5
        fig, axs = plt.subplots(
            1, n_plots, sharey=True, figsize=(width, height), dpi=300)
    else:
        assert len(axs) == n_plots

    median_risk_train = np.median(pred_train)
    print("plot_kms: threshold for stratification:", median_risk_train)

    condition = lambda x: x <= median_risk_train

    return plot_stratified_cohort_km(
        axs, condition, pred_train, train_labels, pred_valid, valid_labels,
        pred_test=pred_test, test_labels=test_labels,
        subtitle=subtitle, save_dir=save_dir, at_risk_counts=at_risk_counts,
        strata_labels=["Low risk", "High risk"],
        y_label=y_label,
        annotation="threshold={:.3f}".format(median_risk_train),
        time_in_month=time_in_month,
        titles=titles,
        set_titles=set_titles,
        table_below_scaling=table_below_scaling)


