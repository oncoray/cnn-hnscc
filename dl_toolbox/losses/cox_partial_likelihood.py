import tensorflow as tf


def cox_log_likelihood(labels, risk):
    """
    The negative of the cox partial log-likelihood (since in DL one
    typically minimizes a loss instead of maximizing likelihood).

    Parameters
    ----------
    labels: 2d numpy array (n_samples x 2)
        first column has survival times, second column event status

    This does not assume that data is ordered by decreasing
    survival time but does it beforehand.
    """
    t = labels[:, 0]
    e = labels[:, 1]

    # sort data
    t_ordered, idx = tf.nn.top_k(t, k=tf.size(t))
    e_ordered = tf.gather(e, indices=idx, axis=0)
    risk_ordered = tf.gather(risk, indices=idx, axis=0)

    # compute likelihood
    sum_risk = tf.reduce_sum(e_ordered * risk_ordered)
    log_sums = tf.log(tf.cumsum(tf.exp(risk_ordered)))
    log_sums = tf.reduce_sum(e_ordered * log_sums)

    lcpl = sum_risk - log_sums
    return lcpl


def neg_cox_log_likelihood(labels, risk):
    return -1. * cox_log_likelihood(labels, risk)


if __name__ == "__main__":
    import numpy as np
    from keras import backend as K

    surv_times = [12.3, 14.1, 9.8, 3.6, 7.2]
    events = [0, 1, 1, 0, 1]
    predicted_risks = [-0.5, 0.3, -0.1, .7, .2]

    labels = np.column_stack((surv_times, events))

    res = K.eval(
        cox_log_likelihood(
            K.variable(labels), K.variable(predicted_risks))
    )

    """
    taken from R (v.3.4.4) with the following code:

    library(survival)
    t <- c(12.3, 14.1, 9.8, 3.6, 7.2)
    e <- c(0, 1, 1, 0, 1)
    x1 <- c(-0.5, 0.3, -0.1, .7, .2)

    # do no iterations, only evaluate likelihood for beta(x1)=1 and risk(x_i) = exp(beta*x_i)
    cph_model <- coxph(Surv(t, e) ~ x1, init=c(1), control=coxph.control(iter.max=0))
    cph_model$loglik
    """
    expected_res = -2.357992

    print("result should be close to {}: {}".format(
        expected_res, res))
