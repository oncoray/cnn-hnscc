from keras.callbacks import Callback
from lifelines.utils import concordance_index

import numpy as np
from warnings import warn


def time_and_event_from_label(time_and_maybe_event):
    # check that all tuples are of the same size
    # assert len(np.unique([len(tup) for tup in arr_of_tuples])) == 1

    # maybe we have time and event status or only the time
    # in the second case we assume that the event happened
    if time_and_maybe_event.ndim >= 2:
        surv = time_and_maybe_event[:, 0]
        event = time_and_maybe_event[:, 1]
    else:
        # silently assume an event status of 1
        print("WARN: time_and_event_from_label got 1d array: assume those are event times and create event_status=1!")
        surv = time_and_maybe_event
        event = np.array([1 for i in range(len(time_and_maybe_event))])

    return surv, event


def check_and_convert_to_list(inputs):
    if not isinstance(inputs, list):
        inputs = [inputs]

    if len(inputs) == 0:
        raise ValueError("No inputs were given.")

    return inputs


class ConcordanceIndex(Callback):
    """Estimate concordance index after every epoch."""

    def _check_input_data_shape(self, data):
        # potentially more than one input in the data
        data = check_and_convert_to_list(data)
        data_shapes = [elem.shape[1:] for elem in data]
        # ensure that model.inputs is always a list
        model_inputs = check_and_convert_to_list(self.model.inputs)
        model_input_shapes = [input._keras_shape[1:] for input in model_inputs]

        for given_shape, model_shape in zip(data_shapes, model_input_shapes):
            assert len(model_shape) == len(given_shape)
            for i, s in enumerate(given_shape):
                if model_shape[i] is not None and model_shape[i] != s:
                    raise ValueError(
                        "Model expects different input shape! Given: {0}, model"
                        " expects: {1}".format(
                            given_shape, model_shape))

    def __init__(self, train_data, valid_data, freq=1, output_idx=0):
        """
        Parameters
        ----------
        train_data: tuple of length 2
            first element np.array of features, second element a list
            of labels where we assume first label to be
            np.array with two columns for survival time and event
            (in this order)

        valid_data: tuple of length 2
            first element np.array of features, second element a list
            of labels where we assume first label to be
            np.array with two columns for survival time and event
            (in this order)
        freq: int
            the epoch frequency that this callback should execute. By default
            it runs every epoch. Set it e.g. to 10 to execute every 10 epochs.

        output_idx: int
            used if a model has multiple outputs so we know which
            one to use to compute the concordance index.
        """

        self.train_data, self.train_labels = train_data
        self.valid_data, self.valid_labels = valid_data
        self.freq = freq
        self.output_idx = output_idx

        self.train_surv = None
        self.train_event = None
        self.valid_surv = None
        self.valid_event = None

    def on_train_begin(self, logs=None):
        # check if we have multiple inputs and the input shape
        # matches the self.model input shape
        self._check_input_data_shape(self.train_data)
        self._check_input_data_shape(self.valid_data)

        # check if we have multiple outputs (i.e. multiple labels)
        self.train_labels = check_and_convert_to_list(self.train_labels)
        self.valid_labels = check_and_convert_to_list(self.valid_labels)

        # assume the first label in the list is the survival tuple
        self.train_surv, self.train_event = time_and_event_from_label(
            self.train_labels[self.output_idx])
        self.valid_surv, self.valid_event = time_and_event_from_label(
            self.valid_labels[self.output_idx])

    def on_epoch_end(self, epoch, logs=None):

        if epoch % self.freq == 0:
            # here the shape of train/valid data needs to match the models
            # input shape
            train_risk = self.model.predict(self.train_data)
            valid_risk = self.model.predict(self.valid_data)

            if isinstance(train_risk, list):
                # model has more than one output
                train_risk = train_risk[self.output_idx]
                valid_risk = valid_risk[self.output_idx]

            # catch issues from lifelines
            try:
                c_train = concordance_index(
                    self.train_surv.squeeze(), train_risk.squeeze(),
                    self.train_event.squeeze())
            except Exception as e:
                c_train = np.nan
                print(warn(e))
            try:
                c_valid = concordance_index(
                    self.valid_surv.squeeze(), valid_risk.squeeze(),
                    self.valid_event.squeeze())
            except Exception as e:
                c_valid = np.nan
                print(warn(e))

            print("ci: {}, val_ci: {}".format(c_train, c_valid))
            logs['ci'] = c_train
            logs['val_ci'] = c_valid
