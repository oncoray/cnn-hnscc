import argparse
import os
import pandas as pd

from functools import partial, update_wrapper

from keras import optimizers

import sys

from ..losses import neg_cox_log_likelihood


class SurvivalCVCmdLineParserBase(object):
    """
    Parses command line args for cross validation and returns them as dictionary
    """
    def __init__(self, description):
        self.parser = None
        self.parser_groups = []
        self._create_parser(
            description=description)

        print("finished self._create_parser")
        # the arguments comming from the commandline
        self.parsed_args, unknown = self.parser.parse_known_args()
        if unknown:
            print("[W]: unknown args passed:", unknown)

        # how we interpret the command line arguments
        self._make_arg_dict()
        print("got arg_dict", self.arg_dict)

    def _create_parser(self, description):
        self.parser = argparse.ArgumentParser(
            description=description)

        self.add_data_handler_args()

        self.add_context_args()

        self.add_cv_args()

        # arguments that are not for any of the categories
        self.parser.add_argument(
            "-o", "--output", type=str, help="output directory (must not exist yet!)", default=os.getcwd())

    def _get_parser_group(self, group_name):
        for g in self.parser_groups:
            if g.title == group_name:
                return g

        return None

    def _get_or_create_group(self, group_name):
        """Returns the parser group if it exists or creates it if not."""

        g = self._get_parser_group(group_name)
        if g is None:
            g = self.parser.add_argument_group(group_name)
            self.parser_groups.append(g)

        return g

    def _add_data_augmentation_args(self, parser_group):
        parser_group.add_argument(
            "--shear_range", type=float, default=0.)
        parser_group.add_argument(
            "--zoom_range", type=float, default=0.)
        parser_group.add_argument(
            "--rotation_range", type=float, default=0.)
        parser_group.add_argument(
            "--horizontal_flip", default=False, action="store_true")
        parser_group.add_argument(
            "--fill_mode", type=str, default="nearest",
            choices=["constant", "nearest", "reflect", "wrap"])

    def add_data_handler_args(self):
        self.parser.add_argument(
            "--input", type=str, help="base path to image input. Needs to contains subdirectories for all specified cohorts.",
            nargs="+")
        self.parser.add_argument(
            "--outcome", type=str, help="The file containing the outcome information.")
        self.parser.add_argument(
            "--id_col", type=str, help="The column name of the outcome file for patient ids.")
        self.parser.add_argument(
            "--time_col", type=str, help="The column name of the outcome file for event times.")
        self.parser.add_argument(
            "--event_col", type=str, help="The column name of the outcome file for event indicators.")

        train_group = self._get_or_create_group("Training")
        train_group.add_argument(
            "-b", "--batch", type=int, help="batch size", default=32)

        augment_group = self._get_or_create_group("Data augmentation")
        augment_group.add_argument(
            "--no_data_augmentation", default=False, action="store_true",
            help="Disable data augmentation. Other augmentation parameters will be ignored.")
        self._add_data_augmentation_args(augment_group)

        preproc_group = self._get_or_create_group("Preprocessing")
        preproc_group.add_argument(
            "--time_perturb", type=float, help="bound on random numbers added to event-times of slices of same patient to avoid ties.", default=0.)
        preproc_group.add_argument(
            "--mask_nontumor_areas", default=False, action="store_true",
            help="Multiplies image and masks pointwise to mask nontumor areas.")

    def _get_data_augmentation_args(self):
        # data-augmentation args
        return {
            "shear_range": self.parsed_args.shear_range,
            "zoom_range": self.parsed_args.zoom_range,
            "rotation_range": self.parsed_args.rotation_range,
            "horizontal_flip": self.parsed_args.horizontal_flip,
            "fill_mode": self.parsed_args.fill_mode
        }

    def get_data_handler_args(self):
        return {
            'input': self.parsed_args.input,
            'outcome': self.parsed_args.outcome,
            'id_col': self.parsed_args.id_col,
            'time_col': self.parsed_args.time_col,
            'event_col': self.parsed_args.event_col,
            'max_time_perturb': self.parsed_args.time_perturb,
            'mask_nontumor_areas': self.parsed_args.mask_nontumor_areas,
            'batch_size': self.parsed_args.batch,
            'no_data_augmentation': self.parsed_args.no_data_augmentation,
            'training_augmentation_args': self._get_data_augmentation_args(),
            'validation_augmentation_args': {}  # TODO: add if we need it
        }

    def add_context_args(self):
        # OPTIMIZATION
        opti_group = self._get_or_create_group("Optimization")
        opti_group.add_argument(
            "--opti", type=str, help="optimizer to use", default="adam", choices=["adam", "sgd"])
        opti_group.add_argument(
            "--lr", type=float, help="learning rate (multiple okay for more than one training of the same model)", default=1.e-4, nargs="+")
        opti_group.add_argument(
            "--loss", type=str, help="loss function", default="cox")

        # Network training
        # add it to the right group
        train_group = self._get_or_create_group("Training")

        train_group.add_argument("-e", "--epochs", type=int, help="number of training epochs (multiple okay for more than one training of the same model)", default=20, nargs="+")
        # also possible to provide a file that contains the IDs to use for training (exclusive with train_fraction)
        train_exclusives = train_group.add_mutually_exclusive_group()
        #train_exclusives = parser.add_mutually_exclusive_group()
        train_exclusives.add_argument("-f", "--fraction", type=float, help="fraction of training data (between 0 and 1)")
        train_exclusives.add_argument("--train_id_file", type=str, help="filepath that contains patient IDs for training")

        train_group.add_argument(
            "--batchnorm", type=str, help="Whether and how to apply batch normalization in the model.",
            default="", choices=["", "pre_act", "post_act"])
        train_group.add_argument(
            "--finalact", type=str, help="Type of activation for the output layer.",
            default="tanh")
        train_group.add_argument(
            "--lrelu", type=float, help="Value for lReLU layers which determines the slope for negative input.",
            default=0.)
        train_group.add_argument(
            "--dropout", type=float, help="Value for dropout probability.",
            default=0.)
        train_group.add_argument(
            "--l1", type=float, help="Value for L1-regularization of weights",
            default=0.)
        train_group.add_argument(
            "--l2", type=float, help="Value for L2-regularization of weights",
            default=0.)

    def get_context_args(self):
        # Optimization args
        if self.parsed_args.opti == "adam":
            print("NOTE: using adam with amsgrad=True!")
            optimizer_cls = update_wrapper(
                partial(optimizers.Adam, amsgrad=True), optimizers.Adam)
        else:
            print(f"NOTE: unknown optimizer {self.parsed_args.opti}. Will use SGD!")
            optimizer_cls = optimizers.SGD

        if self.parsed_args.loss == "cox":
            loss = neg_cox_log_likelihood
        else:
            loss = self.parsed_args.loss

        optimization = {
            "optimizer_cls": optimizer_cls,
            "loss": loss,
            "lr": self.parsed_args.lr if isinstance(
                self.parsed_args.lr, list) else [self.parsed_args.lr],  # can be single or multiple values,
        }

        # training args
        training = {
            "epochs": self.parsed_args.epochs if isinstance(
                self.parsed_args.epochs, list) else [self.parsed_args.epochs],  # can be single or multiple values
            "finalact": self.parsed_args.finalact,
            "lrelu": self.parsed_args.lrelu,
            "dropout": self.parsed_args.dropout,
            "l1": self.parsed_args.l1,
            "l2": self.parsed_args.l2,
        }
        # handle exclusivity of train_fraction and train_ids
        if getattr(self.parsed_args, "fraction") and getattr(self.parsed_args, "train_id_file"):
            # both arguments were set and we need to raise an error
            # but this is excluded by argparse already
            raise IOError("Options 'fraction' and 'train_id_file' can not be set at the same time!")
        elif getattr(self.parsed_args, "fraction"):
            assert (self.parsed_args.fraction >= 0. and self.parsed_args.fraction <= 1.)
            training["train_fraction"] = self.parsed_args.fraction
        elif getattr(self.parsed_args, "train_id_file"):
            training["train_ids"] = pd.read_csv(self.parsed_args.train_id_file, header=None).values.squeeze()
        else:
            # none of the options was set so we default to a fraction
            print("No training cohort option was set. Using fraction=0.8")
            training["train_fraction"] = 0.8

        batchnorm = self.parsed_args.batchnorm
        if batchnorm == "":
            batchnorm = None
        training["batchnorm"] = batchnorm

        d = {}
        d.update(optimization)
        d.update(training)

        return d

    def add_cv_args(self):
        cv_group = self.parser.add_argument_group("Cross validation")
        cv_group.add_argument(
            "-k", "--kfold", type=int, help="number of cross-validation splits", default=3)
        cv_group.add_argument(
            "-r", "--reps", type=int, help="number of runs for cross-validation", default=1)

    def get_cv_args(self):
        return {
            "reps":  self.parsed_args.reps,
            "folds":  self.parsed_args.kfold}

    def _make_arg_dict(self):

        args = {'output_dir': self.parsed_args.output}
        args['data_handler'] = self.get_data_handler_args()
        args['context'] = self.get_context_args()
        args['cross_validation'] = self.get_cv_args()

        self.arg_dict = args


class SurvivalCVCmdLineParser(SurvivalCVCmdLineParserBase):
    def add_data_handler_args(self):
        super().add_data_handler_args()

        # now add the nslices arg to the preprocessing group
        group = self._get_or_create_group("Preprocessing")
        group.add_argument(
            "-n", "--nslices", type=int, default=0, nargs="+", help="number of slices per patient (below and above slice with largest tumor area)")

    def get_data_handler_args(self):
        args = super().get_data_handler_args()

        # preproc args
        if isinstance(self.parsed_args.nslices, int):
            n_around_max = self.parsed_args.nslices
        elif len(self.parsed_args.nslices) != 2:
            raise ValueError("--nslices (-n) should be passed only two values!")
        else:
            n_around_max = tuple(self.parsed_args.nslices)

        args["slices_around_max"] = n_around_max

        return args
