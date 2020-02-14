import json
import os
import numpy as np
import pandas as pd

from dl_toolbox.cross_validation import CrossValidator
from dl_toolbox.cross_validation.cmd_line_parser import SurvivalCVCmdLineParser
from dl_toolbox.cross_validation.ensemble import evaluate_ensemble_cindex
from dl_toolbox.data.data_handler import DataHandler


def run_cv_from_cmd(cv_context_cls, parser_cls=SurvivalCVCmdLineParser,
                    data_handler_cls=DataHandler,
                    ensemble_method=np.mean,
                    ensemble_metric_fns=[evaluate_ensemble_cindex]):
    """
    Parameters
    ----------
    cv_context: instance of SurvivalCVContext
    """
    parser = parser_cls(
        "Cross-validation of survival neural network training.")

    args = parser.arg_dict

    output_dir = args.get("output_dir", "./cv_output")

    data_handler = data_handler_cls(**args['data_handler'])

    cv_context = cv_context_cls(
        data_handler=data_handler, **args['context'])

    cross_validator = CrossValidator(
        cv_context=cv_context, **args['cross_validation'])

    # run the cross validation trainings
    pred_dfs, perf_dfs = cross_validator.run_cross_validation(
        output_dir=output_dir)

    # print the performances and the summaries with CI and p values
    full_perf_df = pd.concat(perf_dfs, ignore_index=True, sort=False)
    tmp = full_perf_df[
        ["rep", "fold"] + [c for c in full_perf_df.columns if c.endswith("_pat")]]
    print("\n\nPERFORMANCE OF CV:\n", tmp)
    print("summary:")
    print(tmp.describe())

    # number of successful stratifications
    pval_cols = [c for c in tmp.columns if "p_val" in c and "_pat" in c]
    for c in pval_cols:
        n_successful = len(tmp[tmp[c] < 0.05])
        print("{}: {}/{} with p<0.05".format(c, n_successful, len(tmp)))

    try:
        ensemble_result = cross_validator.evaluate_ensemble_performance(
            pred_dfs, ensemble_method=ensemble_method,
            ensemble_metric_fns=ensemble_metric_fns,
            output_dir=output_dir)

        print("Ensemble result:")
        for k, v in ensemble_result.items():
            print(k, "\n", v)

    except Exception as e:
        print("Ensemble evaluation failed!", e)

    # also store the commandline args in the output_dir
    cmd_file = os.path.join(output_dir, "cmd_args_used.json")
    with open(cmd_file, "w") as of:
        json.dump(vars(parser.parsed_args), of)