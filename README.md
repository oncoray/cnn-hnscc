# Steps to run our analysis

Please be aware that, due to data protection regulations, it was not possible to make the CT data of our cohort publically available.

1. Installation of the dl_toolbox package
    Navigate to the directory where this README file is located and run (maybe create a new virtualenv before)
    ```
    pip install -r requirements.txt
    python setup.py install
    ```

2. Go into the 'analysis_scripts' directory
    Select and edit one of the shell scripts that you would like to run (e.g. run_cv_train_from_scratch.sh).

    After editing paths and hyperparameters, run the shell script from the command line like

    ```
    ./run_cv_train_from_scratch.sh
    ```

    To get help on available command line options and their meanings,
    you can also run the underlying python script that is called from
    the shell script, e.g.
    ```
    python cv_train_from_scratch.py --help
    ```

    Please note that the `input` argument on the command line expects a path to a
    directory that contains a single subdirectory for each patient named after the
    patients ID. Within the subdirectory, the CT scan and the segmentation mask
    have to be provided as numpy array files ({ct/roi}.npy or {ct/roi}.npz).


# Loading existing models

Because all our models were trained with keras using a custom loss function, models
have to be loaded in the following way (after installation of the dl_toolbox package)

```
from dl_toolbox.losses import neg_cox_log_likelihood
from keras.models import load_model
from keras.utils import CustomObjectScope

with CustomObjectScope({"neg_cox_log_likelihood": neg_cox_log_likelihood}):
    model = load_model(<path_to_the_model>)
```


