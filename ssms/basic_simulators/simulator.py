from ssms.config.config import model_config, boundary_config, drift_config
import numpy as np
import pandas as pd
from copy import deepcopy
import warnings

"""
This module defines the basic simulator function which is the main
workshorse of the package.
In addition some utility functions are provided that help
with preprocessing the output of the simulator function.
"""


def _make_valid_dict(dict_in):
    """Turn all values in dictionary into numpy arrays and make sure,
    that all thetas are either scalars or vectors of the same length

    Arguments:
    ----------
        dict_in: dictionary
            Dictionary of parameters, potentially with different length / types per
            parameter (key)

    Returns:
    --------
        dict_in: dictionary
            Aligned to same size np.float32 np.arrays for every parameter
    """

    collect_lengths = []
    for key, value in dict_in.items():
        # Turn all values into numpy arrays
        if isinstance(value, list):
            dict_in[key] = np.array(value).astype(np.float32)
        elif isinstance(value, int) or isinstance(value, float):
            dict_in[key] = np.array([value]).astype(np.float32)

        # Squeeze all values to make sure they are 1d arrays
        dict_in[key] = np.squeeze(dict_in[key]).astype(np.float32)

        # Check if all thetas are either scalars or vectors of the same length
        if dict_in[key].ndim > 1:
            raise ValueError("Dimension of {} is greater than 1".format(key))
        elif dict_in[key].ndim > 0:
            collect_lengths.append(
                dict_in[key].shape[0]
            )  # add vector parameters to list

    if len(set(collect_lengths)) > 1:
        raise ValueError(
            "thetas have to be either scalars or same length for "
            "all thetas which are not scalars"
        )

    # If there were any thetas provided as vectors (and they had the same length),
    # tile all scalar thetas to that length
    if len(set(collect_lengths)) > 0:
        for key, value in dict_in.items():
            if value.ndim == 0:
                dict_in[key] = np.tile(value, collect_lengths[0])
    else:  # Expand scalars to 1d arrays
        for key, value in dict_in.items():
            if value.ndim == 0:
                dict_in[key] = np.expand_dims(value, axis=0)
    return dict_in


def _theta_dict_to_array(theta=dict(), model_param_list=None):
    """Converts theta dictionary to numpy array for use with simulator function"""
    if model_param_list is None:
        raise ValueError("model_param_list is not supplied")

    return np.stack([theta[param] for param in model_param_list], axis=1).astype(
        np.float32
    )


def _theta_array_to_dict(theta=None, model_param_list=None):
    """Converts theta array to dictionary for use with simulator function"""
    if model_param_list is None:
        raise ValueError("model_param_list is not supplied")
    elif theta is None:
        raise ValueError("theta array is not supplied")
    elif theta.ndim == 1 and len(model_param_list) != theta.shape[0]:
        raise ValueError(
            "model_param_list and theta array do not imply"
            " the same number of parameters"
        )
    elif theta.ndim == 2 and len(model_param_list) != theta.shape[1]:
        raise ValueError(
            "model_param_list and theta array do not imply"
            " the same number of parameters"
        )
    else:
        if theta.ndim == 1:
            theta = np.expand_dims(theta, axis=0).astype(np.float32)
        return {param: theta[:, i] for i, param in enumerate(model_param_list)}


def make_boundary_dict(model_config, model, theta):
    boundary_name = model_config[model]["boundary_name"]
    boundary_params = {
        param_name: value
        for param_name, value in theta.items()
        if param_name in boundary_config[boundary_name]["params"]
    }

    boundary_fun = boundary_config[boundary_name]["fun"]
    boundary_multiplicative = boundary_config[boundary_name]["multiplicative"]
    boundary_dict = {
        "boundary_params": boundary_params,
        "boundary_fun": boundary_fun,
        "boundary_multiplicative": boundary_multiplicative,
    }
    return boundary_dict


def make_drift_dict(model_config, model, theta):
    if "drift_name" in model_config[model].keys():
        drift_name = model_config[model]["drift_name"]
        # print(drift_name)
        # print({param_name: value
        # for param_name, value in theta.items()})
        drift_params = {
            param_name: value
            for param_name, value in theta.items()
            if param_name in drift_config[drift_name]["params"]
        }
        # print('testing drift_params:', drift_params)
        drift_fun = drift_config[drift_name]["fun"]
        drift_dict = {"drift_fun": drift_fun, "drift_params": drift_params}
    else:
        drift_dict = {}
    return drift_dict


# Basic simulators and basic preprocessing
def bin_simulator_output_pointwise(
    out=[0, 0],
    bin_dt=0.04,
    nbins=0,
):  # ['v', 'a', 'w', 't', 'angle']
    """Turns RT part of simulator output into bin-identifier by trial

    Arguments
    ---------
        out: tuple
            Output of the 'simulator' function
        bin_dt: float
            If nbins is 0, this determines the desired
            bin size which in turn automatically
            determines the resulting number of bins.
        nbins: int
            Number of bins to bin reaction time data into.
            If supplied as 0, bin_dt instead determines the
            number of bins automatically.

    Returns
    -------
        2d array. The first columns collects bin-identifiers
        by trial, the second column lists the corresponding choices.
    """
    out_copy = deepcopy(out)

    # Generate bins
    if nbins == 0:
        nbins = int(out["metadata"]["max_t"] / bin_dt)
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, out["metadata"]["max_t"], nbins)
        bins[nbins] = np.inf
    else:
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, out["metadata"]["max_t"], nbins)
        bins[nbins] = np.inf

    out_copy_tmp = deepcopy(out_copy)
    for i in range(out_copy[0].shape[0]):
        for j in range(1, bins.shape[0], 1):
            if out_copy[0][i] > bins[j - 1] and out_copy[0][i] < bins[j]:
                out_copy_tmp[0][i] = j - 1
    out_copy = out_copy_tmp

    out_copy[1][out_copy[1] == -1] = 0

    return np.concatenate([out_copy[0], out_copy[1]], axis=-1).astype(np.int32)


def bin_simulator_output(
    out=None,
    bin_dt=0.04,
    nbins=0,
    max_t=-1,
    freq_cnt=False,
):  # ['v', 'a', 'w', 't', 'angle']
    """Turns RT part of simulator output into bin-identifier by trial

    Arguments
    ---------
        out : tuple
            Output of the 'simulator' function
        bin_dt : float
            If nbins is 0, this determines the desired
            bin size which in turn automatically
            determines the resulting number of bins.
        nbins : int
            Number of bins to bin reaction time data into.
            If supplied as 0, bin_dt instead determines the number of
            bins automatically.
        max_t : int <default=-1>
            Override the 'max_t' metadata as part of the simulator output.
            Sometimes useful, but usually default will do the job.
        freq_cnt : bool <default=False>
            Decide whether to return proportions (default) or counts in bins.

    Returns
    -------
        A histogram of counts or proportions.

    """

    if max_t == -1:
        max_t = out["metadata"]["max_t"]

    # Generate bins
    if nbins == 0:
        nbins = int(max_t / bin_dt)
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf
    else:
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf

    cnt = 0
    counts = np.zeros((nbins, len(out["metadata"]["possible_choices"])))

    for choice in out["metadata"]["possible_choices"]:
        counts[:, cnt] = np.histogram(out["rts"][out["choices"] == choice], bins=bins)[
            0
        ]
        cnt += 1

    if freq_cnt is False:
        counts = counts / out["metadata"]["n_samples"]

    return counts


def bin_arbitrary_fptd(
    out=None,
    bin_dt=0.04,
    nbins=256,
    nchoices=2,
    choice_codes=[-1.0, 1.0],
    max_t=10.0,
):  # ['v', 'a', 'w', 't', 'angle']
    """Takes in simulator output and returns a histogram of bin counts
    Arguments
    ---------
        out: tuple
            Output of the 'simulator' function
        bin_dt : float
            If nbins is 0, this determines the desired bin size
            which in turn automatically determines the resulting number of bins.
        nbins : int
            Number of bins to bin reaction time data into.
            If supplied as 0, bin_dt instead determines the number of
            bins automatically.
        nchoices: int <default=2>
            Number of choices allowed by the simulator.
        choice_codes = list <default=[-1.0, 1.0]
            Choice labels to be used.
        max_t: float
            Maximum RT to consider.

    Returns
    -------
        2d array (nbins, nchoices): A histogram of bin counts
    """

    # Generate bins
    if nbins == 0:
        nbins = int(max_t / bin_dt)
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf
    else:
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf

    cnt = 0
    counts = np.zeros((nbins, nchoices))

    for choice in choice_codes:
        counts[:, cnt] = np.histogram(out[:, 0][out[:, 1] == choice], bins=bins)[0]
        cnt += 1
    return counts


def validate_ssm_parameters(model, theta):

    def check_num_drifts_and_actions(drifts, num_actions):
        drifts = np.array(drifts)
        if drifts.shape[1] != num_actions:
            raise ValueError("Number of drift rates does not match number of actions")

    def check_lba_drifts_sum(drifts):
        v_sum = np.sum(drifts, axis=1)
        if np.any(v_sum <= 0.99) or np.any(v_sum >= 1.01):
            raise ValueError("Drift rates do not sum to 1 for each trial")

    def check_if_z_gt_a(z, a):
        if np.any(z >= a):
            raise ValueError("Starting point z >= a for at least one trial")

    if model in ["lba_3_v1", "lba_angle_3_v1", "rlwm_lba_race_v1"]:
        if model in ["lba_3_v1", "lba_angle_3_v1"]:
            # check_num_drifts_and_actions(theta['v'], model_config[model]['nchoices'])
            check_lba_drifts_sum(theta["v"])
            check_if_z_gt_a(theta["z"], theta["a"])
        elif model in ["rlwm_lba_race_v1"]:
            # check_num_drifts_and_actions(theta['v_RL'], model_config[model]['nchoices'])
            # check_num_drifts_and_actions(theta['v_WM'], model_config[model]['nchoices'])
            check_lba_drifts_sum(theta["v_RL"])
            check_lba_drifts_sum(theta["v_WM"])
            check_if_z_gt_a(theta["z"], theta["a"])


def simulator(
    theta,
    model="angle",
    n_samples=1000,
    delta_t=0.001,
    max_t=20,
    no_noise=False,
    bin_dim=None,
    bin_pointwise=False,
    smooth_unif=True,
    return_option="full",
    random_state=None,
):
    """Basic data simulator for the models included in HDDM.

    Arguments
    ---------
        theta : list, numpy.array, dict or pd.DataFrame
            Parameters of the simulator. If 2d array, each row is treated as a 'trial'
            and the function runs n_sample * n_trials simulations.
        deadline : numpy.array <default=None>
            If supplied, the simulator will run a deadline model. RTs will be returned
        model: str <default='angle'>
            Determines the model that will be simulated.
        n_samples: int <default=1000>
            Number of simulation runs for each row in the theta argument.
        delta_t: float
            Size fo timesteps in simulator (conceptually measured in seconds)
        max_t: float
            Maximum reaction the simulator can reach
        no_noise: bool <default=False>
            Turn noise of (useful for plotting purposes mostly)
        bin_dim: int <default=None>
            Number of bins to use (in case the simulator output is
            supposed to come out as a count histogram)
        bin_pointwise: bool <default=False>
            Wheter or not to bin the output data pointwise.
            If true the 'RT' part of the data is now specifies the
            'bin-number' of a given trial instead of the 'RT' directly.
            You need to specify bin_dim as some number for this to work.
        return_option: str <default='full'>
            Determines what the function returns. Can be either
            'full' or 'minimal'. If 'full' the function returns
            a dictionary with keys 'rts', 'responses' and 'metadata', and
            metadata contains the model parameters and some additional
            information. 'metadata' is a simpler dictionary with less information
            if 'minimal' is chosen.
        random_state: int <default=None>
            Integer passed to random_seed function in the simulator.
            Can be used for reproducibility.

    Return
    ------
    dictionary where keys
        can be (rts, responses, metadata)
        or     (rt-response histogram, metadata)
        or     (rts binned pointwise, responses, metadata)

    """
    # Preprocess theta to be a 2d numpy array with correct column ordering
    # (if supplied as 2d array or list in the first place,
    # user has to supply the correct ordering to begin with)

    if isinstance(theta, list):
        theta = np.asarray(theta).astype(np.float32)
    elif isinstance(theta, np.ndarray):
        theta = theta.astype(np.float32)
    elif isinstance(theta, dict):
        theta = _make_valid_dict(deepcopy(theta))
    elif isinstance(theta, pd.DataFrame):
        theta = theta.to_dict(orient="list")
        theta = {k: np.array(v).astype(np.float32) for k, v in theta.items()}
    else:
        try:
            import torch

            if isinstance(theta, torch.Tensor):
                theta = theta.numpy().astype(np.float32)
            else:
                raise ValueError(
                    "theta is not supplied as list, numpy array,"
                    " dictionary or torch tensor!"
                )
        except ImportError as e:
            raise e

    # Turn theta into array if it is a dictionary
    # if isinstance(theta, dict):
    #     theta = _theta_dict_to_array(theta, model_config[model]["params"])

    if "_deadline" in model:
        deadline = True
        model = model.replace("_deadline", "")
    else:
        deadline = False

    # Make sure theta is a dict going forward
    if not isinstance(theta, dict):
        if deadline:
            theta = _theta_array_to_dict(
                theta, model_config[model]["params"] + ["deadline"]
            )
            warnings.warn(
                "Deadline model request, and theta not supplied as dict."
                + "Make sure to supply the deadline parameters in last position!"
            )
        else:
            theta = _theta_array_to_dict(theta, model_config[model]["params"])

    n_trials = theta[model_config[model]["params"][0]].shape[0]
    if not deadline:
        # print('Setting mock deadline to 999 (this should never have an effect)')
        theta["deadline"] = np.tile(np.array([999], dtype=np.float32), n_trials)

    # Initialize dictionary that collects
    # simulator inputs that are commong across simulator functions
    sim_param_dict = {
        "max_t": max_t,
        "s": 0.0,
        "n_samples": n_samples,
        "n_trials": n_trials,
        "delta_t": delta_t,
        "random_state": random_state,
        "return_option": return_option,
        "smooth": smooth_unif,
    }

    boundary_dict = make_boundary_dict(model_config, model, theta)
    drift_dict = make_drift_dict(model_config, model, theta)

    # 2 choice models (single particle)
    # The correct settings for the noise parameters in the simulator
    # depends on context. We predefine a dictionary to collect all
    # relevant settings here and fill in the correct value given
    # the actual model string.

    noise_dict = {
        "1_particles": 1.0,
        "2_particles": np.tile(
            np.array(
                [1.0] * 2,
                dtype=np.float32,
            ),
            (n_trials, 1),
        ),
        "3_particles": np.tile(
            np.array(
                [1.0] * 3,
                dtype=np.float32,
            ),
            (n_trials, 1),
        ),
        "4_particles": np.tile(
            np.array(
                [1.0] * 4,
                dtype=np.float32,
            ),
            (n_trials, 1),
        ),
        "lba_based_models": 0.1,
    }

    if no_noise:
        noise_dict = {key: value * 0.0 for key, value in noise_dict.items()}

    if model in [
        "glob",
        "ddm",
        "angle",
        "weibull",
        "weibull_cdf",
        "ddm_hddm_base",
        "ddm_legacy",  # AF-TODO what was DDM legacy?
        "levy",
        "levy_angle",
        "full_ddm",
        "full_ddm2",
        "full_ddm_legacy",
        "full_ddm_hddm_base",
        "ddm_sdv",
        "ornstein",
        "ornstein_uhlenbeck",
        "ornstein_angle",
        "gamma_drift",
        "gamma_drift_angle",
    ]:
        sim_param_dict["s"] = noise_dict["1_particles"]

    if model in ["ds_conflict_drift", "ds_conflict_drift_angle"]:
        sim_param_dict["s"] = noise_dict["1_particles"]
        theta["v"] = np.tile(np.array([0], dtype=np.float32), n_trials)

    # Multi-particle models

    #   LBA-based models

    # lba_sd = 0.1
    if model == "lba_3_v1":
        sim_param_dict["sd"] = noise_dict["lba_based_models"]
        theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        theta["z"] = np.expand_dims(theta["z"], axis=1)

    if model == "lba_angle_3_v1":
        sim_param_dict["sd"] = noise_dict["lba_based_models"]
        theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        theta["z"] = np.expand_dims(theta["z"], axis=1)
        theta["theta"] = np.expand_dims(theta["theta"], axis=1)

    if model == "rlwm_lba_race_v1":
        sim_param_dict["sd"] = noise_dict["lba_based_models"]
        theta["v_RL"] = np.column_stack(
            [theta["v_RL_0"], theta["v_RL_1"], theta["v_RL_2"]]
        )
        theta["v_WM"] = np.column_stack(
            [theta["v_WM_0"], theta["v_WM_1"], theta["v_WM_2"]]
        )
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        theta["z"] = np.expand_dims(theta["z"], axis=1)

    validate_ssm_parameters(model, theta)

    # 2 Choice
    if model == "race_2":
        sim_param_dict["s"] = noise_dict["2_particles"]
        theta["z"] = np.column_stack([theta["z0"], theta["z1"]])
        theta["v"] = np.column_stack([theta["v0"], theta["v1"]])
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)

    if model in ["race_no_bias_2", "race_no_bias_angle_2"]:
        sim_param_dict["s"] = noise_dict["2_particles"]
        theta["z"] = np.column_stack([theta["z"], theta["z"]])
        theta["v"] = np.column_stack([theta["v0"], theta["v1"]])
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)

    if model in ["race_no_z_2", "race_no_z_angle_2"]:
        sim_param_dict["s"] = noise_dict["2_particles"]
        theta["z"] = np.tile(np.array([0.0] * 2, dtype=np.float32), (n_trials, 1))
        theta["v"] = np.column_stack([theta["v0"], theta["v1"]])
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)

    # 3 Choice models

    if model == "race_3":
        sim_param_dict["s"] = noise_dict["3_particles"]
        theta["z"] = np.column_stack([theta["z0"], theta["z1"], theta["z2"]])
        theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)

    if model in ["race_no_bias_3", "race_no_bias_angle_3"]:
        sim_param_dict["s"] = noise_dict["3_particles"]
        theta["z"] = np.column_stack([theta["z"], theta["z"], theta["z"]])
        theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)

    if model in ["race_no_z_3", "race_no_z_angle_3"]:
        sim_param_dict["s"] = noise_dict["3_particles"]
        theta["z"] = np.tile(np.array([0.0] * 3, dtype=np.float32), (n_trials, 1))
        theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)

    if model == "lca_3":
        sim_param_dict["s"] = noise_dict["3_particles"]
        theta["z"] = np.column_stack([theta["z0"], theta["z1"], theta["z2"]])
        theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        theta["g"] = np.expand_dims(theta["g"], axis=1)
        theta["b"] = np.expand_dims(theta["b"], axis=1)

    if model in ["lca_no_bias_3", "lca_no_bias_angle_3"]:
        sim_param_dict["s"] = noise_dict["3_particles"]
        theta["z"] = np.column_stack([theta["z"], theta["z"], theta["z"]])
        theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        theta["g"] = np.expand_dims(theta["g"], axis=1)
        theta["b"] = np.expand_dims(theta["b"], axis=1)

    if model in ["lca_no_z_3", "lca_no_z_angle_3"]:
        sim_param_dict["s"] = noise_dict["3_particles"]
        theta["z"] = np.tile(np.array([0.0] * 3, dtype=np.float32), (n_trials, 1))
        theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        theta["g"] = np.expand_dims(theta["g"], axis=1)
        theta["b"] = np.expand_dims(theta["b"], axis=1)

    # 4 Choice models

    if model == "race_4":
        sim_param_dict["s"] = noise_dict["4_particles"]
        theta["z"] = np.column_stack(
            [theta["z0"], theta["z1"], theta["z2"], theta["z3"]]
        )
        theta["v"] = np.column_stack(
            [theta["v0"], theta["v1"], theta["v2"], theta["v3"]]
        )
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)

    if model in ["race_no_bias_4", "race_no_bias_angle_4"]:
        sim_param_dict["s"] = noise_dict["4_particles"]
        theta["z"] = np.column_stack([theta["z"], theta["z"], theta["z"], theta["z"]])
        theta["v"] = np.column_stack(
            [theta["v0"], theta["v1"], theta["v2"], theta["v3"]]
        )
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)

    if model in ["race_no_z_4", "race_no_z_angle_4"]:
        sim_param_dict["s"] = noise_dict["4_particles"]
        theta["z"] = np.tile(np.array([0.0] * 4, dtype=np.float32), (n_trials, 1))
        theta["v"] = np.column_stack(
            [theta["v0"], theta["v1"], theta["v2"], theta["v3"]]
        )
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)

    if model == "lca_4":
        sim_param_dict["s"] = noise_dict["4_particles"]
        theta["z"] = np.column_stack(
            [theta["z0"], theta["z1"], theta["z2"], theta["z3"]]
        )
        theta["v"] = np.column_stack(
            [theta["v0"], theta["v1"], theta["v2"], theta["v3"]]
        )
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        theta["g"] = np.expand_dims(theta["g"], axis=1)
        theta["b"] = np.expand_dims(theta["b"], axis=1)

    if model in ["lca_no_bias_4", "lca_no_bias_angle_4"]:
        sim_param_dict["s"] = noise_dict["4_particles"]
        theta["z"] = np.column_stack([theta["z"], theta["z"], theta["z"], theta["z"]])
        theta["v"] = np.column_stack(
            [theta["v0"], theta["v1"], theta["v2"], theta["v3"]]
        )
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        theta["g"] = np.expand_dims(theta["g"], axis=1)
        theta["b"] = np.expand_dims(theta["b"], axis=1)

    if model in ["lca_no_z_4", "lca_no_z_angle_4"]:
        sim_param_dict["s"] = noise_dict["4_particles"]
        theta["z"] = np.tile(np.array([0.0] * 4, dtype=np.float32), (n_trials, 1))
        theta["v"] = np.column_stack(
            [theta["v0"], theta["v1"], theta["v2"], theta["v3"]]
        )
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        theta["g"] = np.expand_dims(theta["g"], axis=1)
        theta["b"] = np.expand_dims(theta["b"], axis=1)

    # Seq / Parallel models (4 choice)

    z_vec = np.tile(np.array([0.5], dtype=np.float32), reps=n_trials)
    g_zero_vec = np.tile(np.array([0.0], dtype=np.float32), reps=n_trials)
    g_vec_leak = np.tile(np.array([2.0], dtype=np.float32), reps=n_trials)
    s_pre_high_level_choice_zero_vec = np.tile(
        np.array([0.0], dtype=np.float32), reps=n_trials
    )
    s_pre_high_level_choice_one_vec = np.tile(
        np.array([1.0], dtype=np.float32), reps=n_trials
    )

    if model in ["ddm_seq2", "ddm_seq2_traj"]:
        sim_param_dict["s"] = noise_dict["1_particles"]

    if model in [
        "ddm_seq2_no_bias",
        "ddm_seq2_angle_no_bias",
        "ddm_seq2_weibull_no_bias",
        "ddm_seq2_conflict_gamma_no_bias",
    ]:
        sim_param_dict["s"] = noise_dict["1_particles"]
        theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]

    if model == "ddm_par2":
        sim_param_dict["s"] = noise_dict["1_particles"]

    if model in [
        "ddm_par2_no_bias",
        "ddm_par2_angle_no_bias",
        "ddm_par2_weibull_no_bias",
        "ddm_par2_conflict_gamma_no_bias",
    ]:
        sim_param_dict["s"] = noise_dict["1_particles"]
        theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]

    if model == "ddm_mic2_adj":
        sim_param_dict["s"] = noise_dict["1_particles"]
        sim_param_dict["s_pre_high_level_choice"] = s_pre_high_level_choice_one_vec
        theta["g"] = g_zero_vec

    if model in [
        "ddm_mic2_adj_no_bias",
        "ddm_mic2_adj_angle_no_bias",
        "ddm_mic2_adj_weibull_no_bias",
        "ddm_mic2_adj_conflict_gamma_no_bias",
    ]:
        sim_param_dict["s"] = noise_dict["1_particles"]
        theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]
        theta["g"] = g_zero_vec
        sim_param_dict["s_pre_high_level_choice"] = s_pre_high_level_choice_one_vec

    # ----- Ornstein version of mic2_adj ---------
    if model == "ddm_mic2_ornstein":
        sim_param_dict["s"] = noise_dict["1_particles"]
        sim_param_dict["s_pre_high_level_choice"] = s_pre_high_level_choice_one_vec

    if model in [
        "ddm_mic2_ornstein_no_bias",
        "ddm_mic2_ornstein_angle_no_bias",
        "ddm_mic2_ornstein_weibull_no_bias",
        "ddm_mic2_ornstein_conflict_gamma_no_bias",
    ]:
        sim_param_dict["s"] = noise_dict["1_particles"]
        theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]
        sim_param_dict["s_pre_high_level_choice"] = s_pre_high_level_choice_one_vec

    if model in [
        "ddm_mic2_ornstein_no_bias_no_lowdim_noise",
        "ddm_mic2_ornstein_angle_no_bias_no_lowdim_noise",
        "ddm_mic2_ornstein_weibull_no_bias_no_lowdim_noise",
        "ddm_mic2_ornstein_conflict_gamma_no_bias_no_lowdim_noise",
    ]:
        sim_param_dict["s"] = noise_dict["1_particles"]
        theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]
        sim_param_dict["s_pre_high_level_choice"] = s_pre_high_level_choice_zero_vec

    # Leak version of mic2
    if model == "ddm_mic2_leak":
        sim_param_dict["s"] = noise_dict["1_particles"]
        theta["g"] = g_vec_leak
        sim_param_dict["s_pre_high_level_choice"] = s_pre_high_level_choice_one_vec

    if model in [
        "ddm_mic2_leak_no_bias",
        "ddm_mic2_leak_angle_no_bias",
        "ddm_mic2_leak_weibull_no_bias",
        "ddm_mic2_leak_conflict_gamma_no_bias",
    ]:
        sim_param_dict["s"] = noise_dict["1_particles"]
        theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]
        theta["g"] = g_vec_leak
        sim_param_dict["s_pre_high_level_choice"] = s_pre_high_level_choice_one_vec

    if model in [
        "ddm_mic2_leak_no_bias_no_lowdim_noise",
        "ddm_mic2_leak_angle_no_bias_no_lowdim_noise",
        "ddm_mic2_leak_weibull_no_bias_no_lowdim_noise",
        "ddm_mic2_leak_conflict_gamma_no_bias_no_lowdim_noise",
    ]:
        sim_param_dict["s"] = noise_dict["1_particles"]
        theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]
        theta["g"] = g_vec_leak
        sim_param_dict["s_pre_high_level_choice"] = s_pre_high_level_choice_zero_vec

    # ----------------- High level dependent noise scaling --------------
    if model in [
        "ddm_mic2_multinoise_no_bias",
        "ddm_mic2_multinoise_angle_no_bias",
        "ddm_mic2_multinoise_weibull_no_bias",
        "ddm_mic2_multinoise_conflict_gamma_no_bias",
    ]:
        sim_param_dict["s"] = noise_dict["1_particles"]
        theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]

    # ----------------- Tradeoff models -----------------
    if model in [
        "tradeoff_no_bias",
        "tradeoff_angle_no_bias",
        "tradeoff_weibull_no_bias",
        "tradeoff_conflict_gamma_no_bias",
    ]:
        sim_param_dict["s"] = noise_dict["1_particles"]
        theta["zh"], theta["zl1"], theta["zl2"] = [z_vec, z_vec, z_vec]

    # print(theta)
    # print(boundary_dict)
    # print(drift_dict)
    # print(sim_param_dict)

    # Call to the simulator
    x = model_config[model]["simulator"](
        **theta,
        **boundary_dict,
        **drift_dict,
        **sim_param_dict,
    )

    # Additional model outputs, easy to compute:
    # Choice probability
    x["choice_p"] = np.zeros((n_trials, len(x["metadata"]["possible_choices"])))
    x["choice_p_no_omission"] = np.zeros(
        (n_trials, len(x["metadata"]["possible_choices"]))
    )
    x["omission_p"] = np.zeros((n_trials, 1))
    x["nogo_p"] = np.zeros((n_trials, 1))
    x["go_p"] = np.zeros((n_trials, 1))

    for k in range(n_trials):
        out_len = x["rts"][:, k, :].shape[0]
        out_len_no_omission = x["rts"][:, k, :][x["rts"][:, k, :] != -999].shape[0]

        for n, choice in enumerate(x["metadata"]["possible_choices"]):
            x["choice_p"][k, n] = (x["choices"][:, k, :] == choice).sum() / out_len
            if out_len_no_omission > 0:
                x["choice_p_no_omission"][k, n] = (
                    x["choices"][:, k, :][x["rts"][:, k, :] != -999] == choice
                ).sum() / out_len_no_omission
            else:
                x["choice_p_no_omission"][k, n] = -999

    x["omission_p"][k, 0] = (x["rts"][:, k, :] == -999).sum() / out_len
    x["nogo_p"][k, 0] = (
        (x["choices"][:, k, :] != max(x["metadata"]["possible_choices"]))
        | (x["rts"][:, k, :] == -999)
    ).sum() / out_len
    x["go_p"][k, 0] = 1 - x["nogo_p"][k, 0]

    # Choice probability no-omission
    # Calculate choice probability only from rts that did not pass a given deadline

    # Output compatibility
    if n_trials == 1:
        x["rts"] = np.squeeze(x["rts"], axis=1)
        x["choices"] = np.squeeze(x["choices"], axis=1)
    if n_trials > 1 and n_samples == 1:
        x["rts"] = np.squeeze(x["rts"], axis=0)
        x["choices"] = np.squeeze(x["choices"], axis=0)

    x["metadata"]["model"] = model

    x["binned_128"] = np.expand_dims(
        bin_simulator_output(x, nbins=128, max_t=-1, freq_cnt=True), axis=0
    )
    x["binned_256"] = np.expand_dims(
        bin_simulator_output(x, nbins=256, max_t=-1, freq_cnt=True), axis=0
    )

    # Adjust in output to binning choice
    if bin_dim == 0 or bin_dim is None:
        return x
    elif bin_dim > 0 and n_trials == 1 and not bin_pointwise:
        binned_out = bin_simulator_output(x, nbins=bin_dim)
        return {"data": binned_out, "metadata": x["metadata"]}
    elif bin_dim > 0 and n_trials == 1 and bin_pointwise:
        binned_out = bin_simulator_output_pointwise(x, nbins=bin_dim)
        return {
            "rts": np.expand_dims(binned_out[:, 0], axis=1),
            "choices": np.expand_dims(binned_out[:, 1], axis=1),
            "metadata": x["metadata"],
        }
    elif bin_dim > 0 and n_trials > 1 and n_samples == 1 and bin_pointwise:
        binned_out = bin_simulator_output_pointwise(x, nbins=bin_dim)
        return {
            "rts": np.expand_dims(binned_out[:, 0], axis=1),
            "choices": np.expand_dims(binned_out[:, 1], axis=1),
            "metadata": x["metadata"],
        }
    elif bin_dim > 0 and n_trials > 1 and n_samples > 1 and bin_pointwise:
        return (
            "currently n_trials > 1 and n_samples > 1, "
            "will not work together with bin_pointwise"
        )
    elif bin_dim > 0 and n_trials > 1 and not bin_pointwise:
        return "currently binned outputs not implemented for multi-trial simulators"
    elif bin_dim == -1:
        return "invalid bin_dim"
