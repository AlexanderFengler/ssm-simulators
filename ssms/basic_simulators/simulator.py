import ssms.config as config
from . import boundary_functions as bf
from . import drift_functions as df
import numpy as np
import sys
from copy import deepcopy
import cssm


# Basic simulators and basic preprocessing
def bin_simulator_output_pointwise(
    out=[0, 0],
    bin_dt=0.04,
    nbins=0,
):  # ['v', 'a', 'w', 't', 'angle']
    """Turns RT part of simulator output into bin-identifier by trial

    :Arguments:
        out: tuple
            Output of the 'simulator' function
        bin_dt: float
            If nbins is 0, this determines the desired bin size which in turn automatically
            determines the resulting number of bins.
        nbins: int
            Number of bins to bin reaction time data into. If supplied as 0, bin_dt instead determines the number of
            bins automatically.

    :Returns:
        2d array. The first columns collects bin-identifiers by trial, the second column lists the corresponding choices.
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

    cnt = 0
    counts = np.zeros((nbins, len(out["metadata"]["possible_choices"])))

    # data_out = pd.DataFrame(np.zeros(( columns = ['rt', 'response'])
    out_copy_tmp = deepcopy(out_copy)
    for i in range(out_copy[0].shape[0]):
        for j in range(1, bins.shape[0], 1):
            if out_copy[0][i] > bins[j - 1] and out_copy[0][i] < bins[j]:
                out_copy_tmp[0][i] = j - 1
    out_copy = out_copy_tmp
    # np.array(out_copy[0] / (bins[1] - bins[0])).astype(np.int32)

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

    :Arguments:
        out : tuple
            Output of the 'simulator' function
        bin_dt : float
            If nbins is 0, this determines the desired bin size which in turn automatically
            determines the resulting number of bins.
        nbins : int
            Number of bins to bin reaction time data into. If supplied as 0, bin_dt instead determines the number of
            bins automatically.
        max_t : int <default=-1>
            Override the 'max_t' metadata as part of the simulator output. Sometimes useful, but usually
            default will do the job.
        freq_cnt : bool <default=False>
            Decide whether to return proportions (default) or counts in bins.

    :Returns:
        A histogram of counts or proportions.

    """

    # print(out)
    # print('passed')
    # print(out.keys())
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

    if freq_cnt == False:
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
    :Arguments:
        out: tuple
            Output of the 'simulator' function
        bin_dt : float
            If nbins is 0, this determines the desired bin size which in turn automatically
            determines the resulting number of bins.
        nbins : int
            Number of bins to bin reaction time data into. If supplied as 0, bin_dt instead determines the number of
            bins automatically.
        nchoices: int <default=2>
            Number of choices allowed by the simulator.
        choice_codes = list <default=[-1.0, 1.0]
            Choice labels to be used.
        max_t: float
            Maximum RT to consider.

    Returns:
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
        # print(np.histogram(out[:, 0][out[:, 1] == choice], bins = bins)[1])
        cnt += 1
    return counts


def simulator(
    theta,
    model="angle",
    n_samples=1000,
    delta_t=0.001,
    max_t=20,
    no_noise=False,
    bin_dim=None,
    bin_pointwise=False,
    random_state=None,
):
    """Basic data simulator for the models included in HDDM.

    :Arguments:
        theta : list or numpy.array
            Parameters of the simulator. If 2d array, each row is treated as a 'trial'
            and the function runs n_sample * n_trials simulations.
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
            Number of bins to use (in case the simulator output is supposed to come out as a count histogram)
        bin_pointwise: bool <default=False>
            Wheter or not to bin the output data pointwise. If true the 'RT' part of the data is now specifies the
            'bin-number' of a given trial instead of the 'RT' directly. You need to specify bin_dim as some number for this to work.
        random_state: int <default=None>
            Integer passed to random_seed function in the simulator. Can be used for reproducibility.

    :Return: dictionary where keys
        can be (rts, responses, metadata)
        or     (rt-response histogram, metadata)
        or     (rts binned pointwise, responses, metadata)

    """
    # Useful for sbi
    if type(theta) == list:
        # print('theta is supplied as list --> simulator assumes n_trials = 1')
        theta = np.asarray(theta).astype(np.float32)
    elif type(theta) == np.ndarray:
        theta = theta.astype(np.float32)
    else:
        theta = theta.numpy().astype(np.float32)     

    if len(theta.shape) < 2:
        theta = np.expand_dims(theta, axis=0)

    n_trials = theta.shape[0]

    # 2 choice models
    if no_noise:
        s = 0.0
    else:
        s = 1.0

    if model == "glob":
        x = cssm.glob_flexbound(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            alphar=theta[:, 3],
            g=theta[:, 4],
            t=theta[:, 5],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 6]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "test":
        x = cssm.ddm_flexbound(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            boundary_params={},
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "ddm_deadline":
        x = cssm.ddm_flexbound_deadline(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            deadline=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            boundary_params={},
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "ddm":
        x = cssm.ddm_flexbound(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            boundary_params={},
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "ddm_hddm_base":
        x = cssm.ddm(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "angle":
        x = cssm.ddm_flexbound(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 4]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "weibull_cdf":
        x = cssm.ddm_flexbound(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            s=s,
            boundary_fun=bf.weibull_cdf,
            boundary_multiplicative=True,
            boundary_params={"alpha": theta[:, 4], "beta": theta[:, 5]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "levy":
        x = cssm.levy_flexbound(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            alpha_diff=theta[:, 3],
            t=theta[:, 4],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "levy_angle":
        x = cssm.levy_flexbound(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            alpha_diff=theta[:, 3],
            t=theta[:, 4],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 5]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "full_ddm":
        x = cssm.full_ddm(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            sz=theta[:, 4],
            sv=theta[:, 5],
            st=theta[:, 6],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "full_ddm_legacy" or model == "full_ddm_hddm_base":
        x = cssm.full_ddm_hddm_base(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            sz=theta[:, 4],
            sv=theta[:, 5],
            st=theta[:, 6],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "ddm_sdv":
        x = cssm.ddm_sdv(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            sv=theta[:, 4],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "ornstein" or model == "ornstein_uhlenbeck":
        x = cssm.ornstein_uhlenbeck(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            g=theta[:, 3],
            t=theta[:, 4],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "ornstein_angle":
        x = cssm.ornstein_uhlenbeck(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            g=theta[:, 3],
            t=theta[:, 4],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 5]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "gamma_drift":
        x = cssm.ddm_flex(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            s=s,
            boundary_fun=bf.constant,
            drift_fun=df.gamma_drift,
            boundary_multiplicative=True,
            boundary_params={},
            drift_params={"shape": theta[:, 4], "scale": theta[:, 5], "c": theta[:, 6]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "gamma_drift_angle":
        x = cssm.ddm_flex(
            v=theta[:, 0],
            a=theta[:, 1],
            z=theta[:, 2],
            t=theta[:, 3],
            s=s,
            boundary_fun=bf.angle,
            drift_fun=df.gamma_drift,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 4]},
            drift_params={"shape": theta[:, 5], "scale": theta[:, 6], "c": theta[:, 7]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "ds_conflict_drift":
        x = cssm.ddm_flex(
            v=np.tile(np.array([0], dtype=np.float32), n_trials),
            a=theta[:, 0],
            z=theta[:, 1],
            t=theta[:, 2],
            s=s,
            boundary_fun=bf.constant,
            drift_fun=df.ds_conflict_drift,
            boundary_params={},
            drift_params={
                "init_p_t": theta[:, 3],
                "init_p_d": theta[:, 4],
                "slope_t": theta[:, 5],
                "slope_d": theta[:, 6],
                "fixed_p_t": theta[:, 7],
                "coherence_t": theta[:, 8],
                "coherence_d": theta[:, 9],
            },
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "ds_conflict_drift_angle":
        x = cssm.ddm_flex(
            v=np.tile(np.array([0], dtype=np.float32), n_trials),
            a=theta[:, 0],
            z=theta[:, 1],
            t=theta[:, 2],
            s=s,
            boundary_fun=bf.angle,
            drift_fun=df.ds_conflict_drift,
            boundary_params={"theta": theta[:, 10]},
            boundary_multiplicative=False,
            drift_params={
                "init_p_t": theta[:, 3],
                "init_p_d": theta[:, 4],
                "slope_t": theta[:, 5],
                "slope_d": theta[:, 6],
                "fixed_p_t": theta[:, 7],
                "coherence_t": theta[:, 8],
                "coherence_d": theta[:, 9],
            },
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    # 3 Choice models
    if no_noise:
        s = np.tile(np.array([0.0, 0.0, 0.0], dtype=np.float32), (n_trials, 1))
    else:
        s = np.tile(np.array([1.0, 1.0, 1.0], dtype=np.float32), (n_trials, 1))

    if model == "race_3":
        x = cssm.race_model(
            v=theta[:, :3],
            a=theta[:, [3]],
            z=theta[:, 4:7],
            t=theta[:, [7]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "race_no_bias_3":
        x = cssm.race_model(
            v=theta[:, :3],
            a=theta[:, [3]],
            z=np.column_stack([theta[:, [4]], theta[:, [4]], theta[:, [4]]]),
            t=theta[:, [5]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "race_no_bias_angle_3":
        x = cssm.race_model(
            v=theta[:, :3],
            a=theta[:, [3]],
            z=np.column_stack([theta[:, [4]], theta[:, [4]], theta[:, [4]]]),
            t=theta[:, [5]],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 6]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "lca_3":
        x = cssm.lca(
            v=theta[:, :3],
            a=theta[:, [3]],
            z=theta[:, 4:7],
            g=theta[:, [7]],
            b=theta[:, [8]],
            t=theta[:, [9]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "lca_no_bias_3":
        x = cssm.lca(
            v=theta[:, :3],
            a=theta[:, [3]],
            z=np.column_stack([theta[:, [4]], theta[:, [4]], theta[:, [4]]]),
            g=theta[:, [5]],
            b=theta[:, [6]],
            t=theta[:, [7]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "lca_no_bias_angle_3":
        x = cssm.lca(
            v=theta[:, :3],
            a=theta[:, [3]],
            z=np.column_stack([theta[:, [4]], theta[:, [4]], theta[:, [4]]]),
            g=theta[:, [5]],
            b=theta[:, [6]],
            t=theta[:, [7]],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 8]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    # 4 Choice models
    if no_noise:
        s = np.tile(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), (n_trials, 1))
    else:
        s = np.tile(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), (n_trials, 1))

    if model == "race_4":
        x = cssm.race_model(
            v=theta[:, :4],
            a=theta[:, [4]],
            z=theta[:, 5:9],
            t=theta[:, [9]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "race_no_bias_4":
        x = cssm.race_model(
            v=theta[:, :4],
            a=theta[:, [4]],
            z=np.column_stack(
                [theta[:, [5]], theta[:, [5]], theta[:, [5]], theta[:, [5]]]
            ),
            t=theta[:, [6]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "race_no_bias_angle_4":
        x = cssm.race_model(
            v=theta[:, :4],
            a=theta[:, [4]],
            z=np.column_stack(
                [theta[:, [5]], theta[:, [5]], theta[:, [5]], theta[:, [5]]]
            ),
            t=theta[:, [6]],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 7]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "lca_4":
        x = cssm.lca(
            v=theta[:, :4],
            a=theta[:, [4]],
            z=theta[:, 5:9],
            g=theta[:, [9]],
            b=theta[:, [10]],
            t=theta[:, [11]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "lca_no_bias_4":
        x = cssm.lca(
            v=theta[:, :4],
            a=theta[:, [4]],
            z=np.column_stack(
                [theta[:, [5]], theta[:, [5]], theta[:, [5]], theta[:, [5]]]
            ),
            g=theta[:, [6]],
            b=theta[:, [7]],
            t=theta[:, [8]],
            s=s,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    if model == "lca_no_bias_angle_4":
        x = cssm.lca(
            v=theta[:, :4],
            a=theta[:, [4]],
            z=np.column_stack(
                [theta[:, [5]], theta[:, [5]], theta[:, [5]], theta[:, [5]]]
            ),
            g=theta[:, [6]],
            b=theta[:, [7]],
            t=theta[:, [8]],
            s=s,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 9]},
            delta_t=delta_t,
            n_samples=n_samples,
            n_trials=n_trials,
            max_t=max_t,
            random_state=random_state,
        )

    # Seq / Parallel models (4 choice)
    if no_noise:
        s = 0.0
    else:
        s = 1.0

    z_vec = np.tile(np.array([0.5], dtype=np.float32), reps=n_trials)
    a_zero_vec = np.tile(np.array([0.0], dtype=np.float32), reps=n_trials)

    if model == "ddm_seq2":
        x = cssm.ddm_flexbound_seq2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=theta[:, 4],
            z_l_1=theta[:, 5],
            z_l_2=theta[:, 6],
            t=theta[:, 7],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
        )

    if model == "ddm_seq2_no_bias":
        x = cssm.ddm_flexbound_seq2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            t=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
        )

    if model == "ddm_seq2_conflict_gamma_no_bias":
        x = cssm.ddm_flexbound_seq2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=a_zero_vec,
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            t=theta[:, 3],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.conflict_gamma_bound,
            boundary_multiplicative=False,
            boundary_params={
                "a": theta[:, 4],
                "theta": theta[:, 5],
                "scale": theta[:, 6],
                "alpha_gamma": theta[:, 7],
                "scale_gamma": theta[:, 8],
            },
            random_state=random_state,
        )

    if model == "ddm_seq2_angle_no_bias":
        x = cssm.ddm_flexbound_seq2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            t=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 5]},
            random_state=random_state,
        )

    if model == "ddm_seq2_weibull_no_bias":
        x = cssm.ddm_flexbound_seq2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            t=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.weibull_cdf,
            boundary_multiplicative=True,
            boundary_params={"alpha": theta[:, 5], "beta": theta[:, 6]},
            random_state=random_state,
        )

    if model == "ddm_par2":
        x = cssm.ddm_flexbound_par2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=theta[:, 4],
            z_l_1=theta[:, 5],
            z_l_2=theta[:, 6],
            t=theta[:, 7],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
        )

    if model == "ddm_par2_no_bias":
        x = cssm.ddm_flexbound_par2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            t=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
        )

    if model == "ddm_par2_conflict_gamma_no_bias":
        x = cssm.ddm_flexbound_par2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=a_zero_vec,
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            t=theta[:, 3],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.conflict_gamma_bound,
            boundary_multiplicative=False,
            boundary_params={
                "a": theta[:, 4],
                "theta": theta[:, 5],
                "scale": theta[:, 6],
                "alpha_gamma": theta[:, 7],
                "scale_gamma": theta[:, 8],
            },
            random_state=random_state,
        )

    if model == "ddm_par2_angle_no_bias":
        x = cssm.ddm_flexbound_par2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            t=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 5]},
            random_state=random_state,
        )

    if model == "ddm_par2_weibull_no_bias":
        x = cssm.ddm_flexbound_par2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            t=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.weibull_cdf,
            boundary_multiplicative=True,
            boundary_params={"alpha": theta[:, 5], "beta": theta[:, 6]},
            random_state=random_state,
        )

    if model == "ddm_mic2_adj":
        x = cssm.ddm_flexbound_mic2(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=theta[:, 4],
            z_l_1=theta[:, 5],
            z_l_2=theta[:, 6],
            d=theta[:, 7],
            t=theta[:, 8],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
        )

    if model == "ddm_mic2_adj_no_bias":
        x = cssm.ddm_flexbound_mic2_adj(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            t=theta[:, 5],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
        )

    if model == "ddm_mic2_adj_conflict_gamma_no_bias":
        x = cssm.ddm_flexbound_mic2_adj(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=a_zero_vec,
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 3],
            t=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.conflict_gamma_bound,
            boundary_multiplicative=False,
            boundary_params={
                "a": theta[:, 5],
                "theta": theta[:, 6],
                "scale": theta[:, 7],
                "alpha_gamma": theta[:, 8],
                "scale_gamma": theta[:, 9],
            },
            random_state=random_state,
        )

    if model == "ddm_mic2_adj_angle_no_bias":
        x = cssm.ddm_flexbound_mic2_adj(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            t=theta[:, 5],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 6]},
            random_state=random_state,
        )

    if model == "ddm_mic2_adj_weibull_no_bias":
        x = cssm.ddm_flexbound_mic2_adj(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            t=theta[:, 5],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.weibull_cdf,
            boundary_multiplicative=True,
            boundary_params={"alpha": theta[:, 6], "beta": theta[:, 7]},
            random_state=random_state,
        )

    if model == "tradeoff_no_bias":
        x = cssm.ddm_flexbound_tradeoff(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            t=theta[:, 5],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.constant,
            boundary_multiplicative=True,
            boundary_params={},
            random_state=random_state,
        )

    if model == "tradeoff_angle_no_bias":
        x = cssm.ddm_flexbound_tradeoff(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            t=theta[:, 5],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.angle,
            boundary_multiplicative=False,
            boundary_params={"theta": theta[:, 6]},
            random_state=random_state,
        )

    if model == "tradeoff_weibull_no_bias":
        x = cssm.ddm_flexbound_tradeoff(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=theta[:, 3],
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 4],
            t=theta[:, 5],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.weibull_cdf,
            boundary_multiplicative=True,
            boundary_params={"alpha": theta[:, 6], "beta": theta[:, 7]},
            random_state=random_state,
        )

    if model == "tradeoff_conflict_gamma_no_bias":
        x = cssm.ddm_flexbound_tradeoff(
            v_h=theta[:, 0],
            v_l_1=theta[:, 1],
            v_l_2=theta[:, 2],
            a=a_zero_vec,
            z_h=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_1=z_vec,  # np.array([0.5], dtype = np.float32),
            z_l_2=z_vec,  # np.array([0.5], dtype = np.float32),
            d=theta[:, 3],
            t=theta[:, 4],
            s=s,
            n_samples=n_samples,
            n_trials=n_trials,
            delta_t=delta_t,
            max_t=max_t,
            boundary_fun=bf.conflict_gamma_bound,
            boundary_multiplicative=False,
            boundary_params={
                "a": theta[:, 5],
                "theta": theta[:, 6],
                "scale": theta[:, 7],
                "alpha_gamma": theta[:, 8],
                "scale_gamma": theta[:, 9],
            },
            random_state=random_state,
        )

    # Output compatibility
    if n_trials == 1:
        x["rts"] = np.squeeze(x["rts"], axis=1)
        x["choices"] = np.squeeze(x["choices"], axis=1)
        # x = (np.squeeze(x['rts'], axis = 1), np.squeeze(x['choices'], axis = 1), x['metadata'])
    if n_trials > 1 and n_samples == 1:
        x["rts"] = np.squeeze(x["rts"], axis=0)
        x["choices"] = np.squeeze(x["choices"], axis=0)
        # x = (np.squeeze(x['rts'], axis = 0), np.squeeze(x['choices'], axis = 0), x['metadata'])

    x["metadata"]["model"] = model

    if bin_dim == 0 or bin_dim == None:
        return x
    elif bin_dim > 0 and n_trials == 1 and not bin_pointwise:
        # print(x)
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
        return "currently n_trials > 1 and n_samples > 1 will not work together with bin_pointwise"
    elif bin_dim > 0 and n_trials > 1 and not bin_pointwise:
        return "currently binned outputs not implemented for multi-trial simulators"
    elif bin_dim == -1:
        return "invalid bin_dim"
