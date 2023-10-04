from ssms.basic_simulators import boundary_functions as bf
from ssms.basic_simulators import drift_functions as df
import numpy as np

# I need a generic docstring here

"""
    Configuration dictionary for simulators

    Variables:
    ---------
    model_config: dict
        Dictionary containing all the information about the models

    kde_simulation_filters: dict
        Dictionary containing the filters for the KDE simulations

    data_generator_config: dict
        Dictionary containing information for data generator settings.
        Supposed to serve as a starting point and example, which the user then
        modifies to their needs.
"""

# Configuration dictionary for simulators
model_config = {
    "ddm": {
        "name": "ddm",
        "params": ["v", "a", "z", "t"],
        "param_bounds": [[-3.0, 0.3, 0.1, 0.0], [3.0, 2.5, 0.9, 2.0]],
        "boundary": bf.constant,
        "n_params": 4,
        "default_params": [0.0, 1.0, 0.5, 1e-3],
        "hddm_include": ["z"],
        "nchoices": 2,
    },
    "ddm_legacy": {
        "name": "ddm_legacy",
        "params": ["v", "a", "z", "t"],
        "param_bounds": [[-3.0, 0.3, 0.1, 0.0], [3.0, 2.5, 0.9, 2.0]],
        "boundary": bf.constant,
        "n_params": 4,
        "default_params": [0.0, 1.0, 0.5, 1e-3],
        "hddm_include": ["z"],
        "nchoices": 2,
    },
    "ddm_deadline": {
        "name": "ddm_deadline",
        "params": ["v", "a", "z", "t", "deadline"],
        "param_bounds": [[-3.0, 0.3, 0.1, 0.0, 0.1], [3.0, 2.5, 0.9, 2.0, 5.0]],
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 1e-3, 5.0],
        "hddm_include": ["z", "deadline"],
        "nchoices": 2,
    },
    "angle": {
        "name": "angle",
        "params": ["v", "a", "z", "t", "theta"],
        "param_bounds": [[-3.0, 0.3, 0.1, 1e-3, -0.1], [3.0, 3.0, 0.9, 2.0, 1.3]],
        "boundary": bf.angle,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 1e-3, 0.0],
        "hddm_include": ["z", "theta"],
        "nchoices": 2,
    },
    "weibull": {
        "name": "weibull",
        "params": ["v", "a", "z", "t", "alpha", "beta"],
        "param_bounds": [
            [-2.5, 0.3, 0.2, 1e-3, 0.31, 0.31],
            [2.5, 2.5, 0.8, 2.0, 4.99, 6.99],
        ],
        "boundary": bf.weibull_cdf,
        "n_params": 6,
        "default_params": [0.0, 1.0, 0.5, 1e-3, 3.0, 3.0],
        "hddm_include": ["z", "alpha", "beta"],
        "nchoices": 2,
    },
    "levy": {
        "name": "levy",
        "params": ["v", "a", "z", "alpha", "t"],
        "param_bounds": [[-3.0, 0.3, 0.1, 1.0, 1e-3], [3.0, 3.0, 0.9, 2.0, 2]],
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 1.5, 0.1],
        "hddm_include": ["z", "alpha"],
        "nchoices": 2,
    },
    "levy_angle": {
        "name": "levy_angle",
        "params": ["v", "a", "z", "alpha", "t", "theta"],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1.0, 1e-3, -0.1],
            [3.0, 3.0, 0.9, 2.0, 2, 1.3],
        ],
        "boundary": bf.angle,
        "n_params": 6,
        "default_params": [0.0, 1.0, 0.5, 1.5, 0.1, 0.01],
        "hddm_include": ["z", "alpha", "theta"],
        "nchoices": 2,
    },
    "full_ddm": {
        "name": "full_ddm",
        "params": ["v", "a", "z", "t", "sz", "sv", "st"],
        "param_bounds": [
            [-3.0, 0.3, 0.3, 0.25, 1e-3, 1e-3, 1e-3],
            [3.0, 2.5, 0.7, 2.25, 0.2, 2.0, 0.25],
        ],
        "boundary": bf.constant,
        "n_params": 7,
        "default_params": [0.0, 1.0, 0.5, 0.25, 1e-3, 1e-3, 1e-3],
        "hddm_include": ["z", "st", "sv", "sz"],
        "nchoices": 2,
    },
    "gamma_drift": {
        "name": "gamma_drift",
        "params": ["v", "a", "z", "t", "shape", "scale", "c"],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1e-3, 2.0, 0.01, -3.0],
            [3.0, 3.0, 0.9, 2.0, 10.0, 1.0, 3.0],
        ],
        "boundary": bf.constant,
        "drift_fun": df.gamma_drift,
        "n_params": 7,
        "default_params": [0.0, 1.0, 0.5, 0.25, 5.0, 0.5, 1.0],
        "hddm_include": ["z", "t", "shape", "scale", "c"],
        "nchoices": 2,
    },
    "gamma_drift_angle": {
        "name": "gamma_drift_angle",
        "params": ["v", "a", "z", "t", "theta", "shape", "scale", "c"],
        "param_bounds": [
            [-3.0, 0.3, 0.1, 1e-3, -0.1, 2.0, 0.01, -3.0],
            [3.0, 3.0, 0.9, 2.0, 1.3, 10.0, 1.0, 3.0],
        ],
        "boundary": bf.angle,
        "drift_fun": df.gamma_drift,
        "n_params": 7,
        "default_params": [0.0, 1.0, 0.5, 0.25, 0.0, 5.0, 0.5, 1.0],
        "hddm_include": ["z", "t", "theta", "shape", "scale", "c"],
        "nchoices": 2,
    },
    "ds_conflict_drift": {
        "name": "ds_conflict_drift",
        "params": [
            "a",
            "z",
            "t",
            "tinit",
            "dinit",
            "tslope",
            "dslope",
            "tfixedp",
            "tcoh",
            "dcoh",
        ],
        "param_bounds": [
            [0.3, 0.1, 1e-3, 0.0, 0.0, 0.01, 0.01, 0.0, -1.0, -1.0],
            [3.0, 0.9, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0],
        ],
        "boundary": bf.constant,
        "drift_fun": df.ds_conflict_drift,
        "n_params": 10,
        "default_params": [2.0, 0.5, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 0.5, -0.5],
        "hddm_include": [
            "a",
            "z",
            "t",
            "tinit",
            "dinit",
            "tslope",
            "dslope",
            "tfixedp",
            "tcoh",
            "dcoh",
        ],
        "nchoices": 2,
    },
    "ds_conflict_drift_angle": {
        "name": "ds_conflict_drift_angle",
        "params": [
            "a",
            "z",
            "t",
            "tinit",
            "dinit",
            "tslope",
            "dslope",
            "tfixedp",
            "tcoh",
            "dcoh",
            "theta",
        ],
        "param_bounds": [
            [0.3, 0.1, 1e-3, 0.0, 0.0, 0.01, 0.01, 0.0, -1.0, -1.0, 0.0],
            [3.0, 0.9, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.3],
        ],
        "boundary": bf.angle,
        "drift_fun": df.ds_conflict_drift,
        "n_params": 10,
        "default_params": [2.0, 0.5, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 0.5, -0.5, 0.0],
        "hddm_include": [
            "a",
            "z",
            "t",
            "tinit",
            "dinit",
            "tslope",
            "dslope",
            "tfixedp",
            "tcoh",
            "dcoh",
            "theta",
        ],
        "nchoices": 2,
    },
    "ornstein": {
        "name": "ornstein",
        "params": ["v", "a", "z", "g", "t"],
        "param_bounds": [[-2.0, 0.3, 0.1, -1.0, 1e-3], [2.0, 3.0, 0.9, 1.0, 2]],
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 0.0, 1e-3],
        "hddm_include": ["z", "g"],
        "nchoices": 2,
    },
    "ornstein_angle": {
        "name": "ornstein_angle",
        "params": ["v", "a", "z", "g", "t", "theta"],
        "param_bounds": [
            [-2.0, 0.3, 0.1, -1.0, 1e-3, -0.1],
            [2.0, 3.0, 0.9, 1.0, 2, 1.3],
        ],
        "boundary": bf.angle,
        "n_params": 6,
        "default_params": [0.0, 1.0, 0.5, 0.0, 1e-3, 0.1],
        "hddm_include": ["z", "g", "theta"],
        "nchoices": 2,
    },
    "ddm_sdv": {
        "name": "ddm_sdv",
        "params": ["v", "a", "z", "t", "sv"],
        "param_bounds": [[-3.0, 0.3, 0.1, 1e-3, 1e-3], [3.0, 2.5, 0.9, 2.0, 2.5]],
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 1e-3, 1e-3],
        "hddm_include": ["z", "sv"],
        "nchoices": 2,
    },
    "race_3": {
        "name": "race_3",
        "params": ["v0", "v1", "v2", "a", "z0", "z1", "z2", "ndt"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [2.5, 2.5, 2.5, 3.0, 0.9, 0.9, 0.9, 2.0],
        ],
        "boundary": bf.constant,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.5, 1e-3],
        "hddm_include": ["v0", "v1", "v2", "a", "z0", "z1", "z2", "ndt"],
        "nchoices": 3,
    },
    "race_no_bias_3": {
        "name": "race_no_bias_3",
        "params": ["v0", "v1", "v2", "a", "z", "ndt"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [2.5, 2.5, 2.5, 3.0, 0.9, 2.0],
        ],
        "boundary": bf.constant,
        "n_params": 6,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 1e-3],
        "hddm_include": ["v0", "v1", "v2", "a", "z", "ndt"],
        "nchoices": 3,
    },
    "race_no_bias_angle_3": {
        "name": "race_no_bias_angle_3",
        "params": ["v0", "v1", "v2", "a", "z", "ndt", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.1],
            [2.5, 2.5, 2.5, 3.0, 0.9, 2.0, 1.45],
        ],
        "boundary": bf.angle,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 1e-3, 0.0],
        "hddm_include": ["v0", "v1", "v2", "a", "z", "ndt", "theta"],
        "nchoices": 3,
    },
    "race_4": {
        "name": "race_4",
        "params": ["v0", "v1", "v2", "v3", "a", "z0", "z1", "z2", "z3", "ndt"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 0.9, 0.9, 0.9, 2.0],
        ],
        "boundary": bf.constant,
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.5, 0.5, 1e-3],
        "hddm_include": ["v0", "v1", "v2", "a", "z0", "z1", "z2", "ndt"],
        "nchoices": 4,
    },
    "race_no_bias_4": {
        "name": "race_no_bias_4",
        "params": ["v0", "v1", "v2", "v3", "a", "z", "ndt"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 2.0],
        ],
        "boundary": bf.constant,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 1e-3],
        "hddm_include": ["v0", "v1", "v2", "v3", "a", "z", "ndt"],
        "nchoices": 4,
    },
    "race_no_bias_angle_4": {
        "name": "race_no_bias_angle_4",
        "params": ["v0", "v1", "v2", "v3", "a", "z", "ndt", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.1],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 2.0, 1.45],
        ],
        "boundary": bf.angle,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 1e-3, 0.0],
        "hddm_include": ["v0", "v1", "v2", "v3", "a", "z", "ndt", "theta"],
        "nchoices": 4,
    },
    "lca_3": {
        "name": "lca_3",
        "params": ["v0", "v1", "v2", "a", "z0", "z1", "z2", "g", "b", "ndt"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0],
            [2.5, 2.5, 2.5, 3.0, 0.9, 0.9, 0.9, 1.0, 1.0, 2.0],
        ],
        "boundary": bf.constant,
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 1e-3],
        "hddm_include": ["v0", "v1", "v2", "a", "z0", "z1", "z2", "g", "b", "ndt"],
        "nchoices": 3,
    },
    "lca_no_bias_3": {
        "name": "lca_no_bias_3",
        "params": ["v0", "v1", "v2", "a", "z", "g", "b", "ndt"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0],
            [2.5, 2.5, 2.5, 3.0, 0.9, 1.0, 1.0, 2.0],
        ],
        "boundary": bf.constant,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1e-3],
        "hddm_include": ["v0", "v1", "v2", "a", "z", "g", "b", "ndt"],
        "nchoices": 3,
    },
    "lca_no_bias_angle_3": {
        "name": "lca_no_bias_angle_3",
        "params": ["v0", "v1", "v2", "a", "z", "g", "b", "ndt", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0, -1.0],
            [2.5, 2.5, 2.5, 3.0, 0.9, 1.0, 1.0, 2.0, 1.45],
        ],
        "boundary": bf.angle,
        "n_params": 9,
        "default_params": [0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1e-3, 0.0],
        "hddm_include": ["v0", "v1", "v2", "a", "z", "g", "b", "ndt", "theta"],
        "nchoices": 3,
    },
    "lca_4": {
        "name": "lca_4",
        "params": [
            "v0",
            "v1",
            "v2",
            "v3",
            "a",
            "z0",
            "z1",
            "z2",
            "z3",
            "g",
            "b",
            "ndt",
        ],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 2.0],
        ],
        "boundary": bf.constant,
        "n_params": 12,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 1e-3],
        "hddm_include": [
            "v0",
            "v1",
            "v2",
            "v3",
            "a",
            "z0",
            "z1",
            "z2",
            "z3",
            "g",
            "b",
            "ndt",
        ],
        "nchoices": 4,
    },
    "lca_no_bias_4": {
        "name": "lca_no_bias_4",
        "params": ["v0", "v1", "v2", "v3", "a", "z", "g", "b", "ndt"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 1.0, 1.0, 2.0],
        ],
        "boundary": bf.constant,
        "n_params": 9,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1e-3],
        "hddm_include": ["v0", "v1", "v2", "v3", "a", "z", "g", "b", "ndt"],
        "nchoices": 4,
    },
    "lca_no_bias_angle_4": {
        "name": "lca_no_bias_angle_4",
        "params": ["v0", "v1", "v2", "v3", "a", "z", "g", "b", "ndt", "theta"],
        "param_bounds": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0, -0.1],
            [2.5, 2.5, 2.5, 2.5, 3.0, 0.9, 1.0, 1.0, 2.0, 1.45],
        ],
        "boundary": bf.angle,
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1e-3, 0.0],
        "hddm_include": ["v0", "v1", "v2", "v3", "a", "z", "g", "b", "ndt", "theta"],
        "nchoices": 4,
    },
    "ddm_par2": {
        "name": "ddm_par2",
        "params": ["vh", "vl1", "vl2", "a", "zh", "zl1", "zl2", "t"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.2, 0.2, 0.2, 0.0],
            [4.0, 4.0, 4.0, 2.5, 0.8, 0.8, 0.8, 2.0],
        ],
        "boundary": bf.constant,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 1.0],
        "hddm_include": ["vh", "vl1", "vl2", "a", "zh", "zl1", "zl2", "t"],
        "nchoices": 4,
    },
    "ddm_par2_no_bias": {
        "name": "ddm_par2_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "t"],
        "param_bounds": [[-4.0, -4.0, -4.0, 0.3, 0.0], [4.0, 4.0, 4.0, 2.5, 2.0]],
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 0.0, 0.0, 1.0, 1.0],
        "hddm_include": ["vh", "vl1", "vl2", "a", "t"],
        "nchoices": 4,
    },
    "ddm_par2_conflict_gamma_no_bias": {
        "name": "ddm_par2_conflict_gamma_no_bias",
        "params": [
            "vh",
            "vl1",
            "vl2",
            "t",
            "a",
            "theta",
            "scale",
            "alpha_gamma",
            "scale_gamma",
        ],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.0, 0.3, 0.0, 0.0, 1.1, 0.5],
            [4.0, 4.0, 4.0, 2.0, 2.5, 0.5, 5.0, 5.0, 5.0],
        ],
        "boundary": bf.conflict_gamma_bound,
        "n_params": 9,
        "default_params": [0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 1.0, 2, 2],
        "hddm_include": [
            "vh",
            "vl1",
            "vl2",
            "t",
            "a",
            "theta",
            "scale",
            "alpha_gamma",
            "scale_gamma",
        ],
        "nchoices": 4,
    },
    "ddm_par2_angle_no_bias": {
        "name": "ddm_par2_angle_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "t", "theta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, -0.1],
            [4.0, 4.0, 4.0, 2.5, 2.0, 1.0],
        ],
        "boundary": bf.angle,
        "boundary_multiplicative": False,
        "n_params": 6,
        "default_params": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        "hddm_include": ["vh", "vl1", "vl2", "a", "t", "theta"],
        "nchoices": 4,
    },
    "ddm_par2_weibull_no_bias": {
        "name": "ddm_par2_weibull_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "t", "alpha", "beta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.31, 0.31],
            [4.0, 4.0, 4.0, 2.5, 2.0, 4.99, 6.99],
        ],
        "boundary": bf.weibull_cdf,
        "boundary_multiplicative": True,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        "hddm_include": ["vh", "vl1", "vl2", "a", "t", "theta"],
        "nchoices": 4,
    },
    "ddm_seq2": {
        "name": "ddm_seq2",
        "params": ["vh", "vl1", "vl2", "a", "zh", "zl1", "zl2", "t"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.2, 0.2, 0.2, 0.0],
            [4.0, 4.0, 4.0, 2.5, 0.8, 0.8, 0.8, 2.0],
        ],
        "boundary": bf.constant,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 1.0],
        "hddm_include": ["vh", "vl1", "vl2", "a", "zh", "zl1", "zl2", "t"],
        "nchoices": 4,
    },
    "ddm_seq2_no_bias": {
        "name": "ddm_seq2_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "t"],
        "param_bounds": [[-4.0, -4.0, -4.0, 0.3, 0.0], [4.0, 4.0, 4.0, 2.5, 2.0]],
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 0.0, 0.0, 1.0, 1.0],
        "hddm_include": ["vh", "vl1", "vl2", "a", "t"],
        "nchoices": 4,
    },
    "ddm_seq2_conflict_gamma_no_bias": {
        "name": "ddm_seq2_conflict_gamma_no_bias",
        "params": [
            "vh",
            "vl1",
            "vl2",
            "t",
            "a",
            "theta",
            "scale",
            "alpha_gamma",
            "scale_gamma",
        ],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.0, 0.3, 0.0, 0.0, 1.1, 0.5],
            [4.0, 4.0, 4.0, 2.0, 2.5, 0.5, 5.0, 5.0, 5.0],
        ],
        "boundary": bf.conflict_gamma_bound,
        "n_params": 9,
        "default_params": [0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 1.0, 2, 2],
        "hddm_include": [
            "vh",
            "vl1",
            "vl2",
            "t",
            "a",
            "theta",
            "scale",
            "alpha_gamma",
            "scale_gamma",
        ],
        "nchoices": 4,
    },
    "ddm_seq2_angle_no_bias": {
        "name": "ddm_seq2_angle_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "t", "theta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, -0.1],
            [4.0, 4.0, 4.0, 2.5, 2.0, 1.0],
        ],
        "boundary": bf.angle,
        "boundary_multiplicative": False,
        "n_params": 6,
        "default_params": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        "hddm_include": ["vh", "vl1", "vl2", "a", "t", "theta"],
        "nchoices": 4,
    },
    "ddm_seq2_weibull_no_bias": {
        "name": "ddm_seq2_weibull_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "t", "alpha", "beta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.31, 0.31],
            [4.0, 4.0, 4.0, 2.5, 2.0, 4.99, 6.99],
        ],
        "boundary": bf.weibull_cdf,
        "boundary_multiplicative": True,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 1.0, 1.0, 2.5, 3.5],
        "hddm_include": ["vh", "vl1", "vl2", "a", "t", "alpha", "beta"],
        "nchoices": 4,
    },
    "ddm_mic2_adj": {
        "name": "ddm_mic2_adj",
        "params": ["vh", "vl1", "vl2", "a", "zh", "zl1", "zl2", "d", "t"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.2, 0.2, 0.2, 0.0, 0.0],
            [4.0, 4.0, 4.0, 2.5, 0.8, 0.8, 0.8, 1.0, 2.0],
        ],
        "boundary": bf.constant,
        "n_params": 9,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
        "hddm_include": ["vh", "vl1", "vl2", "a", "zh", "zl1", "zl2", "d", "t"],
        "nchoices": 4,
    },
    "ddm_mic2_adj_no_bias": {
        "name": "ddm_mic2_adj_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "d", "t"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.0],
            [4.0, 4.0, 4.0, 2.5, 1.0, 2.0],
        ],
        "boundary": bf.constant,
        "n_params": 6,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 1.0],
        "hddm_include": ["vh", "vl1", "vl2", "a", "zh", "zl1", "zl2", "d", "t"],
        "nchoices": 4,
    },
    "ddm_mic2_adj_conflict_gamma_no_bias": {
        "name": "ddm_mic2_adj_conflict_gamma_no_bias",
        "params": [
            "vh",
            "vl1",
            "vl2",
            "d",
            "t",
            "a",
            "theta",
            "scale",
            "alpha_gamma",
            "scale_gamma",
        ],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.0, 0.0, 0.3, 0.0, 0.0, 1.1, 0.5],
            [4.0, 4.0, 4.0, 1.0, 2.0, 2.5, 0.5, 5.0, 5.0, 5.0],
        ],
        "boundary": bf.conflict_gamma_bound,
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 2, 2],
        "hddm_include": [
            "vh",
            "vl1",
            "vl2",
            "a",
            "d",
            "t",
            "theta",
            "scale",
            "alpha_gamma",
            "scale_gamma",
        ],
        "nchoices": 4,
    },
    "ddm_mic2_adj_angle_no_bias": {
        "name": "ddm_mic2_adj_angle_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "d", "t", "theta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.0, -0.1],
            [4.0, 4.0, 4.0, 2.5, 1.0, 2.0, 1.0],
        ],
        "boundary": bf.angle,
        "boundary_multiplicative": False,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0],
        "hddm_include": ["vh", "vl1", "vl2", "a", "d", "t", "theta"],
        "nchoices": 4,
    },
    "ddm_mic2_adj_weibull_no_bias": {
        "name": "ddm_mic2_adj_weibull_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "d", "t", "alpha", "beta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.0, 0.31, 0.31],
            [4.0, 4.0, 4.0, 2.5, 1.0, 2.0, 4.99, 6.99],
        ],
        "boundary": bf.weibull_cdf,
        "boundary_multiplicative": True,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 2.5, 3.5],
        "hddm_include": ["vh", "vl1", "vl2", "a", "d", "t", "theta"],
        "nchoices": 4,
    },
    "tradeoff_no_bias": {
        "name": "tradeoff_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "d", "t"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.0],
            [4.0, 4.0, 4.0, 2.5, 1.0, 2.0],
        ],
        "boundary": bf.constant,
        "n_params": 6,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 1.0],
        "hddm_include": ["vh", "vl1", "vl2", "a", "d", "t"],
        "nchoices": 4,
    },
    "tradeoff_angle_no_bias": {
        "name": "tradeoff_angle_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "d", "t", "theta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.0, -0.1],
            [4.0, 4.0, 4.0, 2.5, 1.0, 2.0, 1.0],
        ],
        "boundary": bf.angle,
        "boundary_multiplicative": False,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0],
        "hddm_include": ["vh", "vl1", "vl2", "a", "d", "t", "theta"],
        "nchoices": 4,
    },
    "tradeoff_weibull_no_bias": {
        "name": "tradeoff_weibull_no_bias",
        "params": ["vh", "vl1", "vl2", "a", "d", "t", "alpha", "beta"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.3, 0.0, 0.0, 0.31, 0.31],
            [4.0, 4.0, 4.0, 2.5, 1.0, 2.0, 4.99, 6.99],
        ],
        "boundary": bf.weibull_cdf,
        "boundary_multiplicative": True,
        "n_params": 8,
        "default_params": [0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 2.5, 3.5],
        "hddm_include": ["vh", "vl1", "vl2", "a", "d", "t", "theta"],
        "nchoices": 4,
    },
    "tradeoff_conflict_gamma_no_bias": {
        "name": "tradeoff_conflict_gamma_no_bias",
        "params": [
            "vh",
            "vl1",
            "vl2",
            "d",
            "t",
            "a",
            "theta",
            "scale",
            "alphagamma",
            "scalegamma",
        ],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.0, 0.0, 0.3, 0.0, 0.0, 1.1, 0.5],
            [4.0, 4.0, 4.0, 1.0, 2.0, 2.5, 0.5, 5.0, 5.0, 5.0],
        ],
        "boundary": bf.conflict_gamma_bound,
        "boundary_multiplicative": True,
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 2, 2],
        "hddm_include": [
            "vh",
            "vl1",
            "vl2",
            "a",
            "d",
            "t",
            "theta",
            "scale",
            "alphagamma",
            "scalegamma",
        ],
        "nchoices": 4,
    },
    "glob": {
        "name": "glob",
        "params": ["v", "a", "z", "alphar", "g", "t", "theta"],
        "param_bounds": [
            [-3.0, 0.3, 0.15, 1.0, -1.0, 1e-5, 0.0],
            [3.0, 2.0, 0.85, 2.0, 1.0, 2.0, 1.45],
        ],
        "n_params": 7,
        "default_params": [0.0, 1.0, 0.5, 2.0, 0.0, 1.0, 2.5, 3.5],
        "hddm_include": ["z", "alphar", "g", "theta"],
        "nchoices": 2,
        "boundary_multiplicative": False,
        "components": {
            "names": ["g", "alphar", "theta"],
            "off_values": np.float32(np.array([0, 1, 0])),
            "probabilities": np.array([1 / 3, 1 / 3, 1 / 3]),
            "labels": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "n_components": 3,
        },
    },
}

model_config["weibull_cdf"] = model_config["weibull"].copy()
model_config["full_ddm2"] = model_config["full_ddm"].copy()

#### DATASET GENERATOR CONFIGS --------------------------

kde_simulation_filters = {
    "mode": 20,  # != (if mode is max_rt)
    "choice_cnt": 0,  # > (each choice receive at least 10 samples )
    "mean_rt": 17,  # < (mean_rt is smaller than specified value
    "std": 0,  # > (std is positive for each choice)
    "mode_cnt_rel": 0.9,  # < (mode can't be large proportion of all samples)
}

data_generator_config = {
    "cpn_only": {
        "output_folder": "data/cpn_only/",
        "dgp_list": "ddm",  # should be ['ddm'],
        "n_samples": 100000,  # eventually should be {'low': 100000, 'high': 100000},
        "n_parameter_sets": 10000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
    },
    "lan": {
        "output_folder": "data/lan_mlp/",
        "dgp_list": "ddm",  # should be ['ddm'],
        "nbins": 0,
        "n_samples": 100000,  # eventually should be {'low': 100000, 'high': 100000},
        "n_parameter_sets": 10000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "kde_data_mixture_probabilities": [0.8, 0.1, 0.1],
        "simulation_filters": kde_simulation_filters,
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "bin_pointwise": False,
        "separate_response_channels": False,
    },
    "ratio_estimator": {
        "output_folder": "data/ratio/",
        "dgp_list": ["ddm"],
        "nbins": 0,
        "n_samples": {"low": 100000, "high": 100000},
        "n_parameter_sets": 100000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "n_subdatasets": 12,
        "n_trials_per_dataset": 10000,  # EVEN NUMBER ! AF-TODO: Saveguard against odd
        "kde_data_mixture_probabilities": [0.8, 0.1, 0.1],
        "simulation_filters": kde_simulation_filters,
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "bin_pointwise": False,
        "separate_response_channels": False,
    },
    "defective_detector": {
        "output_folder": "data/ratio/",
        "dgp_list": ["ddm"],
        "nbins": 0,
        "n_samples": {"low": 100000, "high": 100000},
        "n_parameter_sets": 100000,
        "n_parameter_sets_rejected": 100,
        "n_training_samples_by_parameter_set": 1000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "n_subdatasets": 12,
        "n_trials_per_dataset": 10000,  # EVEN NUMBER ! AF-TODO: Saveguard against odd
        "kde_data_mixture_probabilities": [0.8, 0.1, 0.1],
        "simulation_filters": kde_simulation_filters,
        "negative_rt_cutoff": -66.77497,
        "n_subruns": 10,
        "bin_pointwise": False,
        "separate_response_channels": False,
    },
    "snpe": {
        "output_folder": "data/snpe_training/",
        "dgp_list": "ddm",  # should be ['ddm'],
        "n_samples": 5000,  # eventually should be {'low': 100000, 'high': 100000},
        "n_parameter_sets": 10000,
        "max_t": 20.0,
        "delta_t": 0.001,
        "pickleprotocol": 4,
        "n_cpus": "all",
        "n_subruns": 10,
        "separate_response_channels": False,
    },
}
##### -----------------------------------------------------
