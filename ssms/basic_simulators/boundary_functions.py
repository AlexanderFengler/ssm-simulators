# External
from scipy.stats import gamma
import numpy as np

# Collection of boundary functions

"""
Module defines a collection of boundary functions for the simulators in the package.
"""


# Constant: (multiplicative)
def constant(t=0):
    """constant boundary function

    Arguments
    ---------
        t (int, optional): _description_. Defaults to 0.

    Returns
    -------
        np.array: Array of boundary values, same length as t
    """
    return 1


# Angle (additive)
def angle(t=1, theta=1):
    """angle boundary function

    Arguments
    ---------
        t (int, optional): _description_. Defaults to 1.
        theta (int, optional): _description_. Defaults to 1.

    Returns
    -------
        np.array: Array of boundary values, same length as t
    """
    return np.multiply(t, (-np.sin(theta) / np.cos(theta)))


# Generalized logistic bound (additive)
def generalized_logistic_bnd(t=1, B=2.0, M=3.0, v=0.5):
    """generalized logistic bound

    Arguments
    ---------
        t (int, optional): Defaults to 1.
        B (float, optional): Defaults to 2.0.
        M (float, optional): Defaults to 3.0.
        v (float, optional): Defaults to 0.5.

    Returns
    -------
        np.array: Array of boundary values, same length as t
    """
    return 1 - (1 / np.power(1 + np.exp(-B * (t - M)), 1 / v))


# Weibull survival fun (multiplicative)
def weibull_cdf(t=1, alpha=1, beta=1):
    """boundary based on weibull survival function.

    Arguments
    ---------
        t (int, optional): Defaults to 1.
        alpha (int, optional): Defaults to 1.
        beta (int, optional): Defaults to 1.

    Returns
    -------
        np.array: Array of boundary values, same length as t
    """
    return np.exp(-np.power(np.divide(t, beta), alpha))


def conflict_gamma_bound(
    t=np.arange(0, 20, 0.1),
    a=0.5,
    theta=0.5,
    scale=1,
    alpha_gamma=1.01,
    scale_gamma=0.3,
):
    """conflict bound that allows initial divergence then collapse

    Arguments
    ---------
        t: np.array or float <default = 1>
            Time/s (with arbitrary measure, but in HDDM it is used as seconds),
            at which to evaluate the bound.
        theta: float <default = 0.5>
            Collapse angle
        scale: float <default = 1.0>
            Scaling the gamma distribution of the boundary
            (since bound does not have to integrate to one)
        a: float <default = 0.5>
            Initial boundary separation
        alpha_gamma: float <default = 1.01>
            alpha parameter for a gamma in scale shape parameterization
        scale_gamma: float <default = 0.3>
            scale parameter for a gamma in scale shape paraemterization
    Returns
    -------
        np.array: Array of boundary values, same length as t

    """

    return np.maximum(
        a
        + scale * gamma.pdf(t, a=alpha_gamma, loc=0, scale=scale_gamma)
        + np.multiply(t, (-np.sin(theta) / np.cos(theta))),
        0,
    )
