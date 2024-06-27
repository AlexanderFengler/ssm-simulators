# External
import numpy as np
from scipy.stats import norm

"""
This module defines a collection of drift functions for the simulators in the package.
"""


def constant(t=np.arange(0, 20, 0.1)):
    """constant drift function

    Arguments
    ---------
        t (_type_, optional): _description_. Defaults to np.arange(0, 20, 0.1).

    Returns
    -------
        np.array: Array of drift values, same length as t

    """
    return np.zeros(t.shape[0])


def gamma_drift(t=np.arange(0, 20, 0.1), shape=2, scale=0.01, c=1.5):
    """Drift function that follows a scaled gamma distribution

    Arguments
    ---------
        t: np.ndarray <default=np.arange(0, 20, 0.1)>
            Timepoints at which to evaluate the drift.
            Usually np.arange() of some sort.
        shape: float <default=2>
            Shape parameter of the gamma distribution
        scale: float <default=0.01>
            Scale parameter of the gamma distribution
        c: float <default=1.5>
            Scalar parameter that scales the peak of
            the gamma distribution.
            (Note this function follows a gamma distribution
            but does not integrate to 1)

    Return
    ------
        np.ndarray
            The gamma drift evaluated at the supplied timepoints t.

    """

    num_ = np.power(t, shape - 1) * np.exp(np.divide(-t, scale))
    div_ = (
        np.power(shape - 1, shape - 1)
        * np.power(scale, shape - 1)
        * np.exp(-(shape - 1))
    )
    return c * np.divide(num_, div_)


def ds_support_analytic(t=np.arange(0, 10, 0.001), init_p=0, fix_point=1, slope=2):
    """Solution to differential equation of the form:
       x' = slope*(fix_point - x),
       with initial condition init_p.
       The solution takes the form:
       (init_p - fix_point) * exp(-slope * t) + fix_point

    Arguments
    ---------
        t: np.ndarray <default=np.arange(0, 20, 0.1)>
            Timepoints at which to evaluate the drift. Usually np.arange() of some sort.
        init_p: float <default=0>
            Initial condition of dynamical system
        fix_point: float <default=1>
            Fixed point of dynamical system
        slope: float <default=0.01>
            Coefficient in exponent of the solution.
    Return
    ------
    np.ndarray
         The gamma drift evaluated at the supplied timepoints t.

    """

    return (init_p - fix_point) * np.exp(-(slope * t)) + fix_point


def ds_conflict_drift(
    t=np.arange(0, 10, 0.001),
    tinit=0,
    dinit=0,
    tslope=1,
    dslope=1,
    tfixedp=1,
    tcoh=1.5,
    dcoh=1.5,
):
    """This drift is inspired by a conflict task which
       involves a target and a distractor stimuli both presented
       simultaneously.
       Two drift timecourses are linearly combined weighted
       by the coherence in the respective target and distractor stimuli.
       Each timecourse follows a dynamical system as described
       in the ds_support_analytic() function.

    Arguments
    ---------
        t: np.ndarray <default=np.arange(0, 20, 0.1)>
            Timepoints at which to evaluate the drift.
            Usually np.arange() of some sort.
        init_p_t: float <default=0>
            Initial condition of target drift timecourse
        init_p_d: float <default=0>
            Initial condition of distractor drift timecourse
        slope_t: float <default=1>
            Slope parameter for target drift timecourse
        slope_d: float <default=1>
            Slope parameter for distractor drift timecourse
        fixed_p_t: float <default=1>
            Fixed point for target drift timecourse
        coherence_t: float <default=1.0>
            Coefficient for the target drift timecourse
        coherence_d: float <default=-1.0>
            Coefficient for the distractor drift timecourse
    Return
    ------
    np.ndarray
         The full drift timecourse evaluated at the supplied timepoints t.
    """

    w_t = ds_support_analytic(t=t, init_p=tinit, fix_point=tfixedp, slope=tslope)

    w_d = ds_support_analytic(t=t, init_p=dinit, fix_point=0, slope=dslope)

    v_t = (w_t * tcoh) + (w_d * dcoh)

    return v_t  # , w_t, w_d


def attend_drift(
    t=np.arange(0, 20, 0.1),
    p_outer=-0.3,
    p_inner=-0.3,
    p_target=0.3,
    r=0.5,
    sda=2,
    alpha=1,
):  # add a scaling factor
    """Drift function for shrinking spotlight model, which involves a time varying
    function dependent on a linearly decreasing standard deviation of attention.

    Arguments
    --------
        t: np.ndarray <default=np.arange(0, 20, 0.1)>
            Timepoints at which to evaluate the drift.
            Usually np.arange() of some sort.
        p_outer: float <default=-0.3>
            perceptual input for outer flankers
        p_inner: float <default=-0.3>
            perceptual input for inner flankers
        p_target: float <default=0.3>
            perceptual input for target flanker
        r: float <default=0.5>
            rate parameter for sda decrease
        sda: float <default=2>
            width of attentional spotlight
        alpha: float <default=1>
            scaling factor of overall drift rate
    Return
    ------
    np.ndarray
        Drift evaluated at timepoints t
    """

    new_sda = sda - r * t  # make sure that the sda doesn't go below 0

    a_outer = norm.sf(1.5, loc=0, scale=new_sda)
    a_inner = norm.cdf(1.5, loc=0, scale=new_sda) - norm.cdf(0.5, loc=0, scale=new_sda)
    a_target = norm.cdf(0.5, loc=0, scale=new_sda) - norm.cdf(
        -0.5, loc=0, scale=new_sda
    )

    v_t = alpha * (2 * p_outer * a_outer + 2 * p_inner * a_inner + p_target * a_target)

    return v_t
