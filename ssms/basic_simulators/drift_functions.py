# External
import numpy as np

"""
This module defines a collection of drift functions for the simulators in the package.
"""


def constant(t_tmp=np.arange(0, 20, 0.1)):
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
    init_p_t=0,
    init_p_d=0,
    slope_t=1,
    slope_d=1,
    fixed_p_t=1,
    coherence_t=1.5,
    coherence_d=1.5,
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

    w_t = ds_support_analytic(t=t, init_p=init_p_t, fix_point=fixed_p_t, slope=slope_t)

    w_d = ds_support_analytic(t=t, init_p=init_p_d, fix_point=0, slope=slope_d)

    v_t = (w_t * coherence_t) + (w_d * coherence_d)

    return v_t  # , w_t, w_d

def SSP_drift(
    t=np.arange(0, 20, 0.001),
    p=1,          
    sda = 0.1,    
    rd = 0.05,    
    congruency=1   
):
    """This drift is inspired by shrinking spotlight model in conflict task which
       involves a target and a distractor(flanker) stimuli both presented
       simultaneously.

    :param np.ndarray t: drift timecourse, defaults to np.arange(0, 20, 0.001), means max 20 seconds. 
    :param float p: perceptual strength, defaults to 1
    :param float sda: spotlight width at stimulus onset, defaults to 0.1
    :param float rd: shrink rate of variance, defaults to 0.05
    :param int congruency: congruency condition with flanker and target, 1 means congruent, -1 means incongruent. 
    :return np.ndarray: The full drift timecourse evaluated at the supplied timepoints t.
    """
    from scipy.stats import norm

    # calculate current sd of spotlight
    sd_t = sda - (rd * t)      # t like x, sd_t like y, and rd like slope
    # sd_t = np.where(sd_t < 0.001, 0.001, sd_t)
    sd_t = sd_t.clip(0.001, None)
    
    # find area of spotlight over target and flanker
    a_target = norm.cdf(0.5, 0, sd_t) - norm.cdf(-0.5, 0, sd_t)
    a_flanker = 1 - a_target

    # current drift rate
    drift = p * (a_target + (congruency * a_flanker))
    
    return drift

def DMC_drift(
    t=np.arange(0, 20, 0.1), 
    vc=0.3, 
    peak=30, 
    shape=3, 
    tau=100,
    congruency=1,
    ):
    """Drift function for Diffusion model in conflict tasks that follows a gamma function by Evans et al. (2020)

    Arguments
    ---------
        t: np.ndarray <default=np.arange(0, 20, 0.1)>
            Timepoints at which to evaluate the drift.
            Usually np.arange() of some sort.
        vc: float <default=0.2>
            The drift for control process. 
        shape: float <default=2>
            Shape parameter of the gamma distribution
        peak: float <default=1.5>
            Scalar parameter that scales the peak of
            the gamma distribution.
            (Note this function follows a gamma distribution
            but does not integrate to 1)
        tau: float <default=0.01>
            tau is the characteristic time.
        congruency int: congruency condition with flanker and target, 1 means congruent, -1 means 

    Return
    ------
        np.ndarray
            The gamma drift evaluated at the supplied timepoints t.

    """

    t = t.clip(1e-3)
    term1 = peak * congruency * np.exp(-t/tau)
    term2 = (t * np.e) / ((shape-1) * tau)
    term3 = ((shape-1) / t) - (1/tau)
    va = term1 * term2**(shape-1) * term3

    v = va + vc
    return v


def DSTP_drift(
    t=np.arange(0, 20, 0.001), 
    vta=0.3, vfl=1, vss=0.5, vp2=2, 
    ass=1.5, zss = 0.5, 
    congruency = 1, 
    delta_t=0.001, sqrt_st=np.sqrt(0.001)
    ):
    """Drift function for dual-stage two-phase modell in conflict tasks that follows a gamma function by Evans et al. (2020). The code is transformed from Luo, J., Yang, M., & Wang, L. (2022). Learned irrelevant stimulus-response associations and proportion congruency effect: A diffusion model account. Journal of Experimental Psychology: Learning, Memory, and Cognition. https://doi.org/10.1037/xlm0001158

    :param np.ndarray t: drift timecourse, default to np.arange(0, 20, 0.001), means max 20 seconds. 
    :param float vta: drift for target in phase 1
    :param float vfl: drift for flanker in phase 1
    :param float vss: drift for stimulus selection
    :param float ass: boundary for stimulus selection
    :param float zss: start poin for stimulus selection. default to 0.5, means there is no selection bias. 
    :param int congruency: congruency condition with flanker and target, 1 means congruent, -1 means incongruent. 
    :param float delta_t, sqrt_st: the time step and scale factor for diffusion process, respectively. 
    :return np.ndarray: The full drift timecourse evaluated at the supplied timepoints t.
    """

    # initiate the X of start point for stimulus selection
    X_ss = zss * ass
    
    # Drift rate in the first phase
    congruency = np.min([congruency, 0])
    mu = vta + congruency * vfl
    
    t = t.shape[0]
    mu_view = np.full(t, mu)
    for step in range(t):
        # Stimulus selection
        X_ss += vss * delta_t + np.random.normal() * sqrt_st
        # Drift rate in the second phase
        if X_ss >= ass:
            mu_view[step:] = vp2
            break
        elif X_ss <= 0 and congruency == 0:
            mu_view[step:] = vp2
            break
        elif X_ss <= 0 and congruency == -1:
            mu_view[step:] = -(vp2)
            break
    return mu_view