# External
import scipy as scp
from scipy.stats import gamma
import numpy as np

def constant(t = np.arange(0, 20, 0.1)):
    return np.zeros(t.shape[0])

def gamma_drift(t = np.arange(0, 20, 0.1),
                shape = 2,
                scale = 0.01,
                c = 1.5):
    """Drift function that follows a scaled gamma distribution

    :Arguments:
        t: np.ndarray <default=np.arange(0, 20, 0.1)>
            Timepoints at which to evaluate the drift. Usually np.arange() of some sort. 
        shape: float <default=2>
            Shape parameter of the gamma distribution
        scale: float <default=0.01>
            Scale parameter of the gamma distribution
        c: float <default=1.5> 
            Scalar parameter that scales the peak of the gamma distribution 
            (Note this function follows a gamma distribution but does not integrate to 1)

    :Return: np.ndarray
         The gamma drift evaluated at the supplied timepoints t.

    """

    num_ = np.power(t, shape - 1) * np.exp(np.divide(-t, scale))
    div_ = np.power(shape - 1, shape - 1) * np.power(scale, shape - 1) * np.exp(- (shape - 1))
    return c * np.divide(num_, div_)

    """Basic data simulator for the models included in HDDM. 


    :Arguments:
        theta : list or numpy.array
            Parameters of the simulator. If 2d array, each row is treated as a 'trial' 
            and the function runs n_sample * n_trials simulations.
        model: str <default='angle'>
            Determines the model that will be simulated.
        n_samples: int <default=1000>
            Number of simulation runs (for each trial if supplied n_trials > 1)
        n_trials: int <default=1>
            Number of trials in a simulations run (this specifically addresses trial by trial parameterizations)
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
    
    :Return: tuple 
        can be (rts, responses, metadata)
        or     (rt-response histogram, metadata)
        or     (rts binned pointwise, responses, metadata)

    """