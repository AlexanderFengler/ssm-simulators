# KDE GENERATORS
import numpy as np
from sklearn.neighbors import KernelDensity


"""
    This module contains a class for generating kdes from data.
"""


# Generate class for log_kdes
class LogKDE:
    """
    Class for generating kdes from (rt, choice) data. Works for any number of choices.

    Attributes
    ----------
        simulator_data: (rt, choice, simulator_info)
            tuple as returned by simulator function
        bandwidth_type: string
            type of bandwidth to use, default is 'silverman'
        auto_bandwidth: boolean
            whether to compute bandwidths automatically, default is True

    Methods
    -------
        compute_bandwidths(type='silverman')
            Computes bandwidths for each choice from rt data.
        generate_base_kdes(auto_bandwidth=True, bandwidth_type='silverman')
            Generates kdes from rt data.
        kde_eval(data=([], []), log_eval=True)
            Evaluates kde log likelihood at chosen points.
        kde_sample(n_samples=2000, use_empirical_choice_p=True, alternate_choice_p=0)
            Samples from a given kde.
        attach_data_from_simulator(simulator_data=([0, 2, 4], [-1, 1, -1]))
            Helper function to transform ddm simulator output
            to dataset suitable for the kde function class.

    Returns:
        _type_: _description_
    """

    # Initialize the class
    def __init__(
        self,
        simulator_data,  # as returned by simulator function
        bandwidth_type="silverman",
        auto_bandwidth=True,
    ):
        self.attach_data_from_simulator(simulator_data)
        self.generate_base_kdes(
            auto_bandwidth=auto_bandwidth, bandwidth_type=bandwidth_type
        )
        self.simulator_info = simulator_data[2]

    # Function to compute bandwidth parameters given data-set
    # (At this point using Silverman rule)
    def compute_bandwidths(self, type="silverman"):
        """
        Computes bandwidths for each choice from rt data.

        Arguments:
        ----------
        type: string
            Type of bandwidth to use, default is 'silverman' which follows silverman rule.

        Returns:
        --------
        bandwidths: list
            List of bandwidths for each choice.
        """

        # For now allows only silverman rule
        self.bandwidths = []
        if type == "silverman":
            for i in range(0, len(self.data["choices"]), 1):
                if len(self.data["rts"][i]) == 0:
                    self.bandwidths.append("no_base_data")
                else:
                    bandwidth_tmp = bandwidth_silverman(
                        sample=np.log(self.data["rts"][i])
                    )
                    if bandwidth_tmp > 0:
                        self.bandwidths.append(bandwidth_tmp)
                    else:
                        self.bandwidths.append("no_base_data")

    # Function to generate basic kdes
    # I call the function generate_base_kdes because
    # in the final evaluation computations
    # we adjust the input and output of the kdes
    # appropriately (we do not use them directly)
    def generate_base_kdes(self, auto_bandwidth=True, bandwidth_type="silverman"):
        """
        Generates kdes from rt data. We apply gaussian kernels to the log of the rts.

        Arguments:
        ----------
        auto_bandwidth: boolean
            Whether to compute bandwidths automatically, default is True.
        bandwidth_type: string
            Type of bandwidth to use, default is 'silverman' which follows silverman rule.

        Returns:
        --------
        base_kdes: list
            List of kdes for each choice. (These get attached to the base_kdes attribute of the class, not returned)
        """
        # Compute bandwidth parameters
        if auto_bandwidth:
            self.compute_bandwidths(type=bandwidth_type)

        # Generate the kdes
        self.base_kdes = []
        for i in range(0, len(self.data["choices"]), 1):
            if self.bandwidths[i] == "no_base_data":
                self.base_kdes.append("no_base_data")
            else:
                self.base_kdes.append(
                    KernelDensity(kernel="gaussian", bandwidth=self.bandwidths[i]).fit(
                        np.log(self.data["rts"][i])
                    )
                )

    # Function to evaluate the kde log likelihood at chosen points
    def kde_eval(self, data=([], []), log_eval=True):  # kde
        """
        Evaluates kde log likelihood at chosen points.

        Arguments:
        ----------
        data: tuple
            Tuple of (rts, choices) to evaluate the kde at.
        log_eval: boolean
            Whether to return log likelihood or likelihood, default is True.

        Returns:
        --------
        log_kde_eval: array
            Array of log likelihoods for each (rt, choice) pair.
        """
        # Initializations
        log_rts = np.log(data[0])
        log_kde_eval = np.log(data[0])
        choices = np.unique(data[1])

        # Main loop
        for c in choices:
            # Get data indices where choice == c
            choice_idx_tmp = np.where(data[1] == c)

            # Main step: Evaluate likelihood for rts corresponding
            # to choice == c
            if self.base_kdes[self.data["choices"].index(c)] == "no_base_data":
                log_kde_eval[choice_idx_tmp] = np.log(
                    1 / self.data["n_trials"]
                ) + np.log(
                    1 / self.simulator_info["max_t"]
                )  # -66.77497 # the number corresponds to log(1e-29)
                # --> should rather be log(1 / n) + log(1 / 20)
            else:
                log_kde_eval[choice_idx_tmp] = (
                    np.log(
                        self.data["choice_proportions"][self.data["choices"].index(c)]
                    )
                    + self.base_kdes[self.data["choices"].index(c)].score_samples(
                        np.expand_dims(log_rts[choice_idx_tmp], 1)
                    )
                    - log_rts[choice_idx_tmp]
                )

        if log_eval is True:
            return log_kde_eval
        else:
            return np.exp(log_kde_eval)

    def kde_sample(
        self, n_samples=2000, use_empirical_choice_p=True, alternate_choice_p=0
    ):
        """
        Samples from a given kde.

        Arguments:
        ----------
        n_samples: int
            Number of samples to draw.
        use_empirical_choice_p: boolean
            Whether to use empirical choice proportions, default is True. (Note 'empirical' here,
            refers to the originally attached datasets that served as the basis to generate the choice-wise
            kdes)
        alternate_choice_p: array
            Array of choice proportions to use, default is 0. (Note 'alternate' here refers to 'alternative'
            to the 'empirical' choice proportions)

        """

        # sorting the which list in ascending order
        # this implies that we return the kde_samples array so that the
        # indices reflect 'choice-labels' as provided in 'which' in ascending order
        # kde_samples = []

        rts = np.zeros((n_samples, 1))
        choices = np.zeros((n_samples, 1))

        n_by_choice = []
        for i in range(0, len(self.data["choices"]), 1):
            if use_empirical_choice_p is True:
                n_by_choice.append(
                    round(n_samples * self.data["choice_proportions"][i])
                )
            else:
                n_by_choice.append(round(n_samples * alternate_choice_p[i]))

        # Catch a potential dimension error if we ended up rounding up twice
        if sum(n_by_choice) > n_samples:
            n_by_choice[np.argmax(n_by_choice)] -= 1
        elif sum(n_by_choice) < n_samples:
            n_by_choice[np.argmax(n_by_choice)] += 1
            choices[n_samples - 1, 0] = np.random.choice(self.data["choices"])

        # Get samples
        cnt_low = 0
        for i in range(0, len(self.data["choices"]), 1):
            if n_by_choice[i] > 0:
                cnt_high = cnt_low + n_by_choice[i]

                if self.base_kdes[i] != "no_base_data":
                    rts[cnt_low:cnt_high] = np.exp(
                        self.base_kdes[i].sample(n_samples=n_by_choice[i])
                    )
                else:
                    rts[cnt_low:cnt_high, 0] = np.random.uniform(
                        low=0, high=self.simulator_info["max_t"], size=n_by_choice[i]
                    )

                choices[cnt_low:cnt_high, 0] = np.repeat(
                    self.data["choices"][i], n_by_choice[i]
                )
                cnt_low = cnt_high

        return (rts, choices, self.simulator_info)

    # Helper function to transform ddm simulator output to dataset suitable for
    # the kde function class
    def attach_data_from_simulator(self, simulator_data=([0, 2, 4], [-1, 1, -1])):
        choices = np.unique(simulator_data[2]["possible_choices"])

        n = len(simulator_data[0])
        self.data = {"rts": [], "choices": [], "choice_proportions": []}

        # Loop through the choices made to get proportions and separated out rts
        for c in choices:
            self.data["choices"].append(c)
            rts_tmp = np.expand_dims(simulator_data[0][simulator_data[1] == c], axis=1)
            prop_tmp = len(rts_tmp) / n
            self.data["rts"].append(rts_tmp)
            self.data["choice_proportions"].append(prop_tmp)

        self.data["n_trials"] = simulator_data[0].shape[0]


# Support functions (accessible from outside the main class defined in script)
def bandwidth_silverman(
    sample=[0, 0, 0],
    std_cutoff=1e-3,
    std_proc="restrict",  # options 'kill', 'restrict'
    std_n_1=10,  # HERE WE CAN ALLOW FOR SOMETHING MORE INTELLIGENT
):
    """
    Computes silverman bandwidth for an array of samples (rts in our context, but general).

    Arguments:
    ----------
    sample: array
        Array of samples to compute bandwidth for.
    std_cutoff: float
        Cutoff for std, default is 1e-3.
        (If sample-std is smaller than this, we either kill it or restrict it to this value)
    std_proc: string
        How to deal with small stds, default is 'restrict'. (Options: 'kill', 'restrict')
    std_n_1: float
        Value to use if n = 1, default is 10. (Not clear if the default is sensible here)

    Returns:
    --------
    bandwidth: float
        Silverman bandwidth for the given sample. This is applied as the bandwidth parameter
        when generating gaussian-based kdes in the LogKDE class.
    """
    # Compute number of samples
    n = len(sample)

    # Deal with very small stds and n = 1 case
    if n > 1:
        # Compute std of sample
        std = np.std(sample)
        if std < std_cutoff:
            if std_proc == "restrict":
                std = std_cutoff
            if std_proc == "kill":
                std = 0
    else:
        # AF-Comment: This is a bit of a weakness (can be arbitrarily incorrect)
        std = std_n_1

    return np.power((4 / 3), 1 / 5) * std * np.power(n, (-1 / 5))
