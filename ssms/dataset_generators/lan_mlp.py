from ssms.basic_simulators.simulator import simulator, bin_simulator_output
from ssms.support_utils import kde_class
import numpy as np
import pickle
import uuid
import os
from scipy.stats import mode
from multiprocessing import Pool
import psutil

from functools import partial

"""
    This module defines a data generator class for use with LANs. 
    The class defined below can be used to generate training data 
    compatible with the expectations of LANs.
"""


class data_generator:
    """The data_generator() class is used to generate training data
      for various likelihood approximators.

    Attributes
    ----------
        generator_config: dict
            Configuation dictionary for the data generator.
            (For an example load ssms.config.data_generator_config['lan'])
        model_config: dict
            Configuration dictionary for the model to be simulated.
            (For an example load ssms.config.model_config['ddm'])
    Methods
    -------
        generate_data_training_uniform(save=False, verbose=True, cpn_only=False)
            Generates training data for LANs.
        get_simulations(theta=None, random_seed=None)
            Generates simulations for a given parameter set.
        _filter_simulations(simulations=None)
            Filters simulations according to the criteria
            specified in the generator_config.
        _make_kde_data(simulations=None, theta=None)
            Generates KDE data from simulations.
        _mlp_get_processed_data_for_theta(random_seed_tuple)
            Helper function for generating training data for MLPs.
        _cpn_get_processed_data_for_theta(random_seed_tuple)
            Helper function for generating training data for CPNs.
        _get_rejected_parameter_setups(random_seed_tuple)
            Helper function that collectes parameters sets which were rejected
            by the filter used in the _filter_simulations() method.
        _make_save_file_name(unique_tag=None)
            Helper function for generating save file names.
        _build_simulator()
            Builds simulator function for LANs.
        _get_ncpus()
            Helper function for determining the number of
            cpus to use for parallelization.

    Returns
    -------
        data_generator object
    """

    def __init__(self, generator_config=None, model_config=None):
        # INIT -----------------------------------------
        if generator_config is None:
            print("No generator_config specified")
            return
        else:
            self.generator_config = generator_config
            self.model_config = model_config
            self._build_simulator()
            self._get_ncpus()

        # Make output folder if not already present
        folder_str_split = self.generator_config["output_folder"].split()

        cnt = 0
        folder_partial = ""
        for folder_str_part in folder_str_split:
            if cnt > 0:
                folder_partial += "/" + folder_str_part
            else:
                folder_partial += folder_str_part

            print("checking: ", folder_partial)

            if not os.path.exists(folder_partial):
                os.makedirs(folder_partial)

    def _get_ncpus(self):
        # Get number of cpus
        if self.generator_config["n_cpus"] == "all":
            n_cpus = psutil.cpu_count(logical=False)
            print("n_cpus used: ", n_cpus)
        else:
            n_cpus = self.generator_config["n_cpus"]

        self.generator_config["n_cpus"] = n_cpus

    def _build_simulator(self):
        self.simulator = partial(
            simulator,
            n_samples=self.generator_config["n_samples"],
            max_t=self.generator_config["max_t"],
            bin_dim=0,
            delta_t=self.generator_config["delta_t"],
            smooth_unif=self.generator_config["smooth_unif"],
        )

    def get_simulations(self, theta=None, random_seed=None):
        out = self.simulator(
            theta=theta,
            model=self.model_config["name"],
            random_state=random_seed,
        )
        return out

    def _filter_simulations(
        self,
        simulations=None,
    ):
        keep = 1
        n_sim = simulations["rts"].shape[0]
        for choice_tmp in simulations["metadata"]["possible_choices"]:
            tmp_rts = simulations["rts"][simulations["choices"] == choice_tmp]
            tmp_n_c = len(tmp_rts)
            if tmp_n_c > 0:
                mode_, mode_cnt_ = mode(tmp_rts, keepdims=False)
                std_ = np.std(tmp_rts)
                mean_ = np.mean(tmp_rts)
                if tmp_n_c < 5:
                    mode_cnt_rel_ = 0
                else:
                    mode_cnt_rel_ = mode_cnt_ / tmp_n_c

            else:
                mode_ = -1
                mode_cnt_ = 0
                mean_ = -1
                std_ = 1
                mode_cnt_rel_ = 0

            # AF-TODO: More flexible way with list of
            # filter objects that provides for each filter
            #  1. Function to compute statistic (e.g. mode)
            #  2. Comparison operator (e.g. <=, != , etc.)
            #  3. Comparator (number to test against)

            keep = (
                keep
                & (mode_ < self.generator_config["simulation_filters"]["mode"])
                & (mean_ <= self.generator_config["simulation_filters"]["mean_rt"])
                & (std_ > self.generator_config["simulation_filters"]["std"])
                & (
                    mode_cnt_rel_
                    <= self.generator_config["simulation_filters"]["mode_cnt_rel"]
                )
                & (tmp_n_c > self.generator_config["simulation_filters"]["choice_cnt"])
            )
        return keep, np.array(
            [mode_, mean_, std_, mode_cnt_rel_, tmp_n_c, n_sim], dtype=np.float32
        )

    def _make_kde_data(self, simulations=None, theta=None):
        n = self.generator_config["n_training_samples_by_parameter_set"]
        p = self.generator_config["kde_data_mixture_probabilities"]
        n_kde = int(n * p[0])
        n_unif_up = int(n * p[1])
        n_unif_down = int(n * p[2])

        if self.generator_config["separate_response_channels"]:
            out = np.zeros(
                (
                    n_kde + n_unif_up + n_unif_down,
                    2 + self.model_config["nchoices"] + len(theta),
                )
            )
        else:
            out = np.zeros((n_kde + n_unif_up + n_unif_down, 3 + len(theta)))

        out[:, : len(theta)] = np.tile(theta, (n_kde + n_unif_up + n_unif_down, 1))

        tmp_kde = kde_class.LogKDE(
            (simulations["rts"], simulations["choices"], simulations["metadata"])
        )

        # Get kde part
        samples_kde = tmp_kde.kde_sample(n_samples=n_kde)
        likelihoods_kde = tmp_kde.kde_eval(data=samples_kde).ravel()

        if self.generator_config["separate_response_channels"]:
            out[:n_kde, (-2 - self.model_config["nchoices"])] = samples_kde[0].ravel()

            r_cnt = 0
            choices = samples_kde[1].ravel()
            for response in simulations["metadata"]["possible_choices"]:
                out[:n_kde, ((-1 - self.model_config["nchoices"]) + r_cnt)] = (
                    choices == response
                ).astype(int)
                r_cnt += 1
        else:
            out[:n_kde, -3] = samples_kde[0].ravel()
            out[:n_kde, -2] = samples_kde[1].ravel()

        out[:n_kde, -1] = likelihoods_kde

        # Get positive uniform part:
        choice_tmp = np.random.choice(
            simulations["metadata"]["possible_choices"], size=n_unif_up
        )

        if simulations["metadata"]["max_t"] < 100:
            rt_tmp = np.random.uniform(
                low=0.0001, high=simulations["metadata"]["max_t"], size=n_unif_up
            )
        else:
            rt_tmp = np.random.uniform(low=0.0001, high=100, size=n_unif_up)

        likelihoods_unif = tmp_kde.kde_eval(data=(rt_tmp, choice_tmp)).ravel()

        out[n_kde : (n_kde + n_unif_up), -3] = rt_tmp
        out[n_kde : (n_kde + n_unif_up), -2] = choice_tmp
        out[n_kde : (n_kde + n_unif_up), -1] = likelihoods_unif

        # Get negative uniform part:
        choice_tmp = np.random.choice(
            simulations["metadata"]["possible_choices"], size=n_unif_down
        )

        rt_tmp = np.random.uniform(low=-1.0, high=0.0001, size=n_unif_down)

        out[(n_kde + n_unif_up) :, -3] = rt_tmp
        out[(n_kde + n_unif_up) :, -2] = choice_tmp
        out[(n_kde + n_unif_up) :, -1] = self.generator_config["negative_rt_cutoff"]
        return out.astype(np.float32)

    def _mlp_get_processed_data_for_theta(self, random_seed_tuple):
        np.random.seed(random_seed_tuple[0])
        keep = 0
        while not keep:
            theta = np.float32(
                np.random.uniform(
                    low=self.model_config["param_bounds"][0],
                    high=self.model_config["param_bounds"][1],
                )
            )

            simulations = self.get_simulations(
                theta=theta, random_seed=random_seed_tuple[1]
            )
            keep, stats = self._filter_simulations(simulations)

        kde_data = self._make_kde_data(simulations=simulations, theta=theta)

        choice_p = np.array(
            [
                (simulations["choices"] == 1.0).sum()
                / simulations["choices"].flatten().shape[0]
            ]
        )
        binned_128 = np.expand_dims(
            bin_simulator_output(simulations, nbins=128, max_t=-1, freq_cnt=True),
            axis=0,
        )
        binned_256 = np.expand_dims(
            bin_simulator_output(simulations, nbins=256, max_t=-1, freq_cnt=True),
            axis=0,
        )

        return {
            "data": kde_data[:, :-1],
            "labels": kde_data[:, -1],
            "choice_p": choice_p,
            "binned_128": binned_128,
            "binned_256": binned_256,
            "theta": np.expand_dims(theta, axis=0),
        }

    def _cpn_get_processed_data_for_theta(self, random_seed_tuple):
        np.random.seed(random_seed_tuple[0])

        # Get a random parameter set
        theta = np.float32(
            np.random.uniform(
                low=self.model_config["param_bounds"][0],
                high=self.model_config["param_bounds"][1],
            )
        )
        # Run the simulator
        simulations = self.get_simulations(
            theta=theta, random_seed=random_seed_tuple[1]
        )

        # Compute the choice probabilities
        choice_p = np.array(
            [
                (simulations["choices"] == 1.0).sum()
                / simulations["choices"].flatten().shape[0]
            ]
        )

        return {
            "choice_p": choice_p,
            "theta": np.expand_dims(theta, axis=0),
        }

    def _get_rejected_parameter_setups(self, random_seed_tuple):
        np.random.seed(random_seed_tuple[0])
        rejected_thetas = []
        keep = 1
        rej_cnt = 0
        while rej_cnt < 100:
            theta = np.float32(
                np.random.uniform(
                    low=self.model_config["param_bounds"][0],
                    high=self.model_config["param_bounds"][1],
                )
            )
            simulations = self.get_simulations(
                theta=theta, random_seed=random_seed_tuple[1]
            )
            keep, stats = self._filter_simulations(simulations)

            if keep == 0:
                print("simulation rejected")
                print("stats: ", stats)
                print("theta", theta)
                rejected_thetas.append(theta)
            rej_cnt += 1

        return rejected_thetas

    def generate_data_training_uniform(self, save=False, verbose=True, cpn_only=False):
        """Generates training data for LANs.

        Arguments
        ---------
            save: bool
                If True, the generated data is saved to disk.
            verbose: bool
                If True, progress is printed to the console.
            cpn_only: bool
                If True, only choice probabilities are computed.
                This is useful for training CPNs.

        Returns
        -------
            data: dict
                Dictionary containing the generated data.
        """
        seeds_1 = np.random.choice(
            400000000, size=self.generator_config["n_parameter_sets"]
        )
        seeds_2 = np.random.choice(
            400000000, size=self.generator_config["n_parameter_sets"]
        )
        seed_args = [(seeds_1[i], seeds_2[i]) for i in range(seeds_1.shape[0])]

        # Inits
        subrun_n = (
            self.generator_config["n_parameter_sets"]
            // self.generator_config["n_subruns"]
        )

        # Get Simulations
        out_list = []
        for i in range(self.generator_config["n_subruns"]):
            if verbose:
                print(
                    "simulation round:",
                    i + 1,
                    " of",
                    self.generator_config["n_subruns"],
                )
            if self.generator_config["n_cpus"] > 1:
                if cpn_only:
                    with Pool(processes=self.generator_config["n_cpus"] - 1) as pool:
                        out_list += pool.map(
                            self._cpn_get_processed_data_for_theta,
                            [
                                k
                                for k in seed_args[
                                    (i * subrun_n) : ((i + 1) * subrun_n)
                                ]
                            ],
                        )
                else:
                    with Pool(processes=self.generator_config["n_cpus"] - 1) as pool:
                        out_list += pool.map(
                            self._mlp_get_processed_data_for_theta,
                            [
                                k
                                for k in seed_args[
                                    (i * subrun_n) : ((i + 1) * subrun_n)
                                ]
                            ],
                        )
            else:
                print("No Multiprocessing, since only one cpu requested!")
                if cpn_only:
                    for k in seed_args[(i * subrun_n) : ((i + 1) * subrun_n)]:
                        out_list.append(self._cpn_get_processed_data_for_theta(k))
                else:
                    for k in seed_args[(i * subrun_n) : ((i + 1) * subrun_n)]:
                        out_list.append(self._mlp_get_processed_data_for_theta(k))
        data = {}

        # Choice probabilities and theta are always needed
        data["choice_p"] = np.concatenate(
            [out_list[k]["choice_p"] for k in range(len(out_list))]
        ).astype(np.float32)
        data["thetas"] = np.concatenate(
            [out_list[k]["theta"] for k in range(len(out_list))]
        ).astype(np.float32)

        # Only if not cpn_only, do we need the rest of the data
        # (which is not computed if cpn_only is selected)
        if not cpn_only:
            data["data"] = np.concatenate(
                [out_list[k]["data"] for k in range(len(out_list))]
            ).astype(np.float32)
            data["labels"] = np.concatenate(
                [out_list[k]["labels"] for k in range(len(out_list))]
            ).astype(np.float32)
            data["binned_128"] = np.concatenate(
                [out_list[k]["binned_128"] for k in range(len(out_list))]
            ).astype(np.float32)
            data["binned_256"] = np.concatenate(
                [out_list[k]["binned_256"] for k in range(len(out_list))]
            ).astype(np.float32)

        # Add metadata to training_data
        data["generator_config"] = self.generator_config
        data["model_config"] = self.model_config

        if save:
            if not os.path.exists(self.generator_config["output_folder"]):
                os.makedirs(self.generator_config["output_folder"])

            full_file_name = (
                self.generator_config["output_folder"]
                + "/"
                + "training_data_"
                + uuid.uuid1().hex
                + ".pickle"
            )  # self.model_config['name'] + '_' + \

            print("Writing to file: ", full_file_name)

            pickle.dump(
                data,
                open(full_file_name, "wb"),
                protocol=self.generator_config["pickleprotocol"],
            )
            return "Dataset completed"

        else:
            return data

    def _nested_get_processed_data(self, random_seed):
        np.random.seed(random_seed)
        theta = np.float32(
            np.random.uniform(
                low=self.model_config["param_bounds"][0],
                high=self.model_config["param_bounds"][1],
            )
        )

        n_components_to_exclude = np.random.choice(
            self.model_config["components"]["n_components"] + 1
        )
        components_to_exclude = np.random.choice(
            self.model_config["components"]["n_components"],
            size=n_components_to_exclude,
            replace=False,
        )

        label_tmp = np.array(
            [1 for i in range(self.model_config["components"]["n_components"])]
        )

        for component_tmp in components_to_exclude:
            # Get component name
            component_name = self.model_config["components"]["names"][component_tmp]
            # Get index number in list of all model parameters
            component_idx_in_full_params = self.model_config["params"].index(
                component_name
            )
            # Set parameter to the 'off_value' that corresponds to a model
            # without this particular model component
            theta[component_idx_in_full_params] = self.model_config["components"][
                "off_values"
            ][component_tmp]
            # Adjust label
            label_tmp = (
                label_tmp - self.model_config["components"]["labels"][component_tmp]
            )

        # Should allow for binned and unbinned version of this
        simulations = self.get_simulations(theta=theta)

        if self.generator_config["nbins"] > 0:
            return {
                "data": np.expand_dims(simulations["data"], axis=0),
                "label_parameters": theta,
                "label_components": label_tmp,
                "metadata": simulations["metadata"],
            }
        else:
            if self.generator_config["separate_response_channels"]:
                choice_data = np.zeros(
                    (simulations["rts"].shape, self.model.config["nchoices"])
                )

                r_cnt = 0
                for response in simulations["metadata"]["possible_choices"]:
                    choice_data[:, r_cnt] = (simulations["choices"] == response).astype(
                        int
                    )
                    r_cnt += 1
                return {
                    "data": np.column_stack([simulations["rts"], choice_data]),
                    "label_parameters": theta,
                    "label_components": label_tmp,
                    "metadata": simulations["metadata"],
                }

            else:
                return {
                    "data": np.column_stack(
                        [simulations["rts"], simulations["choices"]]
                    ),
                    "label_parameters": theta,
                    "label_components": label_tmp,
                    "metadata": simulations["metadata"],
                }

    def generate_data_nested(self, save):
        seeds = np.random.choice(
            400000000, size=self.generator_config["n_parameter_sets"]
        )

        # Inits
        subrun_n = (
            self.generator_config["n_parameter_sets"]
            // self.generator_config["n_subruns"]
        )

        data_list = []
        for i in range(self.generator_config["n_subruns"]):
            print(
                "simulation round: ", i + 1, " of", self.generator_config["n_subruns"]
            )
            with Pool(processes=self.generator_config["n_cpus"]) as pool:
                data_tmp = pool.map(
                    self._nested_get_processed_data,
                    [j for j in seeds[(i * subrun_n) : ((i + 1) * subrun_n)]],
                )

                data_tmp_dict = {}
                data_tmp_dict["data"] = np.float32(
                    np.concatenate([x["data"] for x in data_tmp])
                )
                data_tmp_dict["label_components"] = np.float32(
                    np.concatenate(
                        [
                            np.expand_dims(x["label_components"], axis=0)
                            for x in data_tmp
                        ]
                    )
                )
                data_tmp_dict["label_parameters"] = np.float32(
                    np.concatenate(
                        [
                            np.expand_dims(x["label_parameters"], axis=0)
                            for x in data_tmp
                        ]
                    )
                )
                data_list.append(data_tmp_dict)

        data = {}

        data["data"] = np.float32(np.concatenate([x["data"] for x in data_list]))
        data["label_components"] = np.float32(
            np.concatenate([x["label_components"] for x in data_list])
        )
        data["label_parameters"] = np.float32(
            np.concatenate([x["label_parameters"] for x in data_list])
        )
        data["generator_config"] = self.generator_config
        data["model_config"] = self.model_config

        if save:
            full_file_name = self._make_save_file_name(
                unique_tag="nested_training_data_"
            )
            print("Writing to file: ", full_file_name)

            pickle.dump(
                data,
                open(full_file_name, "wb"),
                protocol=self.generator_config["pickleprotocol"],
            )
            return "Dataset completed"

        else:
            return data

    def generate_data_ratio_estimator(self, save):
        seeds = np.random.choice(400000000, size=self.generator_config["n_subdatasets"])
        print("Starting simulations")

        with Pool(processes=self.generator_config["n_cpus"]) as pool:
            data_tmp = pool.map(self._ratio_estimator_get_processed_data, list(seeds))

        data = {}
        data["data"] = np.float32(np.concatenate([x["data"] for x in data_tmp]))
        data["labels"] = np.float32(np.concatenate([x["labels"] for x in data_tmp]))
        data["generator_config"] = self.generator_config
        data["model_config"] = self.model_config

        if save:
            full_file_name = self._make_save_file_name(
                unique_tag="ratio_training_data_"
            )
            print("Writing to file: ", full_file_name)

            pickle.dump(
                data,
                open(full_file_name, "wb"),
                protocol=self.generator_config["pickleprotocol"],
            )
            return "Dataset completed"

        else:
            return data

    def _training_defective_simulations_get_preprocessed(self, seed):
        np.random.seed(seed)
        rejected_thetas = []
        accepted_thetas = []
        stats_rej = []
        stats_acc = []
        cnt_max = self.generator_config["n_samples"] // 2
        keep = 1
        rej_cnt = 0
        acc_cnt = 0
        while rej_cnt < cnt_max or acc_cnt < cnt_max:
            theta = np.float32(
                np.random.uniform(
                    low=self.model_config["param_bounds"][0],
                    high=self.model_config["param_bounds"][1],
                )
            )
            simulations = self.get_simulations(theta=theta)
            keep, stats = self._filter_simulations(simulations)

            if keep == 0 and rej_cnt < cnt_max:
                print("simulation rejected")
                print("stats: ", stats)
                print("theta", theta)
                rejected_thetas.append(theta)
                stats_rej.append(stats)
                rej_cnt += 1
            elif acc_cnt < cnt_max and keep == 1:
                accepted_thetas.append(theta)
                stats_acc.append(stats)
                acc_cnt += 1
            else:
                pass
        return rejected_thetas

    def _ratio_estimator_get_processed_data(self, random_seed):
        np.random.seed(random_seed)
        theta_real = np.float32(
            np.random.uniform(
                low=self.model_config["param_bounds"][0],
                high=self.model_config["param_bounds"][1],
                size=(
                    self.generator_config["n_trials_per_dataset"],
                    len(self.model_config["param_bounds"][0]),
                ),
            )
        )

        # Should allow for binned and unbinned version of this
        simulations = self.get_simulations(theta=theta_real)
        print("simulations finished!")

        # Generate some new thetas
        theta_fake = np.float32(
            np.random.uniform(
                low=self.model_config["param_bounds"][0],
                high=self.model_config["param_bounds"][1],
                size=(
                    self.generator_config["n_trials_per_dataset"] // 2,
                    len(self.model_config["param_bounds"][0]),
                ),
            )
        )

        # Attach new thetas to existing simulations (negative examples)
        indices_fake = np.random.choice(
            [0, 1], size=self.generator_config["n_trials_per_dataset"] // 2
        )
        theta_real[indices_fake, :] = theta_fake

        if self.generator_config["nbins"] > 0:
            return (
                "Error: Generating data for ratio estimators "
                + "works only for unbinned data at this point"
            )
        else:
            return {
                "data": np.column_stack(
                    [simulations["rts"], simulations["choices"], theta_real]
                ),
                "labels": np.logical_not(indices_fake).astype(int),
            }

    def _make_save_file_name(self, unique_tag=""):
        binned = str(0)
        if self.generator_config["nbins"] > 0:
            binned = str(1)

        training_data_folder = (
            self.generator_config["output_folder"]
            + unique_tag
            + binned
            + "_nbins_"
            + str(self.generator_config["nbins"])
            + "_n_"
            + str(self.generator_config["n_samples"])
        )

        if not os.path.exists(training_data_folder):
            os.makedirs(training_data_folder)

        full_file_name = (
            training_data_folder
            + "/"
            + "training_data_"
            + self.model_config["name"]
            + "_"
            + uuid.uuid1().hex
            + ".pickle"
        )
        return full_file_name

    def generate_rejected_parameterizations(self, save=False):
        seeds = np.random.choice(
            400000000, size=self.generator_config["n_paramseter_sets_rejected"]
        )

        # Get Simulations
        with Pool(processes=self.generator_config["n_cpus"]) as pool:
            rejected_parameterization_list = pool.map(
                self._get_rejected_parameter_setups, seeds
            )
        rejected_parameterization_list = np.concatenate(
            [l_rej for l_rej in rejected_parameterization_list if len(l_rej) > 0]
        )

        if save:
            training_data_folder = (
                self.generator_config["method_folder"]
                + "training_data_binned_"
                + str(int(self.generator_config["binned"]))
                + "_nbins_"
                + str(self.generator_config["nbins"])
                + "_n_"
                + str(self.generator_config["n_samples"])
            )

            if not os.path.exists(training_data_folder):
                os.makedirs(training_data_folder)

            full_file_name = (
                training_data_folder
                + "/"
                + "rejected_parameterizations_"
                + self.generator_config["file_id"]
                + ".pickle"
            )

            print("Writing to file: ", full_file_name)

            pickle.dump(
                np.float32(rejected_parameterization_list),
                open(full_file_name, "wb"),
                protocol=self.generator_config["pickleprotocol"],
            )
            print("Dataset completed")
            return rejected_parameterization_list
        else:
            print("Dataset completed")
            return rejected_parameterization_list
