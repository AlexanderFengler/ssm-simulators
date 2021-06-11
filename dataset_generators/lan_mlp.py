import basic_simulators as bs
from support_utils import kde_class
import pandas as pd
import numpy as np
#from itertools import product
import pickle
import uuid
import os 
import sys
from datetime import datetime
from scipy.stats import truncnorm
from scipy.stats import mode

import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Pool
import psutil
import basic_simulators
import config
from functools import partial

class data_generator():
    def __init__(self,
                 generator_config = None,
                 model_config = None
                 ):
    
    # INIT -----------------------------------------
        if generator_config == None:
            print('No generator_config specified')
            return
        else:
            self.generator_config = generator_config 
            self.model_config = model_config
            self._build_simulator()
            self._get_ncpus()

        # Make output folder if not already present
        folder_str_split = self.generator_config['output_folder'].split()
        
        cnt = 0
        folder_partial = ''
        for folder_str_part in folder_str_split:
            if cnt > 0:
                folder_partial += '/' + folder_str_part
            else:
                folder_partial += folder_str_part
            
            print('checking: ', folder_partial)
            
            if not os.path.exists(folder_partial):
                os.makedirs(folder_partial)
    
    def _get_ncpus(self):
        # Sepfic
        if self.generator_config['n_cpus'] == 'all':
            n_cpus = psutil.cpu_count(logical = False)
            print('n_cpus: ', n_cpus)
        else:
            n_cpus = self.generator_config['n_cpus']

        self.generator_config['n_cpus'] = n_cpus
             
    def _build_simulator(self):
        self.simulator = partial(bs.simulator, 
                                 n_samples = self.generator_config['n_samples'],
                                 max_t = self.generator_config['max_t'],
                                 bin_dim = self.generator_config['nbins'],
                                 delta_t = self.generator_config['delta_t'])
                                 
    def get_simulations(self, theta = None):
        
        out = self.simulator(theta  = theta, 
                             model = self.model_config['name']) # AF-TODO Want to change this so that we accept 
        
        #print(self.config['nbins'])
        return out
        # if self.generator_config['nbins'] == 0:
        #     return out
        
        # elif self.generator_config['nbins'] > 0 and not self.generator_config['bin_pointwise']:
        #     #print('passed')
        #     #print(out['rts'])
        #     return bs.bin_simulator_output(out = out,
        #                                    nbins = self.generator_config['nbins'],
        #                                    max_t = self.generator_config['max_t'])
        #     # return self._bin_simulator_output(simulations = out)
        # elif self.generator_config['nbins'] > 0 and self.generator_config['bin_pointwise']:
        #     pass # AF-TODO: Here return pointwise binned data
        # else:
        #     return 'number bins not accurately specified --> returning from simulator without output'

    def _filter_simulations(self,
                            simulations = None,
                            ):
        
        #debug_tmp_n_c_min = 1000000
        keep = 1
        n_sim = simulations['rts'].shape[0]
        for choice_tmp in simulations['metadata']['possible_choices']:
            tmp_rts = simulations['rts'][simulations['choices'] == choice_tmp]
            tmp_n_c = len(tmp_rts)
            print('tmp_n_c')
            print(tmp_n_c)
            #debug_tmp_n_c_min = np.minimum(debug_tmp_n_c_min, tmp_n_c)
            if tmp_n_c > 0:
                mode_, mode_cnt_ = mode(tmp_rts)
                std_ = np.std(tmp_rts)
                mean_ = np.mean(tmp_rts)
                mode_cnt_rel_ = mode_cnt_ / tmp_n_c
            else:
                mode_ = -1
                mode_cnt_ = 0
                mean_ = -1
                std_ = -1
                mode_cnt_rel_ = 1
            
            # AF-TODO: More flexible way with list of filter objects that provides for each filter 
            #  1. Function to compute statistic (e.g. mode)
            #  2. Comparison operator (e.g. <=, != , etc.)
            #  3. Comparator (number to test against)
            
            keep = keep & \
                   (mode_ != self.generator_config['simulation_filters']['mode']) & \
                   (mean_ < self.generator_config['simulation_filters']['mean_rt']) & \
                   (std_ > self.generator_config['simulation_filters']['std']) & \
                   (mode_cnt_rel_ < self.generator_config['simulation_filters']['mode_cnt_rel']) & \
                   (tmp_n_c > self.generator_config['simulation_filters']['choice_cnt'])
        return keep, np.array([mode_, mean_, std_, mode_cnt_rel_, tmp_n_c, n_sim], dtype = np.float32)
             
    def _make_kde_data(self,
                       simulations = None, 
                       theta = None):
        
        n = self.generator_config['n_training_samples_by_parameter_set']
        p = self.generator_config['kde_data_mixture_probabilities']
        n_kde = int(n * p[0])
        n_unif_up = int(n * p[1])
        n_unif_down = int(n * p[2])
                    
        out = np.zeros((n_kde + n_unif_up + n_unif_down, 
                        3 + len(theta)))
        out[:, :len(theta)] = np.tile(theta, (n_kde + n_unif_up + n_unif_down, 1) )
        
        tmp_kde = kde_class.logkde((simulations['rts'],
                                    simulations['choices'], 
                                    simulations['metadata']))

        # Get kde part
        samples_kde = tmp_kde.kde_sample(n_samples = n_kde)
        likelihoods_kde = tmp_kde.kde_eval(data = samples_kde).ravel()

        out[:n_kde, -3] = samples_kde[0].ravel()
        out[:n_kde, -2] = samples_kde[1].ravel()
        out[:n_kde, -1] = likelihoods_kde

        # Get positive uniform part:
        choice_tmp = np.random.choice(simulations['metadata']['possible_choices'],
                                      size = n_unif_up)

        if simulations['metadata']['max_t'] < 100:
            rt_tmp = np.random.uniform(low = 0.0001,
                                       high = simulations['metadata']['max_t'],
                                       size = n_unif_up)
        else: 
            rt_tmp = np.random.uniform(low = 0.0001, 
                                       high = 100,
                                       size = n_unif_up)

        likelihoods_unif = tmp_kde.kde_eval(data = (rt_tmp, choice_tmp)).ravel()

        out[n_kde:(n_kde + n_unif_up), -3] = rt_tmp
        out[n_kde:(n_kde + n_unif_up), -2] = choice_tmp
        out[n_kde:(n_kde + n_unif_up), -1] = likelihoods_unif

        # Get negative uniform part:
        choice_tmp = np.random.choice(simulations['metadata']['possible_choices'],
                                      size = n_unif_down)

        rt_tmp = np.random.uniform(low = - 1.0,
                                   high = 0.0001,
                                   size = n_unif_down)

        out[(n_kde + n_unif_up):, -3] = rt_tmp
        out[(n_kde + n_unif_up):, -2] = choice_tmp
        out[(n_kde + n_unif_up):, -1] = self.generator_config['negative_rt_cutoff']

        return out.astype(np.float)
    
    def _mlp_get_processed_data_for_theta(self,
                                          random_seed):
        
        np.random.seed(random_seed)
        keep = 0
        while not keep:
            theta = np.float32(np.random.uniform(low = self.model_config['param_bounds'][0], 
                                                 high = self.model_config['param_bounds'][1]))
            
            simulations = self.get_simulations(theta = theta)
            print(theta)
            #print(simulations)
            keep, stats = self._filter_simulations(simulations)
            print(keep)

        data = self._make_kde_data(simulations = simulations,
                                   theta = theta)
        
        return data
                     
    def _cnn_get_processed_data_for_theta(self,
                                          random_seed):
        np.random.seed(random_seed)
        theta = np.float32(np.random.uniform(low = self.model_config['param_bounds'][0], 
                                             high = self.model_config['param_bounds'][1]))
        return {'data': np.expand_dims(self.get_simulations(theta = theta)['data'], axis = 0), 'label': theta}
             
    def _get_rejected_parameter_setups(self,
                                       random_seed):
        np.random.seed(random_seed)
        rejected_thetas = []
        keep = 1
        rej_cnt = 0
        while rej_cnt < 100:
            
            theta = np.float32(np.random.uniform(low = self.model_config['param_bounds'][0], 
                                                 high = self.model_config['param_bounds'][1]))
            simulations = self.get_simulations(theta = theta)
            keep, stats = self._filter_simulations(simulations)
            
            if keep == 0:
                print('simulation rejected')
                print('stats: ', stats)
                print('theta', theta)
                rejected_thetas.append(theta)
                #break

            rej_cnt += 1
        
        return rejected_thetas
          
    def generate_data_training_uniform(self, 
                                       save = False):
        
        seeds = np.random.choice(400000000, size = self.generator_config['n_parameter_sets'])
        
        # Inits
        subrun_n = self.generator_config['n_parameter_sets'] // self.generator_config['n_subruns']
        samples_by_param_set = self.generator_config['n_training_samples_by_parameter_set']
        
        if self.generator_config['nbins'] == 0:
            data_tmp = np.zeros((int(self.generator_config['n_parameter_sets'] * self.generator_config['n_training_samples_by_parameter_set']), 
                                  len(self.model_config['param_bounds'][0]) + 3))

        # Get Simulations 
            for i in range(self.generator_config['n_subruns']):
                print('simulation round:', i + 1 , ' of', self.generator_config['n_subruns'])
                with Pool(processes = self.generator_config['n_cpus']) as pool:
                    data_tmp[(i * subrun_n * samples_by_param_set):((i + 1) * subrun_n * samples_by_param_set), :] = np.concatenate(pool.map(self._mlp_get_processed_data_for_theta, 
                                                                                                              [j for j in seeds[(i * subrun_n):((i + 1) * subrun_n)]]))
                
            data_tmp = np.float32(data_tmp)
            
            data = {}
            data['data'] = data_tmp[:, :-1]
            data['labels'] = data_tmp[:, -1]

                
        else:
            #data_grid = np.zeros((self.generator_config['n_parameter_sets'], self.generator_config['nbins'], self.model_config['nchoices']))
            
            for i in range(self.generator_config['n_subruns']):
                print('simulation round: ', i + 1, ' of', self.generator_config['n_subruns'])
                data_list = []
                with Pool(processes = self.generator_config['n_cpus']) as pool:
                    # data_grid[(i * subrun_n): ((i + 1) * subrun_n), :, :] = np.concatenate(pool.map(self._cnn_get_processed_data_for_theta,
                    #                                                                                 [j for j in seeds[(i * subrun_n):((i + 1) * subrun_n)]]))
                    data_tmp = pool.map(self._cnn_get_processed_data_for_theta,
                                          [j for j in seeds[(i * subrun_n):((i + 1) * subrun_n)]])
                    
                    data_tmp_dict = {}
                    data_tmp_dict['data'] = np.float32(np.concatenate([x['data'] for x in data_tmp]))
                    data_tmp_dict['labels'] = np.float32(np.concatenate([np.expand_dims(x['label'], axis = 0) for x in data_tmp]))
                    data_list.append(data_tmp_dict)
            
            data = {}
            data['data'] = np.float32(np.concatenate([x['data'] for x in data_list]))
            data['labels'] = np.float32(np.concatenate([x['labels'] for x in data_list]))
        
        if save:    
            binned = str(0)
            if self.generator_config['nbins'] > 0:
                binned = str(1)

            training_data_folder = self.generator_config['output_folder'] + \
                                  'training_data_' + \
                                  binned + \
                                  '_nbins_' + str(self.generator_config['nbins']) + \
                                  '_n_' + str(self.generator_config['nsamples'])
            
            if not os.path.exists(training_data_folder):
                os.makedirs(training_data_folder)

            full_file_name = training_data_folder + '/' + \
                             'training_data_' + self.model_config['name'] + '_' + \
                             uuid.uuid1().hex + '.pickle'

            print('Writing to file: ', full_file_name)

            pickle.dump(data,
                        open(full_file_name, 'wb'), 
                        protocol = self.config['pickleprotocol'])
            return 'Dataset completed'
        
        else:
            return data
                 
    def generate_rejected_parameterizations(self, 
                                            save = False):

        seeds = np.random.choice(400000000, size = self.config['n_paramseter_sets_rejected'])

        # Get Simulations 
        with Pool(processes = self.config['n_cpus']) as pool:
            rejected_parameterization_list = pool.map(self._get_rejected_parameter_setups, 
                                                      seeds)
        rejected_parameterization_list = np.concatenate([l for l in rejected_parameterization_list if len(l) > 0])

        if save:
            training_data_folder = self.config['method_folder'] + \
                                  'training_data_binned_' + \
                                  str(int(self.config['binned'])) + \
                                  '_nbins_' + str(self.config['nbins']) + \
                                  '_n_' + str(self.config['nsamples'])

            if not os.path.exists(training_data_folder):
                os.makedirs(training_data_folder)

            full_file_name = training_data_folder + '/' + \
                             'rejected_parameterizations_' + \
                             self.config['file_id'] + '.pickle'

            print('Writing to file: ', full_file_name)

            pickle.dump(np.float32(rejected_parameterization_list),
                        open(full_file_name, 'wb'), 
                        protocol = self.config['pickleprotocol'])

            return 'Dataset completed'

        else:
            return data_grid

    # AF-TD Get accepted parameterizations
    # def generate_accepted_parameterizations(self, )


    # AF-TD Get keep / not keep data-set
   # ----------------------------------------------------

    # def _make_param_grid_hierarchical(self):
        
    #     # Initialize global parameters
    #     params_ranges_half = (np.array(self.model_config['param_bounds'][1]) - np.array(self.model_config['param_bounds'][0])) / 2
        
    #     # Sample global parameters from cushioned parameter space
    #     global_stds = np.random.uniform(low = 0.001,
    #                                     high = params_ranges_half / 10,
    #                                     size = (self.generator_config['n_paramseter_sets'], self.generator_config['nparams']))
    #     global_means = np.random.uniform(low = self.model_config['param_bounds'][0] + (params_ranges_half / 5),
    #                                      high = self.model_config['param_bounds'][1] - (params_ranges_half / 5),
    #                                      size = (self.generator_config['n_paramseter_sets'], self.generator_config['nparams']))

    #     # Initialize local parameters (by condition)
    #     subject_param_grid = np.float32(np.zeros((self.generator_config['n_paramseter_sets'], self.generator_config['nsubjects'], self.generator_config['nparams'])))
        
    #     # Sample by subject parameters from global setup (truncate to never go out of allowed parameter space)
    #     for n in range(self.generator_config['n_paramseter_sets']):
    #         for i in range(self.genrator_config['nsubjects']):
    #             a, b = (self.generator_config['param_bounds'][0] - global_means[n]) / global_stds[n], (self.config['param_bounds'][1] - global_means[n]) / global_stds[n]
    #             subject_param_grid[n, i, :] = np.float32(global_means[n] + truncnorm.rvs(a, b, size = global_stds.shape[1]) * global_stds[n])

    #     return subject_param_grid, global_stds, global_means

    # def generate_data_training_uniform_cnn_mlp(self, 
    #                                            save = False):

    #     seeds = np.random.choice(400000000, size = self.config['n_paramseter_sets'])
        
    #     # Inits
    #     subrun_n = self.config['n_paramseter_sets'] // self.config['n_subruns']
        
    #     data_grid = {'mlp': np.zeros((int(self.config['n_paramseter_sets'] * 1000), 
    #                                   len(self.config['param_bounds'][0]) + 3)),
    #                  'cnn': np.zeros((int(self.config['n_paramseter_sets']), 
    #                                   self.config['nbins'], 
    #                                   self.config['nchoices']))}

    #     # Get Simulations 
    #     for i in range(self.config['n_subruns']):
    #         print('simulation round:', i + 1 , ' of', self.config['n_subruns'])
    #         with Pool(processes = self.config['n_cpus']) as pool:
    #             data_grid[(i * subrun_n * 1000):((i + 1) * subrun_n * 1000), :] = np.concatenate(pool.map(self._mlp_get_processed_data_for_theta, 
    #                                                                                                     [j for j in seeds[(i * subrun_n):((i + 1) * subrun_n)]]))
    #     else:
    #         data_grid = np.zeros((int(self.config['n_paramseter_sets'], self.config['nbins'], self.config['nchoices'])))
            
    #         for i in range(self.config['n_subruns']):
    #             print('simulation round: ', i + 1, ' of', self.config['n_subruns'])
    #             with Pool(processes = self.config['n_cpus']) as pool:
    #                 data_grid[(i * subrun_n): ((i + 1) * subrun_n), :, :] = np.concatenate(pool.map(self._cnn_get_processed_data_for_theta,
    #                                                                                                 [j for j in seeds[(i * subrun_n):((i + 1) * subrun_n)]]))
        
    #     if save:
    #         training_data_folder = self.config['method_folder'] + \
    #                               'training_data_binned_' + \
    #                               str(int(self.config['binned'])) + \
    #                               '_nbins_' + str(self.config['nbins']) + \
    #                               '_n_' + str(self.config['nsamples'])

    #         if not os.path.exists(training_data_folder):
    #             os.makedirs(training_data_folder)

    #         full_file_name = training_data_folder + '/' + \
    #                          'data_' + \
    #                          self.config['file_id'] + '.pickle'
    #         print('Writing to file: ', full_file_name)

    #         pickle.dump(np.float32(data_grid),
    #                     open(full_file_name, 'wb'), 
    #                     protocol = self.config['pickleprotocol'])
    #         return 'Dataset completed'
        
    #     else:
    #         return data_grid
         

   
    # def generate_data_hierarchical(self, save = False):
        
    #     subject_param_grid, global_stds, global_means = self._make_param_grid_hierarchical()
    #     subject_param_grid_adj_sim = np.reshape(subject_param_grid, (-1, self.config['nparams'])).tolist()
    #     subject_param_grid_adj_sim = tuple([(np.array(i),) for i in subject_param_grid_adj_sim])
        
    #     with Pool(processes = self.config['n_cpus']) as pool:
    #         data_grid = np.array(pool.starmap(self.get_simulations, subject_param_grid_adj_sim))
            
    #     if save:
    #         training_data_folder = self.config['method_folder'] + 'parameter_recovery_hierarchical_data_binned_' + str(int(self.config['binned'])) + \
    #                                '_nbins_' + str(self.config['nbins']) + \
    #                                '_n_' + str(self.config['nsamples'])
            
    #         full_file_name = training_data_folder + '/' + \
    #                          self.config['method'] + \
    #                          '_nchoices_' + str(self.config['nchoices']) + \
    #                          '_parameter_recovery_hierarchical_' + \
    #                          'binned_' + str(int(self.config['binned'])) + \
    #                          '_nbins_' + str(self.config['nbins']) + \
    #                          '_nreps_' + str(self.config['nreps']) + \
    #                          '_n_' + str(self.config['nsamples']) + \
    #                          '_nsubj_' + str(self.config['nsubjects']) + \
    #                          '.pickle'
            
    #         if not os.path.exists(training_data_folder):
    #             os.makedirs(training_data_folder)
            
    #         print('saving dataset as ', full_file_name)
            
    #         pickle.dump(([subject_param_grid, global_stds, global_means], 
    #                       np.expand_dims(data_grid, axis = 0),
    #                       self.config['meta']), 
    #                     open(full_file_name, 'wb'), 
    #                     protocol = self.config['pickleprotocol'])
            
    #         return 'Dataset completed'
    #     else:
    #         return ([subject_param_grid, global_stds, global_means], data_grid, meta)




        # def generate_data_parameter_recovery(self, save = False):
        
    #     # Make parameters
    #     theta_list = [np.float32(np.random.uniform(low = self.config['param_bounds'][0], 
    #                                                 high = self.config['param_bounds'][1])) for i in range(self.config['n_paramseter_sets'])]
        
    #     # Get simulations
    #     with Pool(processes = self.config['n_cpus']) as pool:
    #         data_grid = np.array(pool.map(self.get_simulations_param_recov, theta_list))

    #     print('data_grid shape: ', data_grid.shape)
    #     #data_grid = np.expand_dims(data_grid, axis = 0)
        
    #     # Add the binned versions
    #     binned_tmp_256 = np.zeros((self.config['n_paramseter_sets'], 256, 2))
    #     binned_tmp_512 = np.zeros((self.config['n_paramseter_sets'], 512, 2))

    #     for i in range(self.config['n_paramseter_sets']):
    #         print('subset shape: ', data_grid[i, :, 0].shape)
    #         data_tmp = (np.expand_dims(data_grid[i, :, 0], axis = 1),
    #                     np.expand_dims(data_grid[i, :, 1], axis = 1),
    #                     config['meta'])
            
    #         binned_tmp_256[i, :, :] = bs.bin_simulator_output(out = data_tmp,
    #                                                              nbins = 256,
    #                                                              max_t = self.config['binned_max_t'])
            
    #         binned_tmp_512[i, :, :] = bs.bin_simulator_output(out = data_tmp,
    #                                                              nbins = 512,
    #                                                              max_t = self.config['binned_max_t'])

    #         # AF-TODO: Add binned pointwise

    #     # Save to correct destination
    #     if save:
    #         binned_tmp = ['0', '1', '1']
    #         bins_tmp = ['0', '256', '512']
    #         data_arrays = [data_grid, binned_tmp_256, binned_tmp_512]
            
    #         for k in range(len(binned_tmp)):
    #             # Make folder unbinnend
    #             data_folder = self.config['output_folder'] + \
    #                                   'parameter_recovery_data_binned_' + \
    #                                   binned_tmp[k] + \
    #                                   '_nbins_' + bins_tmp[k] + \
    #                                   '_n_' + str(self.config['nsamples'])

    #             if not os.path.exists(data_folder):
    #                 os.makedirs(data_folder)

    #             full_file_name = data_folder + '/' + \
    #                              self.model_config['method'] + \
    #                              '_parameter_recovery_binned_' + \
    #                              binned_tmp[k] + \
    #                              '_nbins_' + bins_tmp[k] + \
    #                              '_nreps_' + str(self.config['nreps']) + \
    #                              '_n_' + str(self.config['n_samples']) + \
    #                              '.pickle'

    #             print('Writing to file: ', full_file_name)
                
    #             pickle.dump({'parameters': np.float32(np.stack(theta_list)), 
    #                          'data': np.float32(data_arrays[k]),
    #                          'metadata': self.config['meta']}, # AF-TD: Need better way, I don't want to use config['meta']!
    #                          open(full_file_name, 'wb'), 
    #                          protocol = self.config['pickleprotocol'])
            
    #         return 'Dataset completed and stored'
        
    #     # Or else return the data
    #     else:
    #         return {'labels': np.float32(np.stack(theta_list)), 'data': np.float32(data_grid)}

    # def get_simulations_param_recov(self, 
    #                                 theta = None):
        
    #     simulations = self.get_simulations(theta = theta)
    #     return np.concatenate([simulations['rts'], simulations[1]], axis = 1)
