import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ssms.basic_simulators import drift_functions as df


import ssms
from ssms.basic_simulators import simulator
import os

# print(list(ssms.config.model_config.keys())[:10])
# print(ssms.config.model_config['ddm'])

sim_out = simulator.simulator(model='ulrich',
                              theta={'v': 0,
                                     'a': 1,
                                     'z': 0.5,
                                     't': 0.5}
                              ,
                              n_samples=1000,
                              no_noise=False)

print(sim_out['choices'])