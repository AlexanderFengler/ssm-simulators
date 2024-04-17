# Globaly settings for cython
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False

# Functions for DDM data simulation
import cython
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.math cimport log, sqrt, pow, fmax, atan, sin, cos, tan, M_PI, M_PI_2
from libc.time cimport time

import numpy as np
cimport numpy as np
import numbers
#import pandas as pd

DTYPE = np.float32

cdef set_seed(random_state):
    """
    if random state is provided,
    this function sets a random state globally for the function. 
    """
    if random_state is None:
        return srand(time(NULL))
    if isinstance(random_state, numbers.Integral):
        return srand(random_state)

# Method to draw random samples from a gaussian
cdef float random_uniform():
    cdef float r = rand()
    return r / RAND_MAX

cdef float random_exponential():
    return - log(random_uniform())

cdef float random_stable(float alpha):
    cdef float eta, u, w, x

    u = M_PI * (random_uniform() - 0.5)
    w = random_exponential()

    if alpha == 1.0:
        eta = M_PI_2 # useless but kept to remain faithful to wikipedia entry
        x = (1.0 / eta) * ((M_PI_2) * tan(u))
    else:
        x = (sin(alpha * u) / (pow(cos(u), 1 / alpha))) * pow(cos(u - (alpha * u)) / w, (1.0 - alpha) / alpha)
    return x

cdef float[:] draw_random_stable(int n, float alpha):

    cdef int i
    cdef float[:] result = np.zeros(n, dtype = DTYPE)

    for i in range(n):
        result[i] = random_stable(alpha)
    return result

cdef float random_gaussian():
    cdef float x1, x2, w
    w = 2.0

    while(w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = ((-2.0 * log(w)) / w) ** 0.5
    return x1 * w

cdef int sign(float x):
    return (x > 0) - (x < 0)

cdef float csum(float[:] x):
    cdef int i
    cdef int n = x.shape[0]
    cdef float total = 0
    
    for i in range(n):
        total += x[i]
    
    return total

## @cythonboundscheck(False)
cdef void assign_random_gaussian_pair(float[:] out, int assign_ix):
    cdef float x1, x2, w
    w = 2.0

    while(w >= 1.0):
        x1 = (2.0 * random_uniform()) - 1.0
        x2 = (2.0 * random_uniform()) - 1.0
        w = (x1 * x1) + (x2 * x2)

    w = ((-2.0 * log(w)) / w) ** 0.5
    out[assign_ix] = x1 * w
    out[assign_ix + 1] = x2 * w # this was x2 * 2 ..... :0 

# @cythonboundscheck(False)
cdef float[:] draw_gaussian(int n):
    # Draws standard normal variables - need to have the variance rescaled
    cdef int i
    cdef float[:] result = np.zeros(n, dtype=DTYPE)
    for i in range(n // 2):

        assign_random_gaussian_pair(result, i * 2)
    if n % 2 == 1:
        result[n - 1] = random_gaussian()
    return result

# DUMMY TEST SIMULATOR ------------------------------------------------------------------------
# Simulate (rt, choice) tuples from: SIMPLE DDM -----------------------------------------------
# Simplest algorithm
# delete random comment
# delete random comment 2
#@cython.boundscheck(False)
#@cython.wraparound(False)

def test(np.ndarray[float, ndim = 1] v, # drift by timestep 'delta_t'
         np.ndarray[float, ndim = 1] a, # boundary separation
         np.ndarray[float, ndim = 1] z,  # between 0 and 1
         np.ndarray[float, ndim = 1] t, # non-decision time
         float s = 1, # noise sigma
         float delta_t = 0.001, # timesteps fraction of seconds
         float max_t = 20, # maximum rt allowed
         int n_samples = 20000, # number of samples considered
         int n_trials = 10,
         random_state = None,
         smooth = False,
         return_option = 'full', # 'full' or 'minimal'
         ):

    set_seed(random_state)
    # Param views
    cdef float[:] v_view = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t)
    cdef float sqrt_st = delta_t_sqrt * s

    cdef float y, t_particle, smooth_u

    #cdef int n
    cdef Py_ssize_t n, k
    cdef int m = 0
    cdef int num_draws = int(max_t / delta_t + 1)
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    
    for k in range(n_trials):
        # Loop over samples
        for n in range(n_samples):
            y = z_view[k] * a_view[k] # reset starting point
            t_particle = 0.0 # reset time

            # Random walker
            while y <= a_view[k] and y >= 0 and t <= max_t:
                y += v_view[k] * delta_t + sqrt_st * gaussian_values[m] # update particle position
                t_particle += delta_t
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            # Note that for purposes of consistency with Navarro and Fuss, 
            # the choice corresponding the lower barrier is +1, higher barrier is -1

            # Apply smoothing with uniform if desired
            if smooth:
                if t_particle == 0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                else:
                    smooth_u = (0.5 - random_uniform()) * delta_t
            else:
                smooth_u = 0.0
            
            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # store rt
            choices_view[n, k, 0] = (-1) * sign(y) # store choice

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'metadata': {'v': v,
                                                            'a': a,
                                                            'z': z,
                                                            't': t,
                                                            's': s,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'ddm',
                                                            'boundary_fun_type': 'constant',
                                                            'possible_choices': [-1, 1]}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'metadata': {'simulator': 'ddm', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': 'constant',
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,}}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')

# ---------------------------------------------------------------------------------------------

# DUMMY TEST SIMULATOR ------------------------------------------------------------------------
# Simulate (rt, choice) tuples from: SIMPLE DDM -----------------------------------------------
# Simplest algorithm
# delete random comment
# delete random comment 2
#@cython.boundscheck(False)
#@cython.wraparound(False)

def test2(np.ndarray[float, ndim = 1] v, # drift by timestep 'delta_t'
          np.ndarray[float, ndim = 1] a, # boundary separation
          np.ndarray[float, ndim = 1] z,  # between 0 and 1
          np.ndarray[float, ndim = 1] t, # non-decision time
          float s = 1, # noise sigma
          float delta_t = 0.001, # timesteps fraction of seconds
          float max_t = 20, # maximum rt allowed
          int n_samples = 20000, # number of samples considered
          int n_trials = 10,
          random_state = None,
          smooth = False,
          return_option = 'full', # 'full' or 'minimal'
          ):

    set_seed(random_state)
    # Param views
    cdef float[:] v_view = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t)
    cdef float sqrt_st = delta_t_sqrt * s

    cdef float y, t_particle, smooth_u

    #cdef int n
    cdef Py_ssize_t n, k
    cdef int m = 0
    cdef int num_draws = int(max_t / delta_t + 1)
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    
    for k in range(n_trials):
        # Loop over samples
        for n in range(n_samples):
            y = z_view[k] * a_view[k] # reset starting point
            t_particle = 0.0 # reset time

            # Random walker
            while y <= a_view[k] and y >= 0 and t <= max_t:
                y += v_view[k] * delta_t + sqrt_st * gaussian_values[m] # update particle position
                t_particle += delta_t
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            # Note that for purposes of consistency with Navarro and Fuss, 
            # the choice corresponding the lower barrier is +1, higher barrier is -1

            # Apply smoothing with uniform if desired
            if smooth:
                if t_particle == 0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                else:
                    smooth_u = (0.5 - random_uniform()) * delta_t
            else:
                smooth_u = 0.0
            
            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # store rt
            choices_view[n, k, 0] = (-1) * sign(y) # store choice

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'metadata': {'v': v,
                                                            'a': a,
                                                            'z': z,
                                                            't': t,
                                                            's': s,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'ddm',
                                                            'boundary_fun_type': 'constant',
                                                            'possible_choices': [-1, 1]}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'metadata': {'simulator': 'ddm', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': 'constant',
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,}}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# ---------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Full DDM with flexible bounds --------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def full_ddm_hddm_base(np.ndarray[float, ndim = 1] v, # = 0,
                       np.ndarray[float, ndim = 1] a, # = 1,
                       np.ndarray[float, ndim = 1] z, # = 0.5,
                       np.ndarray[float, ndim = 1] t, # = 0.0,
                       np.ndarray[float, ndim = 1] sz, # = 0.05,
                       np.ndarray[float, ndim = 1] sv, # = 0.1,
                       np.ndarray[float, ndim = 1] st, # = 0.0,
                       np.ndarray[float, ndim = 1] deadline, # = 0.0,
                       float s = 1,
                       float delta_t = 0.001,
                       float max_t = 20,
                       int n_samples = 20000,
                       int n_trials = 1,
                       random_state = None,
                       smooth = False,
                       return_option = 'full', # 'full' or 'minimal'
                       **kwargs,
                       ):

    set_seed(random_state)
    # cdef int cov_length = np.max([v.size, a.size, w.size, t.size]).astype(int)

    # Param views
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] sz_view = sz
    cdef float[:] sv_view = sv
    cdef float[:] st_view = st
    cdef float[:] deadline_view = deadline

    # Data-structs for trajectory storage
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)

    cdef float y, t_particle, t_tmp, smooth_u, deadline_tmp
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float drift_increment = 0.0
    cdef float[:] gaussian_values = draw_gaussian(num_draws) 

    # Loop over trials
    for k in range(n_trials): 
        # Loop over samples
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k]) 
        for n in range(n_samples):
            # initialize starting point
            y = (z_view[k] * (a_view[k]))  # reset starting position
            
            # get drift by random displacement of v 
            drift_increment = (v_view[k] + sv_view[k] * gaussian_values[m]) * delta_t
            t_tmp = t_view[k] + (2 * (random_uniform() - 0.5) * st_view[k])
            
            # apply uniform displacement on y
            y += 2 * (random_uniform() - 0.5) * sz_view[k]
            
            # increment m appropriately
            m += 1
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0
            
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index
            
            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y

            # Random walker
            while y >= 0 and y <= a_view[k] and t_particle <= deadline_tmp:
                y += drift_increment + (sqrt_st * gaussian_values[m])
                t_particle += delta_t
                ix += 1
                m += 1
                
                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            # Apply smoothing with uniform if desired
            if smooth:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = t_particle + t_tmp + smooth_u # Store rt
            
            if y < 0:
                choices_view[n, k, 0] = 0 # Store choice
            else:
                choices_view[n, k, 0] = 1

            # If the rt exceeds the deadline, set rt to -999 and choice to -1 
            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'metadata': {'v': v,
                                'a': a,
                                'z': z,
                                't': t,
                                'sz': sz,
                                'sv': sv,
                                'st': st,
                                'deadline': deadline,
                                's': s,
                                'delta_t': delta_t,
                                'max_t': max_t,
                                'n_samples': n_samples,
                                'n_trials': n_trials,
                                'simulator': 'full_ddm_hddm_base',
                                'possible_choices': [0, 1],
                                'boundary_fun_type': 'constant',
                                'trajectory': traj}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'metadata': {'simulator': 'full_ddm_hddm_base', 
                                                             'possible_choices': [0, 1],
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             'boundary_fun_type': 'constant'}}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -------------------------------------------------------------------------------------------------


# Simulate (rt, choice) tuples from: SIMPLE DDM -----------------------------------------------
# Simplest algorithm
# delete random comment
# delete random comment 2
#@cython.boundscheck(False)
#@cython.wraparound(False)

def ddm(np.ndarray[float, ndim = 1] v, # drift by timestep 'delta_t'
        np.ndarray[float, ndim = 1] a, # boundary separation
        np.ndarray[float, ndim = 1] z,  # between 0 and 1
        np.ndarray[float, ndim = 1] t, # non-decision time
        np.ndarray[float, ndim = 1] deadline, # maximum rt allowed
        max_t = 20, # maximum rt allowed
        float s = 1, # noise sigma
        float delta_t = 0.001, # timesteps fraction of seconds
        int n_samples = 20000, # number of samples considered
        int n_trials = 10,
        random_state = None,
        return_option = 'full', # 'full' or 'minimal'
        smooth = False,
        **kwargs):

    set_seed(random_state)
    # Param views
    cdef float[:] v_view = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline

    # Data-structs for trajectory storage
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t)
    cdef float sqrt_st = delta_t_sqrt * s

    cdef float y, t_particle, smooth_u, deadline_tmp

    #cdef int n
    cdef Py_ssize_t n, ix, k
    cdef int m = 0
    cdef int num_draws = int(max_t / delta_t + 1)
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    
    for k in range(n_trials):
        # Loop over samples
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
        for n in range(n_samples):
            y = z_view[k] * a_view[k] # reset starting point
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index

            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y

            # Random walker
            while y <= a_view[k] and y >= 0 and t_particle <= deadline_tmp:
                y += v_view[k] * delta_t + sqrt_st * gaussian_values[m] # update particle position
                t_particle += delta_t
                m += 1
                ix += 1

                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y

                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            # Note that for purposes of consistency with Navarro and Fuss, 
            # the choice corresponding the lower barrier is +1, higher barrier is -1

            # Apply smoothing with uniform if desired
            if smooth:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # store rt
            choices_view[n, k, 0] = sign(y) # store choice

            # If the rt exceeds the deadline, set rt to -999 and choice to -1 
            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices,  'metadata': {'v': v,
                                                            'a': a,
                                                            'z': z,
                                                            't': t,
                                                            'deadline': deadline,
                                                            's': s,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'ddm',
                                                            'boundary_fun_type': 'constant',
                                                            'possible_choices': [-1, 1],
                                                            'trajectory': traj}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices,  'metadata': {'simulator': 'ddm', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': 'constant',
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,}}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound(np.ndarray[float, ndim = 1] v,
                  np.ndarray[float, ndim = 1] a,
                  np.ndarray[float, ndim = 1] z,
                  np.ndarray[float, ndim = 1] t,
                  np.ndarray[float, ndim = 1] deadline,
                  float s = 1,
                  float max_t = 20,
                  float delta_t = 0.001,
                  int n_samples = 20000,
                  int n_trials = 1,
                  boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                  boundary_multiplicative = True,
                  boundary_params = {},
                  random_state = None,
                  return_option = 'full',
                  smooth = False,
                  **kwargs,
                  ):

    set_seed(random_state)
    #cdef int cov_length = np.max([v.size, a.size, w.size, t.size]).astype(int)
    # Param views:
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline

    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:,:] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)

    cdef float y, t_particle, smooth_u, deadline_tmp
    cdef Py_ssize_t n
    cdef Py_ssize_t ix
    cdef Py_ssize_t m = 0
    cdef Py_ssize_t k
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef float[:] boundary_view = boundary

    # Loop over samples
    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k]) 
        for n in range(n_samples):
            y = (-1) * boundary_view[0] + (z_view[k] * 2 * (boundary_view[0]))  # reset starting position 
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index
            # if deadline >> max_t, then deadline_tmp = max_t, regardless of t-value, otherwise deadline applies
            # Can improve with less checks
            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y

            # Random walker
            while (y >= (-1) * boundary_view[ix]) and (y <= boundary_view[ix]) and (t_particle <= deadline_tmp):
                y += (v_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                t_particle += delta_t
                ix += 1
                m += 1
                
                # Can improve with less checks
                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y
                
                # Can improve with less checks
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            if smooth:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # Store rt

            #rts_view[n, k, 0] = t_particle + t_view[k] # Store rt
            choices_view[n, k, 0] = sign(y) # Store choice

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices,  'metadata': {'v': v,
                                                              'a': a,
                                                              'z': z,
                                                              't': t,
                                                              's': s,
                                                              'deadline': deadline,
                                                              **boundary_params,
                                                              'delta_t': delta_t,
                                                              'max_t': max_t,
                                                              'n_samples': n_samples,
                                                              'n_trials': n_trials,
                                                              'simulator': 'ddm_flexbound',
                                                              'boundary_fun_type': boundary_fun.__name__,
                                                              'possible_choices': [-1, 1],
                                                              'trajectory': traj,
                                                              'boundary': boundary,
                                                             }}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices,  'metadata': {'simulator': 'ddm_flexbound', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
## ----------------------------------------------------------------------------------------------------


# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES AND FLEXIBLE SLOPE -----------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flex(np.ndarray[float, ndim = 1] v,
             np.ndarray[float, ndim = 1] a,
             np.ndarray[float, ndim = 1] z,
             np.ndarray[float, ndim = 1] t,
             np.ndarray[float, ndim = 1] deadline,
             float s = 1,
             float delta_t = 0.001,
             float max_t = 20,
             int n_samples = 20000,
             int n_trials = 1,
             boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
             drift_fun = None,
             boundary_multiplicative = True,
             boundary_params = {},
             drift_params = {},
             random_state = None,
             return_option = 'full',
             smooth = False,
             **kwargs):

    set_seed(random_state)
    # Param views:
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline

    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:,:] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    drift = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float y, t_particle, smooth_u, deadline_tmp
    cdef Py_ssize_t n 
    cdef Py_ssize_t ix
    cdef Py_ssize_t m = 0
    cdef Py_ssize_t k
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    cdef float[:] boundary_view = boundary
    cdef float[:] drift_view = drift

    # Loop over samples
    for k in range(n_trials):
        # Precompute boundary evaluations and drift evaluations
        
        # Drift
        drift_params_tmp = {key: drift_params[key][k] for key in drift_params.keys()}
        drift[:] = np.add(v_view[k], drift_fun(t = t_s, **drift_params_tmp)).astype(DTYPE)

        # Boundary
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)

        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
        for n in range(n_samples):
            y = (-1) * boundary_view[0] + (z_view[k] * 2 * (boundary_view[0]))  # reset starting position 
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index
            
            # Can improve with less checks
            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y

            # Random walker
            while (y >= (-1) * boundary_view[ix]) and (y <= boundary_view[ix]) and (t_particle <= deadline_tmp):
                y += (drift_view[ix] * delta_t) + (sqrt_st * gaussian_values[m])
                t_particle += delta_t
                ix += 1
                m += 1
                
                # Can improve with less checks
                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y
                
                # Can improve with less checks
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            if smooth:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # Store rt
            choices_view[n, k, 0] = sign(y) # Store choice

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999
            
    if return_option == 'full':
        return {'rts': rts, 'choices': choices,  'metadata': {'v': v,
                                                            'a': a,
                                                            'z': z,
                                                            't': t,
                                                            'deadline': deadline,
                                                            's': s,
                                                            **boundary_params,
                                                            **drift_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'ddm_flex',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'drift_fun_type': boundary_fun.__name__,
                                                            'possible_choices': [-1, 1],
                                                            'trajectory': traj,
                                                            'drift': drift,
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices,  'metadata': {'simulator': 'ddm_flex', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'drift_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# ----------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------
# Simulate (rt, choice) tuples from: Levy Flight with Flex Bound -------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
#def glob_flexbound(np.ndarray[float, ndim = 1] v,
#                   np.ndarray[float, ndim = 1] a,
#                   np.ndarray[float, ndim = 1] z,
#                   np.ndarray[float, ndim = 1] alphar,
#                   np.ndarray[float, ndim = 1] g,
#                   np.ndarray[float, ndim = 1] t,
#                   np.ndarray[float, ndim = 1] deadline,
#                   float s = 1, # strictly speaking this is a variance multiplier here, not THE variance !
#                   float delta_t = 0.001,
#                   float max_t = 20,
#                   int n_samples = 20000,
#                   int n_trials = 1,
#                   boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
#                   boundary_multiplicative = True,
#                   boundary_params = {},
#                   random_state = None,
#                   return_option = 'full',
#                   smooth = False,
#                   **kwargs):
#
#    set_seed(random_state)
#    #cdef int cov_length = np.max([v.size, a.size, w.size, t.size]).astype(int)
#    # Param views:
#    cdef float[:] v_view  = v
#    cdef float[:] a_view = a
#    cdef float[:] z_view = z
#    cdef float[:] alphar_view = alphar
#    cdef float[:] g_view = g
#    cdef float[:] t_view = t
#    cdef float[:] deadline_view = deadline
#
#    # Data-struct for trajectory storage
#    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
#    traj[:, :] = -999 
#    cdef float[:,:] traj_view = traj
#
#    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
#    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
#
#    cdef float[:,:, :] rts_view = rts
#    cdef int[:,:, :] choices_view = choices
#
#    cdef float delta_t_alpha # = pow(delta_t, 1.0 / alpha) # correct scalar so we can use standard normal samples for the brownian motion
#
#    # Boundary storage for the upper bound
#    cdef int num_draws = int((max_t / delta_t) + 1)
#    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
#    boundary = np.zeros(t_s.shape, dtype = DTYPE)
#    cdef float[:] boundary_view = boundary
#
#    cdef float y, t_particle, smooth_u, deadline_tmp
#    cdef Py_ssize_t n 
#    cdef Py_ssize_t ix
#    cdef Py_ssize_t k
#    cdef Py_ssize_t m = 0
#    #cdef int n, ix
#    #cdef int m = 0
#    cdef float[:] alpha_stable_values = draw_random_stable(num_draws, alphar_view[0])
#
#    for k in range(n_trials):
#        delta_t_alpha = s * pow(delta_t, 1.0 / alphar_view[k])
#        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
#
#        # Precompute boundary evaluations
#        if boundary_multiplicative:
#            # print(a)
#            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
#        else:
#            # print(a)
#            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
#
#        deadline_tmp = min(max_t, deadline[k] - t_view[k])
#        # Loop over samples
#        for n in range(n_samples):
#            y = (-1) * boundary_view[0] + (z_view[k] * 2 * (boundary_view[0]))  # reset starting position 
#            t_particle = 0.0 # reset time
#            ix = 0 # reset boundary index
#            if n == 0:
#                if k == 0:
#                    traj_view[0, 0] = y
#
#            # Random walker
#            while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t_particle <= deadline_tmp:
#                y += ((v_view[k] - (g_view[k] * y)) * delta_t) + (delta_t_alpha * alpha_stable_values[m])
#                t_particle += delta_t
#                ix += 1
#                m += 1
#                if n == 0:
#                    if k == 0:
#                        traj_view[ix, 0] = y
#                if m == num_draws:
#                    alpha_stable_values = draw_random_stable(num_draws, alphar_view[k])
#                    m = 0
#
#            if smooth:
#                if t_particle == 0:
#                    smooth_u = random_uniform() * 0.5 * delta_t
#                else:
#                    smooth_u = (0.5 - random_uniform()) * delta_t
#            else:
#                smooth_u = 0.0
#
#            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # Store rt
#            choices_view[n, k, 0] = sign(y) # Store choice
#
#            if rts_view[n, k, 0] > deadline_view[k]:
#                rts_view[n, k, 0] = -999
#                choices_view[n, k, 0] = -1
#
#    if return_option == 'full':
#        return {'rts': rts, 'choices': choices,  'metadata': {'v': v,
#                                                            'a': a,
#                                                            'z': z,
#                                                            't': t,
#                                                            'alphar': alphar,
#                                                            'g': g,
#                                                            's': s,
#                                                            'deadline': deadline,
#                                                            **boundary_params,
#                                                            'delta_t': delta_t,
#                                                            'max_t': max_t,
#                                                            'n_samples': n_samples,
#                                                            'n_trials': n_trials,
#                                                            'simulator': 'glob_flexbound',
#                                                            'boundary_fun_type': boundary_fun.__name__,
#                                                            'possible_choices': [-1, 1],
#                                                            'trajectory': traj,
#                                                            'boundary': boundary}}
#    elif return_option == 'minimal':
#        return {'rts': rts, 'choices': choices,  'metadata': {'simulator': 'glob_flexbound', 
#                                                             'possible_choices': [-1, 1],
#                                                             'boundary_fun_type': boundary_fun.__name__,
#                                                             'n_samples': n_samples,
#                                                             'n_trials': n_trials,
#                                                             }}
#    else:
#        raise ValueError('return_option must be either "full" or "minimal"')
# -------------------------------------------------------------------------------------------------


# Simulate (rt, choice) tuples from: Levy Flight with Flex Bound -------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)

def levy_flexbound(np.ndarray[float, ndim = 1] v,
                   np.ndarray[float, ndim = 1] a,
                   np.ndarray[float, ndim = 1] z,
                   np.ndarray[float, ndim = 1] alpha,
                   np.ndarray[float, ndim = 1] t,
                   np.ndarray[float, ndim = 1] deadline,
                   float s = 1, # strictly speaking this is a variance multiplier here, not THE variance !
                   float delta_t = 0.001,
                   float max_t = 20,
                   int n_samples = 20000,
                   int n_trials = 1,
                   boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                   boundary_multiplicative = True,
                   boundary_params = {},
                   random_state = None,
                   return_option = 'full',
                   smooth = False,
                   **kwargs):

    set_seed(random_state)
    #cdef int cov_length = np.max([v.size, a.size, w.size, t.size]).astype(int)
    # Param views:
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] alpha_view = alpha
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline

    # Data-struct for trajectory storage
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:,:] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:,:, :] rts_view = rts
    cdef int[:,:, :] choices_view = choices

    cdef float delta_t_alpha # = pow(delta_t, 1.0 / alpha) # correct scalar so we can use standard normal samples for the brownian motion

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y, t_particle, smooth_u, deadline_tmp
    cdef Py_ssize_t n 
    cdef Py_ssize_t ix
    cdef Py_ssize_t k
    cdef Py_ssize_t m = 0
    #cdef int n, ix
    #cdef int m = 0
    cdef float[:] alpha_stable_values = draw_random_stable(num_draws, alpha_view[0])

    for k in range(n_trials):
        delta_t_alpha = s * pow(delta_t, 1.0 / alpha_view[k])
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            # print(a)
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            # print(a)
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)

        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
        # Loop over samples
        for n in range(n_samples):
            y = (-1) * boundary_view[0] + (z_view[k] * 2 * (boundary_view[0]))  # reset starting position 
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index
            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y

            # Random walker
            while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t_particle <= deadline_tmp:
                y += (v_view[k] * delta_t) + (delta_t_alpha * alpha_stable_values[m])
                t_particle += delta_t
                ix += 1
                m += 1
                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y
                if m == num_draws:
                    alpha_stable_values = draw_random_stable(num_draws, alpha_view[k])
                    m = 0

            if smooth:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # Store rt
            choices_view[n, k, 0] = sign(y) # Store choice

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999
        
    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'metadata': {'v': v,
                                                            'a': a,
                                                            'z': z,
                                                            't': t,
                                                            'alpha': alpha,
                                                            's': s,
                                                            **boundary_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'levy_flexbound',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'possible_choices': [-1, 1],
                                                            'trajectory': traj,
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices,  'metadata': {'simulator': 'levy_flexbound', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Full DDM with flexible bounds --------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def full_ddm(np.ndarray[float, ndim = 1] v, # = 0,
             np.ndarray[float, ndim = 1] a, # = 1,
             np.ndarray[float, ndim = 1] z, # = 0.5,
             np.ndarray[float, ndim = 1] t, # = 0.0,
             np.ndarray[float, ndim = 1] sz, # = 0.05,
             np.ndarray[float, ndim = 1] sv, # = 0.1,
             np.ndarray[float, ndim = 1] st, # = 0.0,
             np.ndarray[float, ndim = 1] deadline,
             float s = 1,
             float delta_t = 0.001,
             float max_t = 20,
             int n_samples = 20000,
             int n_trials = 1,
             boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
             boundary_multiplicative = True,
             boundary_params = {},
             random_state = None,
             return_option = 'full',
             smooth = False,
             **kwargs):

    set_seed(random_state)
    # cdef int cov_length = np.max([v.size, a.size, w.size, t.size]).astype(int)
    # Param views
    #set_random_state(random_state)
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] sz_view = sz
    cdef float[:] sv_view = sv
    cdef float[:] st_view = st
    cdef float[:] deadline_view = deadline

    # Data-structs for trajectory storage
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y, t_particle, t_tmp, smooth_u, deadline_tmp
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float drift_increment = 0.0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    # Loop over trials
    for k in range(n_trials):
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            # print(a)
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            # print(a)
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
        # Loop over samples
        for n in range(n_samples):
            # initialize starting point
            y = ((-1) * boundary_view[0]) + (z_view[k] * 2.0 * (boundary_view[0]))  # reset starting position
            
            # get drift by random displacement of v 
            drift_increment = (v_view[k] + sv_view[k] * gaussian_values[m]) * delta_t
            t_tmp = t_view[k] + (2 * (random_uniform() - 0.5) * st_view[k])
            
            # apply uniform displacement on y
            y += 2 * (random_uniform() - 0.5) * sz_view[k]
            
            # increment m appropriately
            m += 1
            if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0
            
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index
            
            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y

            # Random walker
            while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t_particle <= deadline_tmp:
                y += drift_increment + (sqrt_st * gaussian_values[m])
                t_particle += delta_t
                ix += 1
                m += 1
                
                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            if smooth:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = t_particle + t_tmp + smooth_u # Store rt
            choices_view[n, k, 0] = np.sign(y) # Store choice

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999
    
    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'metadata': {'v': v,
                                                            'a': a,
                                                            'z': z,
                                                            't': t,
                                                            'sz': sz,
                                                            'sv': sv,
                                                            'st': st,
                                                            'deadline': deadline,
                                                            's': s,
                                                            **boundary_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'full_ddm',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'possible_choices': [-1, 1],
                                                            'trajectory': traj,
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices,  'metadata': {'simulator': 'full_ddm', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')

# -------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Full DDM with flexible bounds --------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_sdv(np.ndarray[float, ndim = 1] v,
            np.ndarray[float, ndim = 1] a,
            np.ndarray[float, ndim = 1] z,
            np.ndarray[float, ndim = 1] t,
            np.ndarray[float, ndim = 1] sv,
            np.ndarray[float, ndim = 1] deadline,
            float s = 1,
            float delta_t = 0.001,
            float max_t = 20,
            int n_samples = 20000,
            int n_trials = 1,
            boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
            boundary_multiplicative = True,
            boundary_params = {},
            random_state = None,
            return_option = 'full',
            smooth = False,
            **kwargs):

    set_seed(random_state)
    # Data-structs for trajectory storage
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:,:] traj_view = traj

    # Param views
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] t_view = t
    cdef float[:] sv_view = sv
    cdef float[:] deadline_view = deadline
    
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y, t_particle, smooth_u, deadline_tmp
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float drift_increment = 0.0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)

        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
        # Loop over samples
        for n in range(n_samples):
            # initialize starting point
            y = ((-1) * boundary_view[0]) + (z_view[k] * 2.0 * (boundary_view[0]))  # reset starting position
            
            # get drift by random displacement of v 
            drift_increment = (v_view[k] + sv_view[k] * gaussian_values[m]) * delta_t
            
            # increment m appropriately
            m += 1
            if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0
            
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index
            
            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y

            # Random walker
            while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t_particle <= deadline_tmp:
                y += drift_increment + (sqrt_st * gaussian_values[m])
                t_particle += delta_t
                ix += 1
                m += 1
                
                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y

                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            if smooth:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u # Store rt
            choices_view[n, k, 0] = np.sign(y) # Store choice

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return { 'rts': rts, 'choices': choices, 'metadata': {'v': v,
                                                            'a': a,
                                                            'z': z,
                                                            't': t,
                                                            'sv': sv,
                                                            'deadline': deadline,
                                                            's': s,
                                                            **boundary_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'ddm_sdv',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'possible_choices': [-1, 1],
                                                            'trajectory': traj,
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices,  'metadata': {'simulator': 'ddm_sdv', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')

# -------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Onstein-Uhlenbeck with flexible bounds -----------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ornstein_uhlenbeck(np.ndarray[float, ndim = 1] v, # drift parameter
                       np.ndarray[float, ndim = 1] a, # initial boundary separation
                       np.ndarray[float, ndim = 1] z, # starting point bias
                       np.ndarray[float, ndim = 1] g, # decay parameter
                       np.ndarray[float, ndim = 1] t,
                       np.ndarray[float, ndim = 1] deadline,
                       float s = 1, # standard deviation
                       float delta_t = 0.001, # size of timestep
                       float max_t = 20, # maximal time in trial
                       int n_samples = 20000, # number of samples from process
                       int n_trials = 1,
                       boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                       boundary_multiplicative = True,
                       boundary_params = {},
                       random_state = None,
                       return_option = 'full',
                       smooth = False,
                       **kwargs):

    set_seed(random_state)
    # Data-structs for trajectory storage
    traj = np.zeros((int(max_t / delta_t) + 1, 1), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:,:] traj_view = traj

    # Param views
    cdef float[:] v_view  = v
    cdef float[:] a_view = a
    cdef float[:] z_view = z
    cdef float[:] g_view = g
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline

    # Initializations
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE) # rt storage
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc) # choice storage

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = np.sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = s * delta_t_sqrt

    # Boundary Storage
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y, t_particle, smooth_u, deadline_tmp
    cdef Py_ssize_t n, ix, k
    cdef Py_ssize_t m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
        # Loop over samples
        for n in range(n_samples):
            y = (-1) * boundary_view[0] + (z_view[k] * 2 * boundary_view[0])
            t_particle = 0.0
            ix = 0

            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y

            # Random walker
            while y >= (-1) * boundary_view[ix] and y <= boundary_view[ix] and t_particle <= deadline_tmp:
                y += ((v_view[k] - (g_view[k] * y)) * delta_t) + sqrt_st * gaussian_values[m]
                t_particle += delta_t
                ix += 1
                m += 1

                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y

                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            if smooth:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u
            choices_view[n, k, 0] = sign(y)

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return { 'rts': rts, 'choices': choices, 'metadata': {'v': v,
                                                            'a': a,
                                                            'z': z,
                                                            'g': g,
                                                            't': t,
                                                            'deadline': deadline,
                                                            's': s,
                                                            **boundary_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'ornstein_uhlenbeck',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'possible_choices': [-1, 1],
                                                            'trajectory': traj,
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices,  'metadata': {'simulator': 'ornstein_uhlenbeck', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# --------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: RACE MODEL WITH N SAMPLES ----------------------------------

# Check if any of the particles in the race model have crossed
# @cythonboundscheck(False)
# @cythonwraparound(False)

# Function that checks boundary crossing of particles
cdef bint check_finished(float[:] particles, float boundary, int n):
    cdef int i # ,n
    #n = particles.shape[0]
    for i in range(n):
        if particles[i] > boundary:
            return True
    return False

#def test_check():
#    # Quick sanity check for the check_finished function
#    temp = np.random.normal(0,1, 10).astype(DTYPE)
#    cdef float[:] temp_view = temp
#    start = time()
#    [check_finished(temp_view, 3) for _ in range(1000000)]
#    print(check_finished(temp_view, 3))
#    end = time()
#    print("cython check: {}".format(start - end))
#    start = time()
#    [(temp > 3).any() for _ in range(1000000)]
#    end = time()
#    print("numpy check: {}".format(start - end))

# @cythonboundscheck(False)
# @cythonwraparound(False)
def race_model(np.ndarray[float, ndim = 2] v,  # np.array expected, one column of floats
               np.ndarray[float, ndim = 2] a, # initial boundary separation
               np.ndarray[float, ndim = 2] z, # np.array expected, one column of floats
               np.ndarray[float, ndim = 2] t, # for now we we don't allow t by choice
               np.ndarray[float, ndim = 2] s, # np.array expected, one column of floats
               np.ndarray[float, ndim = 1] deadline,
               float delta_t = 0.001, # time increment step
               float max_t = 20, # maximum rt allowed
               int n_samples = 2000, 
               int n_trials = 1,
               boundary_fun = None,
               boundary_multiplicative = True,
               boundary_params = {},
               random_state = None,
               return_option = 'full',
               smooth = False,
               **kwargs):

    set_seed(random_state)
    # Param views
    cdef float[:, :] v_view = v
    cdef float[:, :] z_view = z
    cdef float[:, :] a_view = a
    cdef float[:, :] t_view = t
    cdef float[:, :] s_view = s
    cdef float[:] deadline_view = deadline

    cdef float delta_t_sqrt = sqrt(delta_t)
    sqrt_st = delta_t_sqrt * s
    cdef float[:, :] sqrt_st_view = sqrt_st

    cdef int n_particles = v.shape[1]
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    cdef float[:, :, :] rts_view = rts
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef int[:, :, :] choices_view = choices
    
    particles = np.zeros((n_particles), dtype = DTYPE)
    cdef float [:] particles_view = particles

    # TD: Add Trajectory
    traj = np.zeros((int(max_t / delta_t) + 1, n_particles), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:, :] traj_view = traj    

    # Boundary storage
    cdef int num_steps = int((max_t / delta_t) + 1)

    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Initialize variables needed for for loop 
    cdef float t_particle, smooth_u, deadline_tmp
    cdef Py_ssize_t n, ix, j, k
    cdef Py_ssize_t m = 0

    cdef int num_draws = num_steps * n_particles
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k, 0], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k, 0], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k, 0])
        # Loop over samples
        for n in range(n_samples):
            for j in range(n_particles):
                particles_view[j] = z_view[k, j] * boundary_view[0] # Reset particle starting points
            
            t_particle = 0.0 # reset time
            ix = 0

            if n == 0:
                if k == 0:
                    for j in range(n_particles):
                        traj_view[0, j] = particles[j]

            # Random walker
            while not check_finished(particles_view, boundary_view[ix], n_particles) and t_particle <= deadline_tmp:
                for j in range(n_particles):
                    particles_view[j] += (v_view[k, j] * delta_t) + sqrt_st_view[k, j] * gaussian_values[m]
                    particles_view[j] = fmax(0.0, particles_view[j]) # Cut off particles at 0
                    m += 1
                    if m == num_draws:
                        m = 0
                        gaussian_values = draw_gaussian(num_draws)
                t_particle += delta_t
                ix += 1
                if n == 0:
                    if k == 0:
                        for j in range(n_particles):
                            traj_view[ix, j] = particles[j]

            if smooth:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n , k, 0] = t_particle + t[k, 0] + smooth_u # for now no t per choice option
            choices_view[n, k, 0] = np.argmax(particles)
            #rts_view[n, 0] = t + t[choices_view[n, 0]]

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999
            

        # Create some dics
        v_dict = {}
        z_dict = {}
        #t_dict = {}
        for i in range(n_particles):
            v_dict['v' + str(i)] = v[:, i]
            z_dict['z' + str(i)] = z[:, i]
            #t_dict['t_' + str(i)] = t[i] # for now no t by choice

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'metadata': {**v_dict,
                                                            'a': a, 
                                                            **z_dict,
                                                            't': t,
                                                            'deadline': deadline,
                                                            # **t_dict, # for now no t by choice
                                                            's': s,
                                                            **boundary_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'race_model',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'possible_choices': list(np.arange(0, n_particles, 1)),
                                                            'trajectory': traj,
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'metadata': {'simulator': 'race_model', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}

    else:
        raise ValueError('return_option must be either "full" or "minimal"')
    # -------------------------------------------------------------------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)

# Simulate (rt, choice) tuples from: Leaky Competing Accumulator Model -----------------------------
def lca(np.ndarray[float, ndim = 2] v, # drift parameters (np.array expect: one column of floats)
        np.ndarray[float, ndim = 2] a, # criterion height
        np.ndarray[float, ndim = 2] z, # initial bias parameters (np.array expect: one column of floats)
        np.ndarray[float, ndim = 2] g, # decay parameter
        np.ndarray[float, ndim = 2] b, # inhibition parameter
        np.ndarray[float, ndim = 2] t,
        np.ndarray[float, ndim = 2] s, # variance (can be one value or np.array of size as v and w)
        np.ndarray[float, ndim = 1] deadline,
        float delta_t = 0.001, # time-step size in simulator
        float max_t = 20, # maximal time
        int n_samples = 2000, # number of samples to produce
        int n_trials = 1,
        boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
        boundary_multiplicative = True,
        boundary_params = {},
        random_state = None,
        return_option = 'full',
        smooth = False,
        **kwargs):

    set_seed(random_state)
    # Param views
    cdef float[:, :] v_view = v
    cdef float[:, :] a_view = a
    cdef float[:, :] z_view = z
    cdef float[:, :] g_view = g
    cdef float[:, :] b_view = b
    cdef float[:, :] t_view = t
    cdef float[:, :] s_view = s
    cdef float[:] deadline_view = deadline

    # Trajectory
    cdef int n_particles = v.shape[1]
    traj = np.zeros((int(max_t / delta_t) + 1, n_particles), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    cdef float[:, :, :] rts_view = rts
    
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef int[:, :, :] choices_view = choices

    particles = np.zeros(n_particles, dtype = DTYPE)
    cdef float[:] particles_view = particles
    
    particles_reduced_sum = np.zeros(n_particles, dtype = DTYPE)
    cdef float[:] particles_reduced_sum_view = particles_reduced_sum
    
    cdef float delta_t_sqrt = sqrt(delta_t)
    sqrt_st = s * delta_t_sqrt
    cdef float[:, :] sqrt_st_view = sqrt_st
    
    cdef Py_ssize_t n, i, ix, k
    cdef Py_ssize_t m = 0
    cdef float t_par, particles_sum, smooth_u, deadline_tmp
    
    # Boundary storage                                                             
    cdef int num_steps = int((max_t / delta_t) + 2)

    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef int num_draws = num_steps * n_particles
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k, 0], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k, 0], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)

        deadline_tmp = min(max_t, deadline_view[k] - t_view[k, 0])
        for n in range(n_samples):
            # Reset particle starting points
            for i in range(n_particles):
                particles_view[i] = z_view[k, i] * boundary_view[0]
            
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index

            if n == 0:
                if k == 0:
                    for i in range(n_particles):
                        traj_view[0, i] = particles[i]

            while not check_finished(particles_view, boundary_view[ix], n_particles) and t_particle <= deadline_tmp:
                # calculate current sum over particle positions
                particles_sum = csum(particles_view)
                
                # update particle positions 
                for i in range(n_particles):
                    particles_reduced_sum_view[i] = (- 1) * particles_view[i] + particles_sum
                    particles_view[i] += ((v_view[k, i] - (g_view[k, 0] * particles_view[i]) - \
                            (b_view[k, 0] * particles_reduced_sum_view[i])) * delta_t) + (sqrt_st_view[k, i] * gaussian_values[m])
                    particles_view[i] = fmax(0.0, particles_view[i])
                    m += 1

                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0
                
                t_particle += delta_t # increment time
                ix += 1 # increment boundary index

                if n == 0:
                    if k == 0:
                        for i in range(n_particles):
                            traj_view[ix, i] = particles[i]

            if smooth:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0
        
            choices_view[n, k, 0] = np.argmax(particles) # store choices for sample n
            rts_view[n, k, 0] = t_particle + t_view[k, 0] + smooth_u # t[choices_view[n, 0]] # store reaction time for sample n

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999
        
    # Create some dics
    v_dict = {}
    z_dict = {}
    #t_dict = {}
    
    for i in range(n_particles):
        v_dict['v' + str(i)] = v[:, i]
        z_dict['z' + str(i)] = z[:, i]

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'metadata': {**v_dict,
                                                            'a': a,
                                                            **z_dict,
                                                            'g': g,
                                                            'b': b,
                                                            't': t,
                                                            'deadline': deadline,
                                                            's': s,
                                                            **boundary_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator' : 'lca',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'possible_choices': list(np.arange(0, n_particles, 1)),
                                                            'trajectory': traj,
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'metadata': {'simulator': 'lca', 
                                                             'possible_choices': [-1, 1],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -----------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_seq2(np.ndarray[float, ndim = 1] vh,
                       np.ndarray[float, ndim = 1] vl1,
                       np.ndarray[float, ndim = 1] vl2,
                       np.ndarray[float, ndim = 1] a,
                       np.ndarray[float, ndim = 1] zh,
                       np.ndarray[float, ndim = 1] zl1,
                       np.ndarray[float, ndim = 1] zl2,
                       np.ndarray[float, ndim = 1] t,
                       np.ndarray[float, ndim = 1] deadline,
                       float s = 1,
                       float delta_t = 0.001,
                       float max_t = 20,
                       int n_samples = 20000,
                       int n_trials = 1,
                       print_info = True,
                       boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                       boundary_multiplicative = True,
                       boundary_params = {},
                       random_state = None,
                       return_option = 'full',
                       smooth = False,
                       **kwargs):

    set_seed(random_state)
    # Param views
    cdef float[:] vh_view = vh
    cdef float[:] vl1_view = vl1
    cdef float[:] vl2_view = vl2
    cdef float[:] a_view = a
    cdef float[:] zh_view = zh
    cdef float[:] zl1_view = zl1
    cdef float[:] zl2_view = zl2
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    # TD: Add Trajectory
    traj = np.zeros((int(max_t / delta_t) + 1, 3), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:, :] traj_view = traj

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y_h, t_particle, t_particle1, t_particle2, y_l, y_l1, y_l2, smooth_u, deadline_tmp
    cdef Py_ssize_t n, ix, ix1, ix2, k
    cdef Py_ssize_t m = 0
    #cdef Py_ssize_t traj_id
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
        
        # Loop over samples
        for n in range(n_samples):
            t_particle = 0.0 # reset time
            ix = 0 # reset boundary index

            # Random walker 1
            y_h = (-1) * boundary_view[0] + (zh_view[k] * 2 * (boundary_view[0]))  # reset starting position 
            
            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y_h

            while y_h >= (-1) * boundary_view[ix] and y_h <= boundary_view[ix] and t_particle <= deadline_tmp:
                y_h += (vh_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                t_particle += delta_t
                ix += 1
                m += 1
                
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y_h

            # If we are already at maximum t, to generate a choice we just sample from a bernoulli
            if t_particle >= max_t:
                # High dim choice depends on position of particle
                if boundary_view[ix] <= 0:
                    if random_uniform() <= 0.5:
                        choices_view[n, k, 0] += 2
                elif random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                        choices_view[n, k, 0] += 2

                # Low dim choice random (didn't even get to process it if rt is at max after first choice)
                # so we just apply a priori bias
                if choices_view[n, k, 0] == 0:
                    if random_uniform() <= zl1_view[k]:
                        choices_view[n, k, 0] += 1
                else:
                    if random_uniform() <= zl2_view[k]:
                        choices_view[n, k, 0] += 1
                rts_view[n, k, 0] = t_particle
            else:
                # If boundary is negative (or 0) already, we flip a coin
                if boundary_view[ix] <= 0:
                    if random_uniform() <= 0.5:
                        choices_view[n, k, 0] += 2
                # Otherwise apply rule from above
                elif random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                    choices_view[n, k, 0] += 2

                y_l1 = (-1) * boundary_view[ix] + (zl1_view[k] * 2 * (boundary_view[ix]))
                y_l2 = (-1) * boundary_view[ix] + (zl2_view[k] * 2 * (boundary_view[ix])) 
                
                ix1 = ix
                t_particle1 = t_particle
                ix2 = ix
                t_particle2 = t_particle
                
                if choices_view[n, k, 0] == 0:
                    # ix1 = ix
                    # t_particle1 = t_particle
                    #v_l = vl1_view[k]
                    #z_l = zl1_view[k]
                    
                    # In case boundary is negative already, we flip a coin with bias determined by w_l_ parameter
                    if (y_l1 >= boundary_view[ix]) or (y_l1 <= ((-1) * boundary_view[ix])):
                        if random_uniform() < zl1_view[k]:
                            choices_view[n, k, 0] += 1
                    
                    if n == 0:
                        if k == 0:
                            traj_view[ix, 1] = y_l1
                else:
                    # ix2 = ix
                    # t_particle2 = t_particle
                    # v_l = vl2_view[k]
                    # z_l = zl2_view[k]
                    
                    # In case boundary is negative already, we flip a coin with bias determined by w_l_ parameter
                    if (y_l2 >= boundary_view[ix]) or (y_l2 <= ((-1) * boundary_view[ix])):
                        if random_uniform() < zl2_view[k]:
                            choices_view[n, k, 0] += 1

                    if n == 0:
                        if k == 0:
                            traj_view[ix, 2] = y_l2

                # Random walker low level (1)
                if (choices_view[n, k, 0] == 0) | ((n == 0) & (k == 0)):
                    while (y_l1 >= ((-1) * boundary_view[ix1])) and (y_l1 <= boundary_view[ix1]) and (t_particle1 <= deadline_tmp):
                        y_l1 += (vl1_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                        t_particle1 += delta_t
                        ix1 += 1
                        m += 1
                        if m == num_draws:
                            gaussian_values = draw_gaussian(num_draws)
                            m = 0

                        if n == 0:
                            if k == 0:
                                traj_view[ix1, 1] = y_l1

                # Random walker low level (2)
                if (choices_view[n, k, 0] == 2) | ((n == 0) & (k == 0)):
                    while (y_l2 >= ((-1) * boundary_view[ix2])) and (y_l2 <= boundary_view[ix2]) and (t_particle2 <= deadline_tmp):
                        y_l2 += (vl2_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                        t_particle2 += delta_t
                        ix2 += 1
                        m += 1
                        if m == num_draws:
                            gaussian_values = draw_gaussian(num_draws)
                            m = 0

                        if n == 0:
                            if k == 0:
                                traj_view[ix2, 2] = y_l2

                # Get back to single t_particle 
                if (choices_view[n, k, 0] == 0):
                    t_particle = t_particle1
                    ix = ix1
                    y_l = y_l1
                else:
                    t_particle = t_particle2
                    ix = ix2
                    y_l = y_l2

            if smooth:
                if t_particle == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif t_particle < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = t_particle + t_view[k] + smooth_u

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add one deterministically
            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically
            
            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if random_uniform() <= 0.5:
                    choices_view[n, k, 0] += 1
            # Otherwise apply rule from above
            elif random_uniform() <= ((y_l + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 1

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                    rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'metadata': {'vh': vh,
                                                            'vl1': vl1,
                                                            'vl2': vl2,
                                                            'a': a,
                                                            'zh': zh,
                                                            'zl1': zl1,
                                                            'zl2': zl2,
                                                            't': t,
                                                            'deadline': deadline,
                                                            's': s,
                                                            **boundary_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'ddm_flexbound',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'trajectory': traj,
                                                            'possible_choices': [0, 1, 2, 3],
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'metadata': {'simulator': 'ddm_flexbound', 
                                                             'possible_choices': [0, 1, 2, 3],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -----------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_par2(np.ndarray[float, ndim = 1] vh, 
                       np.ndarray[float, ndim = 1] vl1,
                       np.ndarray[float, ndim = 1] vl2,
                       np.ndarray[float, ndim = 1] a,
                       np.ndarray[float, ndim = 1] zh,
                       np.ndarray[float, ndim = 1] zl1,
                       np.ndarray[float, ndim = 1] zl2,
                       np.ndarray[float, ndim = 1] t,
                       np.ndarray[float, ndim = 1] deadline,
                       float s = 1,
                       float delta_t = 0.001,
                       float max_t = 20,
                       int n_samples = 20000,
                       int n_trials = 1,
                       print_info = True,
                       boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                       boundary_multiplicative = True,
                       boundary_params = {},
                       random_state = None,
                       return_option = 'full',
                       smooth = False,
                       **kwargs):

    set_seed(random_state)
    # Param views
    cdef float[:] vh_view = vh
    cdef float[:] vl1_view = vl1
    cdef float[:] vl2_view = vl2
    cdef float[:] a_view = a
    cdef float[:] zh_view = zh
    cdef float[:] zl1_view = zl1
    cdef float[:] zl2_view = zl2
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline

    # TD: Add trajectory --> Tricky here because the simulator is optimized to include only two instead of three particles (high dimension choice determines which low dimension choice will matter for ultimate choice)
    # TD: Add Trajectory
    traj = np.zeros((int(max_t / delta_t) + 1, 3), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:, :] traj_view = traj

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    rts_high = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    rts_low = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef float[:, :, :] rts_high_view = rts_high
    cdef float[:, :, :] rts_low_view = rts_low
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)

    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    cdef float y_h, y_l, y_l1, y_l2, v_l, v_l1, v_l2, t_h, t_l, t_l1, t_l2, smooth_u, deadline_tmp
    cdef Py_ssize_t n, ix, ix1, ix2, k
    cdef Py_ssize_t m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            # print(a)
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            # print(a)
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
        # Loop over samples
        for n in range(n_samples):
            t_h = 0.0 # reset time high dimension
            t_l1 = 0.0 # reset time low dimension (1)
            t_l2 = 0.0 # reset time low dimension (2)
            t_l = 0.0 # reset time low dimension (1 or 2)
            ix = 0 # reset boundary index

            # Initialize walkers
            y_h = (-1) * boundary_view[0] + (zh_view[k] * 2 * (boundary_view[0])) 

            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y_h

            # Random walks until y_h hits bound
            while (y_h >= (-1) * boundary_view[ix]) and (y_h <= boundary_view[ix]) and (t_h <= deadline_tmp):
                y_h += (vh_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                t_h += delta_t
                ix += 1
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y_h

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add 2 deterministically (correct)
            # y at lower bound --> choice_view[n, k, 0] stay the same deterministically (mistake)

            # if boundary is negative (or 0) already, we flip a coin 
            if boundary_view[ix] <= 0:
                if random_uniform() <= 0.5:
                    choices_view[n, k, 0] += 2
            # Otherwise apply rule from above
            elif random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2

            # Initialize lower level walkers
            y_l1 = (-1) * boundary_view[0] + (zl1_view[k] * 2 * (boundary_view[0])) 
            y_l2 = (-1) * boundary_view[0] + (zl2_view[k] * 2 * (boundary_view[0])) 

            # Random walker lower level (1)
            if (choices_view[n, k, 0] == 0) | ((n == 0) & (k == 0)):
                ix1 = 0
                while (y_l1 >= (-1) * boundary_view[ix1]) and (y_l1 <= boundary_view[ix1]) and (t_l1 <= deadline_tmp):
                    y_l1 += (vl1_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                    t_l1 += delta_t
                    ix1 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix1, 1] = y_l1

            # Random walker lower level (2)
            if (choices_view[n, k, 0] == 2) | ((n == 0) & (k == 0)):
                ix2 = 0
                while (y_l2 >= (-1) * boundary_view[ix2]) and (y_l2 <= boundary_view[ix2]) and (t_l2 <= deadline_tmp):
                    y_l2 += (vl2_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                    t_l2 += delta_t
                    ix2 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix2, 2] = y_l2

            # Consider only relevant lower-dim walker for final rt
            if (choices_view[n, k, 0] == 0):
                t_l = t_l1
                y_l = y_l1
                ix = ix1
            else:
                t_l = t_l2
                y_l = y_l2
                ix = ix2
            
            if smooth:
                if t_h == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif fmax(t_h, t_l) < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k] + smooth_u
            rts_high_view[n, k, 0] = t_h + t_view[k]
            rts_low_view[n, k, 0] = t_l + t_view[k]

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add one deterministically
            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically
            
            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if random_uniform() <= 0.5:
                    choices_view[n, k, 0] += 1
            # Otherwise apply rule from above
            elif random_uniform() <= ((y_l + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 1

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'rts_low': rts_low, 'rts_high': rts_high, 
                'metadata': {'vh': vh,
                            'vl1': vl1,
                            'vl2': vl2,
                            'a': a,
                            'zh': zh,
                            'zl1': zl1,
                            'zl2': zl2,
                            't': t,
                            'deadline': deadline,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'n_trials': n_trials,
                            'simulator': 'ddm_flexbound',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [0, 1, 2, 3],
                            'trajectory': traj,
                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'rts_low': rts_low, 'rts_high': rts_high, 
                'metadata': {'simulator': 'ddm_flexbound', 
                             'possible_choices': [0, 1, 2, 3],
                             'boundary_fun_type': boundary_fun.__name__,
                             'n_samples': n_samples,
                             'n_trials': n_trials,
                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -----------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_mic2_ornstein(np.ndarray[float, ndim = 1] vh, 
                                np.ndarray[float, ndim = 1] vl1,
                                np.ndarray[float, ndim = 1] vl2,
                                np.ndarray[float, ndim = 1] a,
                                np.ndarray[float, ndim = 1] zh,
                                np.ndarray[float, ndim = 1] zl1,
                                np.ndarray[float, ndim = 1] zl2,
                                np.ndarray[float, ndim = 1] d, # damper (1 --> no drift on low level until high level done, 0 --> full drift on low level)
                                np.ndarray[float, ndim = 1] g, # inhibition parameter for the low dim choice procress while high dim is running
                                np.ndarray[float, ndim = 1] t,
                                np.ndarray[float, ndim = 1] deadline,
                                np.ndarray[float, ndim = 1] s_pre_high_level_choice,
                                float s = 1.0,
                                float delta_t = 0.001,
                                float max_t = 20,
                                int n_samples = 20000,
                                int n_trials = 1,
                                print_info = True,
                                boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                                boundary_multiplicative = True,
                                boundary_params = {},
                                random_state = None,
                                return_option = 'full',
                                smooth = False,
                                **kwargs):

    set_seed(random_state)
    # Param views
    cdef float[:] vh_view = vh
    cdef float[:] vl1_view = vl1
    cdef float[:] vl2_view = vl2
    cdef float[:] a_view = a
    cdef float[:] zh_view = zh
    cdef float[:] zl1_view = zl1
    cdef float[:] zl2_view = zl2
    cdef float[:] d_view = d
    cdef float[:] g_view = g
    cdef float[:] t_view = t
    cdef float[:] s_pre_high_level_choice_view = s_pre_high_level_choice
    cdef float[:] deadline_view = deadline

    # TD: Add trajectory --> same issue as with par2 model above... might need to make a separate simulator for trajectories
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    rts_low = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    rts_high = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)

    cdef float[:, :, :] rts_view = rts
    cdef float[:, :, :] rts_high_view = rts_high
    cdef float[:, :, :] rts_low_view = rts_low
    cdef int[:, :, :] choices_view = choices

    traj = np.zeros((int(max_t / delta_t) + 1, 3), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:, :] traj_view = traj

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Y particle trace
    bias_trace_l1 = np.zeros(num_draws, dtype = DTYPE)
    bias_trace_l2 = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] bias_trace_l1_view = bias_trace_l1
    cdef float[:] bias_trace_l2_view = bias_trace_l2

    cdef float y_h, y_l, y_l1, y_l2
    cdef float v_l, v_l1, v_l2,
    cdef float t_h, t_l, t_l1, t_l2, smooth_u, deadline_tmp
    cdef Py_ssize_t n, ix, ix1, ix2, ix_l, ix_tmp, ix1_tmp, ix2_tmp, k
    cdef Py_ssize_t m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
        # Loop over samples
        for n in range(n_samples):
            choices_view[n, k, 0] = 0 # reset choice
            t_h = 0 # reset time high dimension
            t_l = 0 # reset time low dimension
            t_l1 = 0 # reset time low dimension (1)
            t_l2 = 0 # reset time low dimension (2)
            ix = 0 # reset boundary index
            ix1 = 0 # reset boundary index (1)
            ix2 = 0 # reset boundary index (2)

            # Initialize walkers
            # Particle
            y_h = (-1) * boundary_view[0] + (zh_view[k] * 2 * (boundary_view[0])) 
            # Relative particle position (used as resource allocator for low dim choice)
            bias_trace_l2_view[0] = ((y_h + boundary_view[0]) / (2 * boundary_view[0]))
            bias_trace_l1_view[0] = 1.0 - bias_trace_l2_view[0]

            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y_h

            # Random walks until y_h hits bound
            while (y_h >= ((-1) * boundary_view[ix])) and ((y_h <= boundary_view[ix])) and (t_h <= deadline_tmp):
                y_h += (vh_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                bias_trace_l2_view[ix] = ((y_h + boundary_view[ix]) / (2 * boundary_view[ix]))
                bias_trace_l1_view[ix] = 1.0 - bias_trace_l2_view[ix]
                t_h += delta_t
                ix += 1
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y_h

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add 2 deterministically
            # y at lower bound --> choice_view[n, k, 0] stay the same deterministically

            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if random_uniform() <= 0.5:
                    choices_view[n, k, 0] += 2
            # Otherwise, apply rule from above
            elif random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2

            y_l2 = (- 1) * boundary_view[0] + (zl2_view[k] * 2 * (boundary_view[0]))
            y_l1 = (- 1) * boundary_view[0] + (zl1_view[k] * 2 * (boundary_view[0]))
            
            if choices_view[n, k, 0] == 0:
                 # Fill bias tracea until max_rt reached
                ix1_tmp = ix + 1
                while ix1_tmp < num_draws:
                    bias_trace_l2_view[ix1_tmp] = 0.0
                    bias_trace_l1_view[ix1_tmp] = 1.0
                    ix1_tmp += 1

            else: # Store intermediate choice
                # Fill bias tracea until max_rt reached
                ix2_tmp = ix + 1
                while ix2_tmp < num_draws:
                    bias_trace_l2_view[ix2_tmp] = 1.0
                    bias_trace_l1_view[ix2_tmp] = 0.0
                    ix2_tmp += 1

            # lower level random walker (1)
            if (choices_view[n, k, 0] == 0) | ((n == 0) & (k == 0)):
                while (y_l1 >= ((-1) * boundary_view[ix1])) and (y_l1 <= boundary_view[ix1]) and (t_l1 <= deadline_tmp):
                    if (bias_trace_l1_view[ix1] < 1) and (bias_trace_l1_view[ix1] > 0):
                        # main propagation if bias_trace is between 0 and 1 (high level choice is not yet made)
                        y_l1 += (((vl1_view[k] * bias_trace_l1_view[ix1] * (1 - d_view[k])) - (g_view[k] * y_l1)) * delta_t)
                        # add gaussian displacement
                        y_l1 += (sqrt_st * gaussian_values[m]) * s_pre_high_level_choice_view[k] 
                    else:
                        # main propagation if bias_trace is not between 0 and 1 (high level choice is already made)
                        y_l1 += (vl1_view[k] * delta_t)
                        # add gaussian displacement
                        y_l1 += (sqrt_st * gaussian_values[m])
                    
                    
                    # propagate time and indices
                    t_l1 += delta_t
                    ix1 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix1, 1] = y_l1

            # lower level random walker (2)
            if (choices_view[n, k, 0] == 2) | ((n == 0) & (k == 0)):
                while (y_l2 >= ((-1) * boundary_view[ix2])) and (y_l2 <= boundary_view[ix2]) and (t_l2 <= deadline_tmp):
                    if (bias_trace_l2_view[ix2] < 1) and (bias_trace_l2_view[ix2] > 0):
                        # main propagation if bias_trace is between 0 and 1 (high level choice is not yet made)
                        y_l2 += (((vl2_view[k] * bias_trace_l2_view[ix2] * (1 - d_view[k])) - (g_view[k] * y_l2)) * delta_t)
                        # add gaussian displacement
                        y_l2 += (sqrt_st * gaussian_values[m]) * s_pre_high_level_choice_view[k] 
                    else:
                        # main propagation if bias_trace is not between 0 and 1 (high level choice is already made)
                        y_l2 += (vl2_view[k] * delta_t)
                        # add gaussian displacement
                        y_l2 += (sqrt_st * gaussian_values[m])
                    
                    # propagate time and indices
                    t_l2 += delta_t
                    ix2 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix2, 2] = y_l2

            # Get back to single y_l and t_l
            if (choices_view[n, k, 0] == 0):
                t_l = t_l1
                y_l = y_l1
                ix_l = ix1
            else:
                t_l = t_l2
                y_l = y_l2
                ix_l = ix2

            if smooth:
                if t_h == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif fmax(t_h, t_l) < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]
            rts_high_view[n, k, 0] = t_h + t_view[k]
            rts_low_view[n, k, 0] = t_l + t_view[k]

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add one deterministically
            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically

            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if random_uniform() <= 0.5:
                    choices_view[n, k, 0] += 1
            # Otherwise apply rule from above
            elif random_uniform() <= ((y_l + boundary_view[ix_l]) / (2 * boundary_view[ix_l])):
                choices_view[n, k, 0] += 1

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
                'metadata': {'vh': vh,
                            'vl1': vl1,
                            'vl2': vl2,
                            'a': a,
                            'zh': zh,
                            'zl1': zl1,
                            'zl2': zl2,
                            'd': d,
                            't': t,
                            'deadline': deadline,
                            's_pre_high_level_choice': s_pre_high_level_choice,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'n_trials': n_trials,
                            'simulator': 'ddm_flexbound_mic2_adj',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [0, 1, 2, 3],
                            'trajectory': traj,
                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
                'metadata': {'simulator': 'ddm_flexbound_mic2_adj', 
                             'possible_choices': [0, 1, 2, 3],
                             'boundary_fun_type': boundary_fun.__name__,
                             'n_samples': n_samples,
                             'n_trials': n_trials,
                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# ----------------------------------------------------------------------------------------------------


# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_mic2_multinoise(np.ndarray[float, ndim = 1] vh, 
                                  np.ndarray[float, ndim = 1] vl1,
                                  np.ndarray[float, ndim = 1] vl2,
                                  np.ndarray[float, ndim = 1] a,
                                  np.ndarray[float, ndim = 1] zh,
                                  np.ndarray[float, ndim = 1] zl1,
                                  np.ndarray[float, ndim = 1] zl2,
                                  np.ndarray[float, ndim = 1] d, # damper (1 --> no drift on low level until high level done, 0 --> full drift on low level)
                                  np.ndarray[float, ndim = 1] t,
                                  np.ndarray[float, ndim = 1] deadline,
                                  float s = 1.0,
                                  float delta_t = 0.001,
                                  float max_t = 20,
                                  int n_samples = 20000,
                                  int n_trials = 1,
                                  print_info = True,
                                  boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                                  boundary_multiplicative = True,
                                  boundary_params = {},
                                  random_state = None,
                                  return_option = 'full',
                                  smooth = False,
                                  **kwargs):

    set_seed(random_state)
    # Param views
    cdef float[:] vh_view = vh
    cdef float[:] vl1_view = vl1
    cdef float[:] vl2_view = vl2
    cdef float[:] a_view = a
    cdef float[:] zh_view = zh
    cdef float[:] zl1_view = zl1
    cdef float[:] zl2_view = zl2
    cdef float[:] d_view = d
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline

    # TD: Add trajectory --> same issue as with par2 model above... might need to make a separate simulator for trajectories
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    rts_low = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    rts_high = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)

    cdef float[:, :, :] rts_view = rts
    cdef float[:, :, :] rts_high_view = rts_high
    cdef float[:, :, :] rts_low_view = rts_low
    cdef int[:, :, :] choices_view = choices

    traj = np.zeros((int(max_t / delta_t) + 1, 3), dtype = DTYPE)
    traj[:, :] = -999 
    cdef float[:, :] traj_view = traj

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Y particle trace
    bias_trace_l1 = np.zeros(num_draws, dtype = DTYPE)
    bias_trace_l2 = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] bias_trace_l1_view = bias_trace_l1
    cdef float[:] bias_trace_l2_view = bias_trace_l2

    cdef float y_h, y_l, y_l1, y_l2
    cdef float v_l, v_l1, v_l2
    cdef float t_h, t_l, t_l1, t_l2, smooth_u, deadline_tmp
    cdef Py_ssize_t n, ix, ix1, ix2, ix_l, ix_tmp, ix1_tmp, ix2_tmp, k
    cdef Py_ssize_t m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
        # Loop over samples
        for n in range(n_samples):
            choices_view[n, k, 0] = 0 # reset choice
            t_h = 0 # reset time high dimension
            t_l = 0 # reset time low dimension
            t_l1 = 0 # reset time low dimension (1)
            t_l2 = 0 # reset time low dimension (2)
            ix = 0 # reset boundary index
            ix1 = 0 # reset boundary index (1)
            ix2 = 0 # reset boundary index (2)

            # Initialize walkers
            # Particle
            y_h = (-1) * boundary_view[0] + (zh_view[k] * 2 * (boundary_view[0])) 
            # Relative particle position (used as resource allocator for low dim choice)
            bias_trace_l2_view[0] = ((y_h + boundary_view[0]) / (2 * boundary_view[0]))
            bias_trace_l1_view[0] = 1.0 - bias_trace_l2_view[0]

            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y_h

            # Random walks until y_h hits bound
            while (y_h >= ((-1) * boundary_view[ix])) and ((y_h <= boundary_view[ix])) and (t_h <= deadline_tmp):
                y_h += (vh_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                bias_trace_l2_view[ix] = ((y_h + boundary_view[ix]) / (2 * boundary_view[ix]))
                bias_trace_l1_view[ix] = 1.0 - bias_trace_l2_view[ix]
                t_h += delta_t
                ix += 1
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

                if n == 0:
                    if k == 0:
                        traj_view[ix, 0] = y_h

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add 2 deterministically
            # y at lower bound --> choice_view[n, k, 0] stay the same deterministically

            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if random_uniform() <= 0.5:
                    choices_view[n, k, 0] += 2
            # Otherwise, apply rule from above
            elif random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2

            y_l2 = (- 1) * boundary_view[0] + (zl2_view[k] * 2 * (boundary_view[0]))
            y_l1 = (- 1) * boundary_view[0] + (zl1_view[k] * 2 * (boundary_view[0]))

            if choices_view[n, k, 0] == 0:
                 # Fill bias tracea until max_rt reached
                ix1_tmp = ix + 1
                while ix1_tmp < num_draws:
                    bias_trace_l2_view[ix1_tmp] = 0.0
                    bias_trace_l1_view[ix1_tmp] = 1.0
                    ix1_tmp += 1

            else: # Store intermediate choice
                # Fill bias tracea until max_rt reached
                ix2_tmp = ix + 1
                while ix2_tmp < num_draws:
                    bias_trace_l2_view[ix2_tmp] = 1.0
                    bias_trace_l1_view[ix2_tmp] = 0.0
                    ix2_tmp += 1

            # lower level random walker (1)
            if (choices_view[n, k, 0] == 0) | ((n == 0) & (k == 0)):
                while (y_l1 >= ((-1) * boundary_view[ix1])) and (y_l1 <= boundary_view[ix1]) and (t_l1 <= deadline_tmp):
                    if (bias_trace_l1_view[ix1] < 1) and (bias_trace_l1_view[ix1] > 0):
                        # main propagation if bias_trace is between 0 and 1 (high level choice is not yet made)
                        y_l1 += (((vl1_view[k] * bias_trace_l1_view[ix1] * (1 - d_view[k]))) * delta_t)
                        # add gaussian displacement
                        # we multiply by bias_trace_view to make low level variance depend on high level trace
                        y_l1 += (sqrt_st * gaussian_values[m]) * bias_trace_l1_view[ix1]
                    else:
                        # main propagation if bias_trace is not between 0 and 1 (high level choice is already made)
                        y_l1 += (vl1_view[k] * delta_t)
                        # add gaussian displacement
                        y_l1 += (sqrt_st * gaussian_values[m])
                    
                    
                    # propagate time and indices
                    t_l1 += delta_t
                    ix1 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix1, 1] = y_l1

            # lower level random walker (2)
            if (choices_view[n, k, 0] == 2) | ((n == 0) & (k == 0)):
                while (y_l2 >= ((-1) * boundary_view[ix2])) and (y_l2 <= boundary_view[ix2]) and (t_l2 <= deadline_tmp):
                    if (bias_trace_l2_view[ix2] < 1) and (bias_trace_l2_view[ix2] > 0):
                        # main propagation if bias_trace is between 0 and 1 (high level choice is not yet made)
                        y_l2 += (((vl2_view[k] * bias_trace_l2_view[ix2] * (1 - d_view[k]))) * delta_t)
                        # add gaussian displacement
                        # we multiply by bias_trace_view to make low level variance depend on high level trace
                        y_l2 += (sqrt_st * gaussian_values[m]) * bias_trace_l2_view[ix2]
                    else:
                        # main propagation if bias_trace is not between 0 and 1 (high level choice is already made)
                        y_l2 += (vl2_view[k] * delta_t)
                        # add gaussian displacement
                        y_l2 += (sqrt_st * gaussian_values[m])
                    
                    
                    # propagate time and indices
                    t_l2 += delta_t
                    ix2 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix2, 2] = y_l2

            # Get back to single y_l and t_l
            if (choices_view[n, k, 0] == 0):
                t_l = t_l1
                y_l = y_l1
                ix_l = ix1
            else:
                t_l = t_l2
                y_l = y_l2
                ix_l = ix2

            if smooth:
                if t_h == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif fmax(t_h, t_l) < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]
            rts_high_view[n, k, 0] = t_h + t_view[k]
            rts_low_view[n, k, 0] = t_l + t_view[k]

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add one deterministically
            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically

            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if random_uniform() <= 0.5:
                    choices_view[n, k, 0] += 1
            # Otherwise apply rule from above
            elif random_uniform() <= ((y_l + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 1

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
                'metadata': {'vh': vh,
                            'vl1': vl1,
                            'vl2': vl2,
                            'a': a,
                            'zh': zh,
                            'zl1': zl1,
                            'zl2': zl2,
                            'd': d,
                            't': t,
                            'deadline': deadline,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'n_trials': n_trials,
                            'simulator': 'ddm_flexbound_mic2_adj',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [0, 1, 2, 3],
                            'trajectory': traj,
                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
                'metadata': {'simulator': 'ddm_flexbound_mic2_adj', 
                             'possible_choices': [0, 1, 2, 3],
                             'boundary_fun_type': boundary_fun.__name__,
                             'n_samples': n_samples,
                             'n_trials': n_trials,
                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# ----------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_mic2_ornstein_multinoise(np.ndarray[float, ndim = 1] vh, 
                                           np.ndarray[float, ndim = 1] vl1,
                                           np.ndarray[float, ndim = 1] vl2,
                                           np.ndarray[float, ndim = 1] a,
                                           np.ndarray[float, ndim = 1] zh,
                                           np.ndarray[float, ndim = 1] zl1,
                                           np.ndarray[float, ndim = 1] zl2,
                                           np.ndarray[float, ndim = 1] d, # damper (1 --> no drift on low level until high level done, 0 --> full drift on low level)
                                           np.ndarray[float, ndim = 1] g, # inhibition parameter for the low dim choice procress while high dim is running
                                           np.ndarray[float, ndim = 1] t,
                                           np.ndarray[float, ndim = 1] deadline,
                                           float s = 1.0,
                                           float delta_t = 0.001,
                                           float max_t = 20,
                                           int n_samples = 20000,
                                           int n_trials = 1,
                                           print_info = True,
                                           boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                                           boundary_multiplicative = True,
                                           boundary_params = {},
                                           random_state = None,
                                           return_option = 'full',
                                           smooth = False,
                                           **kwargs):

    set_seed(random_state)
    # Param views
    cdef float[:] vh_view = vh
    cdef float[:] vl1_view = vl1
    cdef float[:] vl2_view = vl2
    cdef float[:] a_view = a
    cdef float[:] zh_view = zh
    cdef float[:] zl1_view = zl1
    cdef float[:] zl2_view = zl2
    cdef float[:] d_view = d
    cdef float[:] g_view = g
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline

    # TD: Add trajectory --> same issue as with par2 model above... might need to make a separate simulator for trajectories
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    rts_low = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    rts_high = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)

    cdef float[:, :, :] rts_view = rts
    cdef float[:, :, :] rts_high_view = rts_high
    cdef float[:, :, :] rts_low_view = rts_low
    cdef int[:, :, :] choices_view = choices

    traj = np.zeros((int(max_t / delta_t) + 1, 3), dtype = DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Y particle trace
    bias_trace_l1 = np.zeros(num_draws, dtype = DTYPE)
    bias_trace_l2 = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] bias_trace_l1_view = bias_trace_l1
    cdef float[:] bias_trace_l2_view = bias_trace_l2

    cdef float y_h, y_l, y_l1, y_l2
    cdef float v_l, v_l1, v_l2, 
    cdef float t_h, t_l, t_l1, t_l2, smooth_u, deadline_tmp
    cdef Py_ssize_t n, ix, ix1, ix2, ix_l, ix_tmp, ix1_tmp, ix2_tmp, k
    cdef Py_ssize_t m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
        # Loop over samples
        for n in range(n_samples):
            choices_view[n, k, 0] = 0 # reset choice
            t_h = 0 # reset time high dimension
            t_l = 0 # reset time low dimension
            t_l1 = 0 # reset time low dimension (1)
            t_l2 = 0 # reset time low dimension (2)
            ix = 0 # reset boundary index
            ix1 = 0 # reset boundary index (1)
            ix2 = 0 # reset boundary index (2)

            # Initialize walkers
            # Particle
            y_h = (-1) * boundary_view[0] + (zh_view[k] * 2 * (boundary_view[0])) 
            # Relative particle position (used as resource allocator for low dim choice)
            bias_trace_l2_view[0] = ((y_h + boundary_view[0]) / (2 * boundary_view[0]))
            bias_trace_l1_view[0] = 1.0 - bias_trace_l2_view[0]

            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y_h

            # Random walks until y_h hits bound
            while (y_h >= ((-1) * boundary_view[ix])) and ((y_h <= boundary_view[ix])) and (t_h <= deadline_tmp):
                y_h += (vh_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                bias_trace_l2_view[ix] = ((y_h + boundary_view[ix]) / (2 * boundary_view[ix]))
                bias_trace_l1_view[ix] = 1.0 - bias_trace_l2_view[ix]
                t_h += delta_t
                ix += 1
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add 2 deterministically
            # y at lower bound --> choice_view[n, k, 0] stay the same deterministically

            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if random_uniform() <= 0.5:
                    choices_view[n, k, 0] += 2
            # Otherwise, apply rule from above
            elif random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2
           
            y_l2 = (- 1) * boundary_view[0] + (zl2_view[k] * 2 * (boundary_view[0]))
            y_l1 = (- 1) * boundary_view[0] + (zl1_view[k] * 2 * (boundary_view[0]))
            
            if choices_view[n, k, 0] == 0:
                 # Fill bias tracea until max_rt reached
                ix1_tmp = ix + 1
                while ix1_tmp < num_draws:
                    bias_trace_l2_view[ix1_tmp] = 0.0
                    bias_trace_l1_view[ix1_tmp] = 1.0
                    ix1_tmp += 1

            else: # Store intermediate choice
                # Fill bias tracea until max_rt reached
                ix2_tmp = ix + 1
                while ix2_tmp < num_draws:
                    bias_trace_l2_view[ix2_tmp] = 1.0
                    bias_trace_l1_view[ix2_tmp] = 0.0
                    ix2_tmp += 1

            # lower level random walker (1)
            if (choices_view[n, k, 0] == 0) | ((n == 0) & (k == 0)):
                while (y_l1 >= ((-1) * boundary_view[ix1])) and (y_l1 <= boundary_view[ix1]) and (t_l1 <= deadline_tmp):
                    if (bias_trace_l1_view[ix1] < 1) and (bias_trace_l1_view[ix1] > 0):
                        # main propagation if bias_trace is between 0 and 1 (high level choice is not yet made)
                        y_l1 += (((vl1_view[k] * bias_trace_l1_view[ix1] * (1 - d_view[k])) - (g_view[k] * y_l1)) * delta_t)
                        # add gaussian displacement
                        y_l1 += (sqrt_st * gaussian_values[m]) * bias_trace_l1_view[ix1] 
                    else:
                        # main propagation if bias_trace is not between 0 and 1 (high level choice is already made)
                        y_l1 += (vl1_view[k] * delta_t)
                        # add gaussian displacement
                        y_l1 += (sqrt_st * gaussian_values[m])
                    
                    
                    # propagate time and indices
                    t_l1 += delta_t
                    ix1 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix1, 1] = y_l1

            # lower level random walker (2)
            if (choices_view[n, k, 0] == 2) | ((n == 0) & (k == 0)):
                while (y_l2 >= ((-1) * boundary_view[ix2])) and (y_l2 <= boundary_view[ix2]) and (t_l2 <= deadline_tmp):
                    if (bias_trace_l2_view[ix2] < 1) and (bias_trace_l2_view[ix2] > 0):
                        # main propagation if bias_trace is between 0 and 1 (high level choice is not yet made)
                        y_l2 += (((vl2_view[k] * bias_trace_l2_view[ix2] * (1 - d_view[k])) - (g_view[k] * y_l2)) * delta_t)
                        # add gaussian displacement
                        y_l2 += (sqrt_st * gaussian_values[m]) * bias_trace_l2_view[ix2] 
                    else:
                        # main propagation if bias_trace is not between 0 and 1 (high level choice is already made)
                        y_l2 += (vl2_view[k] * delta_t)
                        # add gaussian displacement
                        y_l2 += (sqrt_st * gaussian_values[m])
                    
                    # propagate time and indices
                    t_l2 += delta_t
                    ix2 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix2, 2] = y_l2

            # Get back to single y_l and t_l
            if (choices_view[n, k, 0] == 0):
                t_l = t_l1
                y_l = y_l1
                ix_l = ix1
            else:
                t_l = t_l2
                y_l = y_l2
                ix_l = ix2

            if smooth:
                if t_h == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif fmax(t_h, t_l) < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]
            rts_high_view[n, k, 0] = t_h + t_view[k]
            rts_low_view[n, k, 0] = t_l + t_view[k]

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add one deterministically
            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically

            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if random_uniform() <= 0.5:
                    choices_view[n, k, 0] += 1
            # Otherwise apply rule from above
            elif random_uniform() <= ((y_l + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 1

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
                'metadata': {'vh': vh,
                            'vl1': vl1,
                            'vl2': vl2,
                            'a': a,
                            'zh': zh,
                            'zl1': zl1,
                            'zl2': zl2,
                            'd': d,
                            't': t,
                            'deadline': deadline,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'n_trials': n_trials,
                            'simulator': 'ddm_flexbound_mic2_adj',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [0, 1, 2, 3],
                            'trajectory': traj,
                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
                'metadata': {'simulator': 'ddm_flexbound_mic2_adj', 
                             'possible_choices': [0, 1, 2, 3],
                             'boundary_fun_type': boundary_fun.__name__,
                             'n_samples': n_samples,
                             'n_trials': n_trials,
                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# ----------------------------------------------------------------------------------------------------


# Simulate (rt, choice) tuples from: Vanilla LBA Model without ndt -----------------------------
def lba_vanilla(np.ndarray[float, ndim = 2] v, 
        np.ndarray[float, ndim = 2] a, 
        np.ndarray[float, ndim = 2] z, 
        np.ndarray[float, ndim = 1] deadline,
        float sd, # std dev of Normal from where we sample vs
        float ndt = 0, # ndt is supposed to be 0 by default because of parameter identifiability issues
        int nact = 3,
        int n_samples = 2000,
        int n_trials = 1,
        float max_t = 20,
        **kwargs
        ):

    # v_t = np.random.normal(v, sd)
    # print(len(z), nact, np.array([z]*nact).transpose().shape)
    # z_t = np.random.uniform(np.zeros((len(z), nact)), np.array([z]*nact).transpose(), (len(z), nact))

    # Param views
    cdef float[:, :] v_view = v
    cdef float[:, :] a_view = a
    cdef float[:, :] z_view = z

    cdef float[:] deadline_view = deadline

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    cdef float[:, :, :] rts_view = rts
    
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef int[:, :, :] choices_view = choices
    
    cdef Py_ssize_t n, k, i

    for k in range(n_trials):
        
        for n in range(n_samples):
            zs = np.random.uniform(0, z_view[k], nact)

            vs = np.abs(np.random.normal(v_view[k], sd)) # np.abs() to avoid negative vs

            x_t = ([a_view[k]]*nact - zs)/vs
        
            choices_view[n, k, 0] = np.argmin(x_t) # store choices for sample n
            rts_view[n, k, 0] = np.min(x_t) + ndt  # store reaction time for sample n

            # If the rt exceeds the deadline, set rt to -999
            if rts_view[n, k, 0] >= deadline_view[k]:
                rts_view[n, k, 0] = -999
        

    v_dict = {}    
    for i in range(nact):
        v_dict['v_' + str(i)] = v[:, i]

    return {'rts': rts, 'choices': choices, 'metadata': {**v_dict,
                                                         'a': a,
                                                         'z': z,
                                                         'deadline': deadline,
                                                         'sd': sd,
                                                         'n_samples': n_samples,
                                                         'simulator' : 'lba_vanilla',
                                                         'possible_choices': list(np.arange(0, nact, 1)),
                                                         'max_t': max_t,
                                                         }}



# Simulate (rt, choice) tuples from: Collapsing bound angle LBA Model -----------------------------
def lba_angle(np.ndarray[float, ndim = 2] v, 
        np.ndarray[float, ndim = 2] a, 
        np.ndarray[float, ndim = 2] z,  
        np.ndarray[float, ndim = 2] theta,
        np.ndarray[float, ndim = 1] deadline,
        float sd, # std dev 
        float ndt = 0, # ndt is supposed to be 0 by default because of parameter identifiability issues
        int nact = 3,
        int n_samples = 2000,
        int n_trials = 1,
        float max_t = 20,
        **kwargs
        ):

    # Param views
    cdef float[:, :] v_view = v
    cdef float[:, :] a_view = a
    cdef float[:, :] z_view = z
    cdef float[:, :] theta_view = theta

    cdef float[:] deadline_view = deadline

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    cdef float[:, :, :] rts_view = rts
    
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef int[:, :, :] choices_view = choices
    
    cdef Py_ssize_t n, k, i

    for k in range(n_trials):
        
        for n in range(n_samples):
            zs = np.random.uniform(0, z_view[k], nact)

            vs = np.abs(np.random.normal(v_view[k], sd)) # np.abs() to avoid negative vs
            x_t = ([a_view[k]]*nact - zs)/(vs + np.tan(theta_view[k, 0]))
        
            choices_view[n, k, 0] = np.argmin(x_t) # store choices for sample n
            rts_view[n, k, 0] = np.min(x_t) + ndt # store reaction time for sample n

            # If the rt exceeds the deadline, set rt to -999
            if rts_view[n, k, 0] >= deadline_view[k]:
                rts_view[n, k, 0] = -999

            # if np.min(x_t) <= 0:
            #     print("\n ssms sim error: ", a[k], zs, vs, np.tan(theta[k]))
    
    v_dict = {}  
    for i in range(nact):
        v_dict['v_' + str(i)] = v[:, i]

    return {'rts': rts, 'choices': choices, 'metadata': {**v_dict,
                                                         'a': a,
                                                         'z': z,
                                                         'theta': theta,
                                                         'deadline': deadline,
                                                         'sd': sd,
                                                         'n_samples': n_samples,
                                                         'simulator' : 'lba_angle',
                                                         'possible_choices': list(np.arange(0, nact, 1)),
                                                         'max_t': max_t,
                                                         }}


# Simulate (rt, choice) tuples from: RLWM LBA Race Model without ndt -----------------------------
def rlwm_lba_race(np.ndarray[float, ndim = 2] v_RL, # RL drift parameters (np.array expect: one column of floats)
        np.ndarray[float, ndim = 2] v_WM, # WM drift parameters (np.array expect: one column of floats)
        np.ndarray[float, ndim = 2] a, # criterion height
        np.ndarray[float, ndim = 2] z, # initial bias parameters (np.array expect: one column of floats)
        np.ndarray[float, ndim = 1] deadline,
        float sd, # std dev of Normal from where we sample vs
        float ndt = 0, # ndt is supposed to be 0 by default because of parameter identifiability issues
        int nact = 3,
        int n_samples = 2000,
        int n_trials = 1,
        float max_t = 20,
        **kwargs
        ):

    # v_t = np.random.normal(v, sd)
    # print(len(z), nact, np.array([z]*nact).transpose().shape)
    # z_t = np.random.uniform(np.zeros((len(z), nact)), np.array([z]*nact).transpose(), (len(z), nact))

    # Param views
    cdef float[:, :] v_RL_view = v_RL
    cdef float[:, :] v_WM_view = v_WM
    cdef float[:, :] a_view = a
    cdef float[:, :] z_view = z

    cdef float[:] deadline_view = deadline

    cdef np.ndarray[float, ndim = 1] zs
    cdef np.ndarray[double, ndim = 2] x_t_RL
    cdef np.ndarray[double, ndim = 2] x_t_WM
    cdef np.ndarray[double, ndim = 1] vs_RL
    cdef np.ndarray[double, ndim = 1] vs_WM

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    cdef float[:, :, :] rts_view = rts
    
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef int[:, :, :] choices_view = choices
    
    cdef Py_ssize_t n, k, i

    for k in range(n_trials):
        
        for n in range(n_samples):
            zs = np.random.uniform(0, z_view[k], nact).astype(DTYPE)

            vs_RL = np.abs(np.random.normal(v_RL_view[k], sd)) # np.abs() to avoid negative vs
            vs_WM = np.abs(np.random.normal(v_WM_view[k], sd)) # np.abs() to avoid negative vs

            x_t_RL = ([a_view[k]]*nact - zs)/vs_RL
            x_t_WM = ([a_view[k]]*nact - zs)/vs_WM

            if np.min(x_t_RL) <= np.min(x_t_WM):
                rts_view[n, k, 0] = np.min(x_t_RL) + ndt  # store reaction time for sample n
                choices_view[n, k, 0] = np.argmin(x_t_RL) # store choices for sample n
            else:
                rts_view[n, k, 0] = np.min(x_t_WM) + ndt  # store reaction time for sample n
                choices_view[n, k, 0] = np.argmin(x_t_WM) # store choices for sample n  
            
            # If the rt exceeds the deadline, set rt to -999
            if rts_view[n, k, 0] >= deadline_view[k]:
                rts_view[n, k, 0] = -999
        

    v_dict = {}    
    for i in range(nact):
        v_dict['v_RL_' + str(i)] = v_RL[:, i]
        v_dict['v_WM_' + str(i)] = v_WM[:, i]

    return {'rts': rts, 'choices': choices, 'metadata': {**v_dict,
                                                         'a': a,
                                                         'z': z,
                                                         'deadline': deadline,
                                                         'sd': sd,
                                                         'n_samples': n_samples,
                                                         'simulator' : 'rlwm_lba_race',
                                                         'possible_choices': list(np.arange(0, nact, 1)),
                                                         'max_t': max_t,
                                                         }}
# ----------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_mic2_unnormalized_ornstein_multinoise(np.ndarray[float, ndim = 1] vh, 
                                                        np.ndarray[float, ndim = 1] vl1,
                                                        np.ndarray[float, ndim = 1] vl2,
                                                        np.ndarray[float, ndim = 1] a,
                                                        np.ndarray[float, ndim = 1] zh,
                                                        np.ndarray[float, ndim = 1] zl1,
                                                        np.ndarray[float, ndim = 1] zl2,
                                                        np.ndarray[float, ndim = 1] d, # damper (1 --> no drift on low level until high level done, 0 --> full drift on low level)
                                                        np.ndarray[float, ndim = 1] g, # inhibition parameter for the low dim choice procress while high dim is running
                                                        np.ndarray[float, ndim = 1] t,
                                                        np.ndarray[float, ndim = 1] deadline,
                                                        float s = 1.0,
                                                        float delta_t = 0.001,
                                                        float max_t = 20,
                                                        int n_samples = 20000,
                                                        int n_trials = 1,
                                                        print_info = True,
                                                        boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                                                        boundary_multiplicative = True,
                                                        boundary_params = {},
                                                        random_state = None,
                                                        return_option = 'full',
                                                        smooth = False,
                                                        **kwargs):

    set_seed(random_state)
    # Param views
    cdef float[:] vh_view = vh
    cdef float[:] vl1_view = vl1
    cdef float[:] vl2_view = vl2
    cdef float[:] a_view = a
    cdef float[:] zh_view = zh
    cdef float[:] zl1_view = zl1
    cdef float[:] zl2_view = zl2
    cdef float[:] d_view = d
    cdef float[:] g_view = g
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline

    # TD: Add trajectory --> same issue as with par2 model above... might need to make a separate simulator for trajectories
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    rts_low = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    rts_high = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)

    cdef float[:, :, :] rts_view = rts
    cdef float[:, :, :] rts_high_view = rts_high
    cdef float[:, :, :] rts_low_view = rts_low
    cdef int[:, :, :] choices_view = choices

    traj = np.zeros((int(max_t / delta_t) + 1, 3), dtype = DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Y particle trace
    bias_trace_l1 = np.zeros(num_draws, dtype = DTYPE)
    bias_trace_l2 = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] bias_trace_l1_view = bias_trace_l1
    cdef float[:] bias_trace_l2_view = bias_trace_l2

    cdef float y_h, y_l, y_l1, y_l2
    cdef float v_l, v_l1, v_l2, 
    cdef float t_h, t_l, t_l1, t_l2, smooth_u, deadline_tmp
    cdef Py_ssize_t n, ix, ix1, ix2, ix_l, ix_tmp, ix1_tmp, ix2_tmp, k
    cdef Py_ssize_t m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
        # Loop over samples
        for n in range(n_samples):
            choices_view[n, k, 0] = 0 # reset choice
            t_h = 0 # reset time high dimension
            t_l = 0 # reset time low dimension
            t_l1 = 0 # reset time low dimension (1)
            t_l2 = 0 # reset time low dimension (2)
            ix = 0 # reset boundary index
            ix1 = 0 # reset boundary index (1)
            ix2 = 0 # reset boundary index (2)

            # Initialize walkers
            # Particle
            y_h = (-1) * boundary_view[0] + (zh_view[k] * 2 * (boundary_view[0])) 
            # Relative particle position (used as resource allocator for low dim choice)
            bias_trace_l2_view[0] = ((y_h + boundary_view[0]) / (2))
            bias_trace_l1_view[0] = boundary_view[0] - bias_trace_l2_view[0]

            if n == 0:
                if k == 0:
                    traj_view[0, 0] = y_h

            # Random walks until y_h hits bound
            while (y_h >= ((-1) * boundary_view[ix])) and ((y_h <= boundary_view[ix])) and (t_h <= deadline_tmp):
                y_h += (vh_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                bias_trace_l2_view[ix] = ((y_h + boundary_view[ix]) / (2))
                bias_trace_l1_view[ix] = boundary_view[ix] - bias_trace_l2_view[ix]
                t_h += delta_t
                ix += 1
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add 2 deterministically
            # y at lower bound --> choice_view[n, k, 0] stay the same deterministically

            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if random_uniform() <= 0.5:
                    choices_view[n, k, 0] += 2
            # Otherwise, apply rule from above
            elif random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2
           
            y_l2 = (- 1) * boundary_view[0] + (zl2_view[k] * 2 * (boundary_view[0]))
            y_l1 = (- 1) * boundary_view[0] + (zl1_view[k] * 2 * (boundary_view[0]))
            
            if choices_view[n, k, 0] == 0:
                 # Fill bias trace a until max_rt reached
                ix1_tmp = ix + 1
                while ix1_tmp < num_draws:
                    bias_trace_l2_view[ix1_tmp] = 0.0
                    bias_trace_l1_view[ix1_tmp] = 1.0
                    ix1_tmp += 1

            else: # Store intermediate choice
                # Fill bias tracea until max_rt reached
                ix2_tmp = ix + 1
                while ix2_tmp < num_draws:
                    bias_trace_l2_view[ix2_tmp] = 1.0
                    bias_trace_l1_view[ix2_tmp] = 0.0
                    ix2_tmp += 1

            # lower level random walker (1)
            if (choices_view[n, k, 0] == 0) | ((n == 0) & (k == 0)):
                while (y_l1 >= ((-1) * boundary_view[ix1])) and (y_l1 <= boundary_view[ix1]) and (t_l1 <= deadline_tmp):
                    if (bias_trace_l1_view[ix1] < boundary_view[ix1]) and (bias_trace_l1_view[ix1] > 0):
                        # main propagation if bias_trace is between 0 and 1 (high level choice is not yet made)
                        y_l1 += (((vl1_view[k] * bias_trace_l1_view[ix1] * (1 - d_view[k])) - (g_view[k] * y_l1)) * delta_t)
                        # add gaussian displacement
                        y_l1 += (sqrt_st * gaussian_values[m]) * bias_trace_l1_view[ix1] 
                    else:
                        # main propagation if bias_trace is not between 0 and 1 (high level choice is already made)
                        y_l1 += (vl1_view[k] * delta_t)
                        # add gaussian displacement
                        y_l1 += (sqrt_st * gaussian_values[m])
                    
                    # propagate time and indices
                    t_l1 += delta_t
                    ix1 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix1, 1] = y_l1

            # lower level random walker (2)
            if (choices_view[n, k, 0] == 2) | ((n == 0) & (k == 0)):
                while (y_l2 >= ((-1) * boundary_view[ix2])) and (y_l2 <= boundary_view[ix2]) and (t_l2 <= deadline_tmp):
                    if (bias_trace_l2_view[ix2] < boundary_view[ix2]) and (bias_trace_l2_view[ix2] > 0):
                        # main propagation if bias_trace is between 0 and 1 (high level choice is not yet made)
                        y_l2 += (((vl2_view[k] * bias_trace_l2_view[ix2] * (1 - d_view[k])) - (g_view[k] * y_l2)) * delta_t)
                        # add gaussian displacement
                        y_l2 += (sqrt_st * gaussian_values[m]) * bias_trace_l2_view[ix2] 
                    else:
                        # main propagation if bias_trace is not between 0 and 1 (high level choice is already made)
                        y_l2 += (vl2_view[k] * delta_t)
                        # add gaussian displacement
                        y_l2 += (sqrt_st * gaussian_values[m])
                    
                    # propagate time and indices
                    t_l2 += delta_t
                    ix2 += 1
                    m += 1
                    if m == num_draws:
                        gaussian_values = draw_gaussian(num_draws)
                        m = 0

                    if n == 0:
                        if k == 0:
                            traj_view[ix2, 2] = y_l2

            # Get back to single y_l and t_l
            if (choices_view[n, k, 0] == 0):
                t_l = t_l1
                y_l = y_l1
                ix_l = ix1
            else:
                t_l = t_l2
                y_l = y_l2
                ix_l = ix2

            if smooth:
                if t_h == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif fmax(t_h, t_l) < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]
            rts_high_view[n, k, 0] = t_h + t_view[k]
            rts_low_view[n, k, 0] = t_l + t_view[k]

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add one deterministically
            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically

            # If boundary is negative (or 0) already, we flip a coin
            if boundary_view[ix] <= 0:
                if random_uniform() <= 0.5:
                    choices_view[n, k, 0] += 1
            # Otherwise apply rule from above
            elif random_uniform() <= ((y_l + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 1

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
                'metadata': {'vh': vh,
                            'vl1': vl1,
                            'vl2': vl2,
                            'a': a,
                            'zh': zh,
                            'zl1': zl1,
                            'zl2': zl2,
                            'd': d,
                            't': t,
                            'deadline': deadline,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'n_trials': n_trials,
                            'simulator': 'ddm_flexbound_mic2_adj',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [0, 1, 2, 3],
                            'trajectory': traj,
                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
                'metadata': {'simulator': 'ddm_flexbound_mic2_adj', 
                             'possible_choices': [0, 1, 2, 3],
                             'boundary_fun_type': boundary_fun.__name__,
                             'n_samples': n_samples,
                             'n_trials': n_trials,
                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# ----------------------------------------------------------------------------------------------------


## Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
## @cythonboundscheck(False)
## @cythonwraparound(False)
#def ddm_flexbound_mic2_unnormalized_ornstein_multinoise(np.ndarray[float, ndim = 1] vh, 
#                                                        np.ndarray[float, ndim = 1] vl1,
#                                                        np.ndarray[float, ndim = 1] vl2,
#                                                        np.ndarray[float, ndim = 1] a,
#                                                        np.ndarray[float, ndim = 1] zh,
#                                                        np.ndarray[float, ndim = 1] zl1,
#                                                        np.ndarray[float, ndim = 1] zl2,
#                                                        np.ndarray[float, ndim = 1] d, # damper (1 --> no drift on low level until high level done, 0 --> full drift on low level)
#                                                        np.ndarray[float, ndim = 1] g, # inhibition parameter for the low dim choice procress while high dim is running
#                                                        np.ndarray[float, ndim = 1] t,
#                                                        np.ndarray[float, ndim = 1] deadline,
#                                                        float s = 1.0,
#                                                        float delta_t = 0.001,
#                                                        float max_t = 20,
#                                                        int n_samples = 20000,
#                                                        int n_trials = 1,
#                                                        print_info = True,
#                                                        boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
#                                                        boundary_multiplicative = True,
#                                                        boundary_params = {},
#                                                        random_state = None,
#                                                        return_option = 'full',
#                                                        smooth = False,
#                                                        **kwargs):
#
#    set_seed(random_state)
#    # Param views
#    cdef float[:] vh_view = vh
#    cdef float[:] vl1_view = vl1
#    cdef float[:] vl2_view = vl2
#    cdef float[:] a_view = a
#    cdef float[:] zh_view = zh
#    cdef float[:] zl1_view = zl1
#    cdef float[:] zl2_view = zl2
#    cdef float[:] d_view = d
#    cdef float[:] g_view = g
#    cdef float[:] t_view = t
#    cdef float[:] deadline_view = deadline
#
#    # TD: Add trajectory --> same issue as with par2 model above... might need to make a separate simulator for trajectories
#    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
#    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
#    rts_low = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
#    rts_high = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
#
#    cdef float[:, :, :] rts_view = rts
#    cdef float[:, :, :] rts_high_view = rts_high
#    cdef float[:, :, :] rts_low_view = rts_low
#    cdef int[:, :, :] choices_view = choices
#
#    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
#    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step
#
#    # Boundary storage for the upper bound
#    cdef int num_draws = int((max_t / delta_t) + 1)
#    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
#    boundary = np.zeros(t_s.shape, dtype = DTYPE)
#    cdef float[:] boundary_view = boundary
#
#    # Y particle trace
#    bias_trace = np.zeros(num_draws, dtype = DTYPE)
#    cdef float[:] bias_trace_view = bias_trace
#
#    cdef float y_h, y_l, v_l, t_h, t_l, smooth_u, deadline_tmp
#    cdef Py_ssize_t n, ix, ix_tmp, k
#    cdef Py_ssize_t m = 0
#    cdef float[:] gaussian_values = draw_gaussian(num_draws)
#
#    for k in range(n_trials):
#        # Precompute boundary evaluations
#        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}
#
#        # Precompute boundary evaluations
#        if boundary_multiplicative:
#            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
#        else:
#            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
#    
#        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
#        # Loop over samples
#        for n in range(n_samples):
#            choices_view[n, k, 0] = 0 # reset choice
#            t_h = 0 # reset time high dimension
#            t_l = 0 # reset time low dimension
#            ix = 0 # reset boundary index
#
#            # Initialize walkers
#            # Particle
#            y_h = (-1) * boundary_view[0] + (zh_view[k] * 2 * (boundary_view[0])) 
#            # Relative particle position (used as resource allocator for low dim choice) / UNNORMALIZED
#            bias_trace_view[0] = ((y_h + boundary_view[0]) / 2)
#
#            # Random walks until y_h hits bound
#            while (y_h >= ((-1) * boundary_view[ix])) and ((y_h <= boundary_view[ix])) and (t_h <= deadline_tmp):
#                y_h += (vh_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
#                bias_trace_view[ix] = ((y_h + boundary_view[ix]) / 2)
#                t_h += delta_t
#                ix += 1
#                m += 1
#                if m == num_draws:
#                    gaussian_values = draw_gaussian(num_draws)
#                    m = 0
#
#            # The probability of making a 'mistake' 1 - (relative y position)
#            # y at upper bound --> choices_view[n, k, 0] add 2 deterministically
#            # y at lower bound --> choice_view[n, k, 0] stay the same deterministically
#
#            # If boundary is negative (or 0) already, we flip a coin
#            if boundary_view[ix] <= 0:
#                if random_uniform() <= 0.5:
#                    choices_view[n, k, 0] += 2
#            # Otherwise, apply rule from above
#            elif random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
#                choices_view[n, k, 0] += 2
#           
#            if choices_view[n, k, 0] == 2:
#                y_l = (- 1) * boundary_view[0] + (zl2_view[k] * 2 * (boundary_view[0])) 
#                v_l = vl2_view[k]
#
#                # Fill bias trace until max_rt reached
#                ix_tmp = ix + 1
#                while ix_tmp < num_draws:
#                    bias_trace_view[ix_tmp] = max(boundary_view[ix_tmp], 0)
#                    ix_tmp += 1
#
#            else: # Store intermediate choice
#                y_l = (- 1) * boundary_view[0] + (zl1_view[k] * 2 * (boundary_view[0])) 
#                v_l = vl1_view[k]
#
#                # Fill bias trace until max_rt reached
#                ix_tmp = ix + 1
#                while ix_tmp < num_draws:
#                    bias_trace_view[ix_tmp] = 0.0
#                    ix_tmp += 1
#
#                # We need to reverse the bias_trace if we took the lower choice
#                ix_tmp = 0 
#                while ix_tmp < num_draws:
#                    bias_trace_view[ix_tmp] = max(boundary_view[ix_tmp] - bias_trace_view[ix_tmp], 0)
#                    ix_tmp += 1
#
#            # Random walks until the y_l corresponding to y_h hits bound
#            ix = 0
#            while (y_l >= ((-1) * boundary_view[ix])) and (y_l <= boundary_view[ix]) and (t_l <= deadline_tmp):
#                if (bias_trace_view[ix] < boundary_view[ix]) and (bias_trace_view[ix] > 0):
#                    # main propagation if bias_trace is between 0 and 1 (high level choice is not yet made)
#                    y_l += (((v_l * bias_trace_view[ix] * (1 - d_view[k])) - (g_view[k] * y_l)) * delta_t)
#                    # add gaussian displacement
#                    y_l += (sqrt_st * gaussian_values[m]) * bias_trace_view[ix] 
#                else:
#                    # main propagation if bias_trace is not between 0 and 1 (high level choice is already made)
#                    y_l += (v_l * delta_t)
#                    # add gaussian displacement
#                    y_l += (sqrt_st * gaussian_values[m])
#                
#                # propagate time and indices
#                t_l += delta_t
#                ix += 1
#                m += 1
#                if m == num_draws:
#                    gaussian_values = draw_gaussian(num_draws)
#                    m = 0
#
#            if smooth:
#                if t_h == 0:
#                    smooth_u = random_uniform() * 0.5 * delta_t
#                else:
#                    smooth_u = (0.5 - random_uniform()) * delta_t
#            else:
#                smooth_u = 0.0
#
#            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]
#            rts_high_view[n, k, 0] = t_h + t_view[k]
#            rts_low_view[n, k, 0] = t_l + t_view[k]
#
#            # The probability of making a 'mistake' 1 - (relative y position)
#            # y at upper bound --> choices_view[n, k, 0] add one deterministically
#            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically
#
#            # If boundary is negative (or 0) already, we flip a coin
#            if boundary_view[ix] <= 0:
#                if random_uniform() <= 0.5:
#                    choices_view[n, k, 0] += 1
#            # Otherwise apply rule from above
#            elif random_uniform() <= ((y_l + boundary_view[ix]) / (2 * boundary_view[ix])):
#                choices_view[n, k, 0] += 1
#
#            if rts_view[n, k, 0] > deadline_tmp:
#                rts_view[n, k, 0] = -999
#
#    if return_option == 'full':
#        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
#                'metadata': {'vh': vh,
#                            'vl1': vl1,
#                            'vl2': vl2,
#                            'a': a,
#                            'zh': zh,
#                            'zl1': zl1,
#                            'zl2': zl2,
#                            'd': d,
#                            't': t,
#                            'deadline': deadline,
#                            's': s,
#                            **boundary_params,
#                            'delta_t': delta_t,
#                            'max_t': max_t,
#                            'n_samples': n_samples,
#                            'n_trials': n_trials,
#                            'simulator': 'ddm_flexbound_mic2_adj',
#                            'boundary_fun_type': boundary_fun.__name__,
#                            'possible_choices': [0, 1, 2, 3],
#                            'trajectory': 'This simulator does not yet allow for trajectory simulation',
#                            'boundary': boundary}}
#    elif return_option == 'minimal':
#        return {'rts': rts, 'choices': choices, 'rts_high': rts_high, 'rts_low': rts_low, 
#                'metadata': {'simulator': 'ddm_flexbound_mic2_adj', 
#                             'possible_choices': [0, 1, 2, 3],
#                             'boundary_fun_type': boundary_fun.__name__,
#                             'n_samples': n_samples,
#                             'n_trials': n_trials,
#                             }}
#    else:
#        raise ValueError('return_option must be either "full" or "minimal"')
## ----------------------------------------------------------------------------------------------------


# Simulate (rt, choice) tuples from: DDM WITH FLEXIBLE BOUNDARIES ------------------------------------
# @cythonboundscheck(False)
# @cythonwraparound(False)
def ddm_flexbound_tradeoff(np.ndarray[float, ndim = 1] vh, 
                           np.ndarray[float, ndim = 1] vl1,
                           np.ndarray[float, ndim = 1] vl2,
                           np.ndarray[float, ndim = 1] a,
                           np.ndarray[float, ndim = 1] zh,
                           np.ndarray[float, ndim = 1] zl1,
                           np.ndarray[float, ndim = 1] zl2,
                           np.ndarray[float, ndim = 1] d, # d for 'dampen' effect on drift parameter
                           np.ndarray[float, ndim = 1] t,
                           np.ndarray[float, ndim = 1] deadline,
                           float s = 1,
                           float delta_t = 0.001,
                           float max_t = 20,
                           int n_samples = 20000,
                           int n_trials = 1,
                           print_info = True,
                           boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                           boundary_multiplicative = True,
                           boundary_params = {},
                           random_state = None,
                           return_option = 'full',
                           smooth = False,
                           **kwargs):

    set_seed(random_state)
    # Param views
    cdef float[:] vh_view = vh
    cdef float[:] vl1_view = vl1
    cdef float[:] vl2_view = vl2
    cdef float[:] a_view = a
    cdef float[:] zh_view = zh
    cdef float[:] zl1_view = zl1
    cdef float[:] zl2_view = zl2
    cdef float[:] d_view = d
    cdef float[:] t_view = t
    cdef float[:] deadline_view = deadline

    # TD: Add trajectory --> same issue as with par2 model above... might need to make a separate simulator for trajectories

    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)

    cdef float[:, :, :] rts_view = rts
    cdef int[:, :, :] choices_view = choices

    cdef float delta_t_sqrt = sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    t_s = np.arange(0, max_t + delta_t, delta_t).astype(DTYPE)
    boundary = np.zeros(t_s.shape, dtype = DTYPE)
    cdef float[:] boundary_view = boundary

    # Y particle trace
    bias_trace = np.zeros(num_draws, dtype = DTYPE)
    cdef float[:] bias_trace_view = bias_trace

    cdef float y_h, y_l, v_l, t_h, t_l, tmp_pos_dep, smooth_u, deadline_tmp
    cdef Py_ssize_t n, ix, ix_tmp, k
    cdef Py_ssize_t m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for k in range(n_trials):
        # Precompute boundary evaluations
        boundary_params_tmp = {key: boundary_params[key][k] for key in boundary_params.keys()}

        # Precompute boundary evaluations
        if boundary_multiplicative:
            boundary[:] = np.multiply(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
        else:
            boundary[:] = np.add(a_view[k], boundary_fun(t = t_s, **boundary_params_tmp)).astype(DTYPE)
    
        deadline_tmp = min(max_t, deadline_view[k] - t_view[k])
        # Loop over samples
        for n in range(n_samples):
            choices_view[n, k, 0] = 0 # reset choice
            t_h = 0 # reset time high dimension
            t_l = 0 # reset time low dimension
            ix = 0 # reset boundary index

            # Initialize walkers
            y_h = (-1) * boundary_view[0] + (zh_view[k] * 2 * (boundary_view[0])) 
            bias_trace_view[0] = ((y_h + boundary_view[0]) / (2 * boundary_view[0]))

            # Random walks until y_h hits bound
            while (y_h >= ((-1) * boundary_view[ix])) and ((y_h <= boundary_view[ix])) and (t_h <= deadline_tmp):
                y_h += (vh_view[k] * delta_t) + (sqrt_st * gaussian_values[m])
                bias_trace_view[ix] = ((y_h + boundary_view[ix]) / (2 * boundary_view[ix]))
                t_h += delta_t
                ix += 1
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add 2 deterministically
            # y at lower bound --> choice_view[n, k, 0] stay the same deterministically
            if random_uniform() <= ((y_h + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 2
           
            if choices_view[n, k, 0] == 2:
                y_l = (- 1) * boundary_view[0] + (zl2_view[k] * 2 * (boundary_view[0])) 
                v_l = vl2_view[k]

                # Fill bias trace until max_rt reached
                ix_tmp = ix + 1
                while ix_tmp < num_draws:
                    bias_trace_view[ix_tmp] = 1.0
                    ix_tmp += 1

            else: # Store intermediate choice
                y_l = (- 1) * boundary_view[0] + (zl1_view[k] * 2 * (boundary_view[0])) 
                v_l = vl1_view[k]

                # Fill bias trace until max_rt reached
                ix_tmp = ix + 1
                while ix_tmp < num_draws:
                    bias_trace_view[ix_tmp] = 0.0
                    ix_tmp += 1

                #We need to reverse the bias_trace if we took the lower choice
                ix_tmp = 0 
                while ix_tmp < num_draws:
                    bias_trace_view[ix_tmp] = 1.0 - bias_trace_view[ix_tmp]
                    ix_tmp += 1

                #print('new bias_trace: ', bias_trace)
            
            # Random walks until the y_l corresponding to y_h hits bound
            ix = 0
            while (y_l >= ((-1) * boundary_view[ix])) and (y_l <= boundary_view[ix]) and (t_l <= deadline_tmp):
                # Compute local position dependence
                # AF-todo: can't understand what the idea here is anymore
                # especially why bias_trace_view is flipped (-1) here
                tmp_pos_dep = (1 + (d_view[k] * (bias_trace_view[ix] - 1.0))) / (2 - d_view[k])

                if (bias_trace_view[ix] < 1) and (bias_trace_view[ix] > 0):
                    # Before high-dim choice is taken
                    y_l += tmp_pos_dep * (v_l * delta_t) # Add drift
                    y_l += tmp_pos_dep * sqrt_st * gaussian_values[m] # Add noise
                else:
                    # After high-dim choice is taken
                    y_l += (v_l * delta_t) # Add drift
                    y_l += sqrt_st * gaussian_values[m] # Add noise
    
                t_l += delta_t # update time for low_dim choice
                ix += 1 # update time index
                m += 1 # update rv couter

                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0

            if smooth:
                if t_h == 0.0:
                    smooth_u = random_uniform() * 0.5 * delta_t
                elif fmax(t_h, t_l) < deadline_tmp:
                    smooth_u = (0.5 - random_uniform()) * delta_t
                else:
                    smooth_u = 0.0
            else:
                smooth_u = 0.0

            rts_view[n, k, 0] = fmax(t_h, t_l) + t_view[k]

            # The probability of making a 'mistake' 1 - (relative y position)
            # y at upper bound --> choices_view[n, k, 0] add one deterministically
            # y at lower bound --> choice_view[n, k, 0] stays the same deterministically
            if random_uniform() <= ((y_l + boundary_view[ix]) / (2 * boundary_view[ix])):
                choices_view[n, k, 0] += 1

            if (rts_view[n, k, 0] >= deadline_view[k]) | (deadline_view[k] <= 0):
                rts_view[n, k, 0] = -999

    if return_option == 'full':
        return {'rts': rts, 'choices': choices, 'metadata': {'vh': vh,
                                                            'vl1': vl1,
                                                            'vl2': vl2,
                                                            'a': a,
                                                            'zh': zh,
                                                            'zl1': zl1,
                                                            'zl2': zl2,
                                                            'd': d,
                                                            't': t,
                                                            'deadline': deadline,
                                                            's': s,
                                                            **boundary_params,
                                                            'delta_t': delta_t,
                                                            'max_t': max_t,
                                                            'n_samples': n_samples,
                                                            'n_trials': n_trials,
                                                            'simulator': 'ddm_flexbound_mic2_adj',
                                                            'boundary_fun_type': boundary_fun.__name__,
                                                            'possible_choices': [0, 1, 2, 3],
                                                            'trajectory': 'This simulator does not yet allow for trajectory simulation',
                                                            'boundary': boundary}}
    elif return_option == 'minimal':
        return {'rts': rts, 'choices': choices, 'metadata': {'simulator': 'ddm_flexbound_mic2_adj', 
                                                             'possible_choices': [0, 1, 2, 3],
                                                             'boundary_fun_type': boundary_fun.__name__,
                                                             'n_samples': n_samples,
                                                             'n_trials': n_trials,
                                                             }}
    else:
        raise ValueError('return_option must be either "full" or "minimal"')
# -----------------------------------------------------------------------------------------------