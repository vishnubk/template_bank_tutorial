import numpy as np
#import matplotlib.pyplot as plt
import sympy as sy
sy.init_printing(use_unicode=True)
import pickle
import sys, time
import math
import emcee
from multiprocessing import Pool
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
from schwimmbad import MPIPool
import multiprocessing
from skmonaco import mcquad
import sys
import corner

"""This Module has been built to Calculate and Generate Random Templates based on Messenger et al. 2008 (arXiv:0809.5223). 
These templates are used to do a fully coherent search for circular binary orbits in radio observations. 

      Receipe to Generate Random Template Bank!

1. Initialise your signal phase model and calculate the metric tensor of your parameter space.

2. Compute determinant of this metric. This will be used later as a constant density function to distribute templates
   in your parameter space.

2. Compute proper volume/volume integral of your parameter space and calculate number of required templates 
   based on required coverage and mismatch.

4. For each proposal template, draw random values from angular velocity, 
   projected radius and orbital phase (parameters of interest).

5. Implement a MCMC based on metropolis hastings algorithm using square root of the determinant of the metric tensor as your constant density function. Write Results to file. 
"""

parser = argparse.ArgumentParser(description='Generate a template-bank for a user-defined no. of templates for coherent full keplerian circular orbit search')
parser.add_argument('-o', '--output_path', help='Output path to save results',  default="os.getcwd()")
parser.add_argument('-t', '--obs_time', help='Observation time in minutes', default='72', type=float)
parser.add_argument('-p', '--porb_low', help='Lower limit of Orbital Period in minutes', default='360', type=float)
parser.add_argument('-P', '--porb_high', help='Upper limit of Orbital Period in minutes', type=float)
parser.add_argument('-c', '--max_comp_mass', help='Maximum mass of Companion in solar mass units', default='8', type=float)
parser.add_argument('-d', '--min_pulsar_mass', help='Minimum mass of Pulsar in solar mass units', default='1.4', type=float)
parser.add_argument('-s', '--spin_period', help='Fastest spin period of pulsar in ms', default='5', type=float)
parser.add_argument('-f', '--fraction', help='Probability fraction of orbits of different inclination angles', default='1', type=float)
parser.add_argument('-b', '--coverage', help='Coverage of template-bank', default='0.9', type=float)
parser.add_argument('-m', '--mismatch', help='Mismatch of template-bank', default='0.2', type=float)
parser.add_argument('-n', '--ncpus', help='Number of CPUs to use for calculation', type=int)
parser.add_argument('-i', '--nmc', help='Number of iterations for monte-carlo integration', default='100000', type=int)
parser.add_argument('-file', '--output_filename', help='output filename', default='5D_template_bank', type=str)


args = parser.parse_args()
#Either use the command line arguments or use the default values which is ten times the observation time
args.porb_high = args.porb_high if args.porb_high else args.obs_time * 10


f, tau, omega, psi, phi, t, T, a, pi, f0 = sy.symbols('f \\tau \\Omega \\psi \phi t T a \pi f_0')
sy.init_printing(use_unicode=True) #pretty printing

## Phase Model for Circular Binary Orbits
phi = 2 * pi * f * (t + tau * sy.sin(omega * t + psi))
def time_average(a):
    b = (1/T) * sy.integrate(a, (t, 0, T))
    return b

variables=[f, tau, omega, psi]

metric_tensor=np.empty(shape=(4,4), dtype=object)
for i in range(len(variables)):
    for j in range(len(variables)):
         metric_tensor[i][j]=(time_average(sy.diff(phi,variables[i]) * sy.diff(phi,variables[j])) - time_average(sy.diff(phi,variables[i])) * time_average(sy.diff(phi,variables[j])))

metric_tensor_w_f_row_column = metric_tensor[1:4,1:4]
variables=[tau, omega, psi]
metric_tensor3D=np.empty(shape=(3,3), dtype=object)
for i in range(len(variables)):
    for j in range(len(variables)):
        metric_tensor3D[i][j]=metric_tensor_w_f_row_column[i][j] * metric_tensor[0][0] - (metric_tensor[0][i+1] * \
                                                                                         metric_tensor[j+1][0])
metric_tensor3D=sy.Matrix(metric_tensor3D)

''' matrix.det() method in sympy does an in-built symplification in python3 which gives wrong results! If in python2.7, you can run metric_tensor3D.det(), however in python3 stick to this workaround by manually ggiving the formula for a determinant '''


A = sy.Matrix(3, 3, sy.symbols('A:3:3'))
det_metric_tensor3D = A.det().subs(zip(list(A), list(metric_tensor3D)))
det_metric_tensor3D=det_metric_tensor3D/metric_tensor[0][0]**3
expr=det_metric_tensor3D**0.5
expr_numpy = sy.lambdify([f, psi, omega, tau, T, pi], expr, "numpy")

def det_sq_root(angular_velocity, projected_radius, orbital_phase, freq, obs_time):

    return expr_numpy(freq, orbital_phase, angular_velocity, projected_radius, obs_time, np.pi)


def calculate_alpha(sini, max_companion_mass, min_pulsar_mass, current_candidate_mass, current_companion_mass):
    ''' Basically, this checks if a certain mass range is covered by your asini trials
        Explanation:
              Say min_pulsar_mass = 1.4 M0, max_companion_mass = 8M0, if args.fraction = 1, then as long as current_pulsar_mass is above the minimum pulsar mass and  
              current_companion_mass is below the max_companion_mass, then probability of detection = 1. This probability assumes that the orbit is circular, 
              and your data is sensitive enough to find the pulsar for a given coverage and mismatch.
              
              If current_companion_mass > 8 M0 or current_pulsar_mass < 1.4 M0, then p < 1, and we only partially cover that parameter space based on our asini trials 
         '''

    alpha = sini * max_companion_mass * ((current_candidate_mass + current_companion_mass)**(2/3))/(current_companion_mass * \
     (max_companion_mass + min_pulsar_mass)**(2/3))
    p = 1 - np.sqrt(1 - alpha**2)
    return p

def number_templates(dimension,coverage,mismatch,volume):
    ''' This calculates the required random templates based on volume, coverage and mismatch '''
    n_dim_ball_volume = math.pow(np.pi,dimension/2)/math.gamma((dimension/2) + 1)
    N=math.log(1-coverage)/math.log(1-math.pow(mismatch,dimension/2) * n_dim_ball_volume/volume)
    return N



def log_posterior(theta, freq, obs_time, lowest_angular_velocity, highest_angular_velocity, max_initial_orbital_phase):

    ''' theta is a three-dimensional vector of our model holding all the orbital template parameters '''
    
    angular_velocity, projected_radius, orbital_phase  = theta
    if not (lowest_angular_velocity < angular_velocity < highest_angular_velocity and 0. < orbital_phase < max_initial_orbital_phase):

        return -np.inf

    max_projected_radius = alpha * G**(1/3) * max_companion_mass * M_0 * angular_velocity**(-2/3)/(c * ((min_pulsar_mass + max_companion_mass) * M_0)**(2/3))
    if not (0. < projected_radius <= max_projected_radius):
        return -np.inf

    determinant = expr_numpy(freq, orbital_phase, angular_velocity, projected_radius, obs_time, np.pi)

    if determinant == 0:
        return -np.inf

    if math.isnan(determinant):
        return -np.inf

    determinant = np.log(determinant)
    return determinant

def volume_integral(t, max_companion_mass, spin_freq, obs_time, min_pulsar_mass):
    angular_velocity, projected_radius, orbital_phase = t
    max_projected_radius = alpha * G**(1/3) * max_companion_mass * M_0 * angular_velocity**(-2/3)/(c * ((min_pulsar_mass + max_companion_mass) * M_0)**(2/3))
    if projected_radius <= max_projected_radius:

        return expr_numpy(spin_freq, orbital_phase, angular_velocity, projected_radius, obs_time, np.pi)

    return 0

 

#total_templates = int(np.around(float(sys.argv[1])))
''' Define your prior ranges'''

G = 6.67e-11
M_0 = 1.989e+30
c = 2.99792458e+08
pi_1 = np.pi
obs_time = args.obs_time * 60 # 18 mins
p_orb_upper_limit = args.porb_high * 60
p_orb_low_limit = args.porb_low * 60 #1.5 hours
min_pulsar_mass = args.min_pulsar_mass
max_companion_mass = args.max_comp_mass
alpha = args.fraction
coverage = args.coverage
mismatch = args.mismatch
if args.ncpus:
    ncpus = args.ncpus
else:
    ncpus = os.cpu_count()
fastest_spin_period_ms = args.spin_period
spin_freq = 1/(fastest_spin_period_ms * 1e-03) # 5ms
volume_integral_iterations = args.nmc
batch_size_integration = int(volume_integral_iterations/ncpus) # This is used to batch the integration and parallelise it in mcquad
max_initial_orbital_phase = 2 * np.pi
max_longitude_periastron = 2 * np.pi
output_path = args.output_path
lowest_angular_velocity = 2 * np.pi/p_orb_upper_limit
highest_angular_velocity = 2 * np.pi/p_orb_low_limit
highest_limit_projected_radius = (alpha * G**(1/3) * max_companion_mass * M_0 * highest_angular_velocity**(-2/3)/(c * ((min_pulsar_mass + max_companion_mass) * M_0)**(2/3))) + 1.0
probability = calculate_alpha(alpha,max_companion_mass,min_pulsar_mass,min_pulsar_mass,max_companion_mass)


# Define the limits of your coordinates here in xl and xu
volume_integral_result, estimated_volume_integral_error = mcquad(volume_integral,npoints = volume_integral_iterations, xl= [(2 * np.pi)/p_orb_upper_limit, 0., 0.], \
                                     xu=[(2 * np.pi)/p_orb_low_limit, highest_limit_projected_radius, max_initial_orbital_phase], nprocs=ncpus, batch_size=batch_size_integration, args=[max_companion_mass, spin_freq, obs_time, min_pulsar_mass])

print('Volume Integral: ', volume_integral_result, 'Volume Integral Error: ', estimated_volume_integral_error)
print('Volume integral error is: %.2f' %((estimated_volume_integral_error/volume_integral_result) * 100), ' %')
total_templates_required = number_templates(3,coverage,mismatch,np.around(volume_integral_result))

    
print('observation time (mins):', obs_time/60, 'mass companion:', max_companion_mass, 'orbital period low (hrs):', p_orb_low_limit/3600, 'orbital period high (hrs):', p_orb_upper_limit/3600, 'spin period (ms):', (1/spin_freq) * 1e+3, 'prob:', \
 probability, 'templates: ', total_templates_required, 'integration error percentage: ', (estimated_volume_integral_error/volume_integral_result) * 100, 'coverage: ', coverage, 'mismatch: ', mismatch, 'phase: ', max_initial_orbital_phase)

if not args.output_filename:
    sys.exit()


ndim = 3
np.random.seed(42)
nwalkers = 800
burn_in_steps = 100
filename = output_path + args.output_filename + '.h5'
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)
angular_velocity_start_values = np.random.uniform(lowest_angular_velocity, highest_angular_velocity, nwalkers)
max_projected_radius = alpha * G**(1/3) * max_companion_mass * M_0 * highest_angular_velocity**(-2/3)/(c * ((min_pulsar_mass + max_companion_mass) * M_0)**(2/3))
projected_radius_start_values = np.random.uniform(0., max_projected_radius, nwalkers)
orbital_phase_start_values = np.random.uniform(0., max_initial_orbital_phase, nwalkers)

initial_guess = np.column_stack((angular_velocity_start_values, projected_radius_start_values, orbital_phase_start_values))

with Pool(processes = ncpus) as pool:
#with MPIPool() as pool:
    # if not pool.is_master():
    #     pool.wait()
    #     sys.exit(0)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, backend=backend, args=[spin_freq, obs_time, lowest_angular_velocity, highest_angular_velocity, max_initial_orbital_phase], pool=pool)
    start = time.time()
    state = sampler.run_mcmc(initial_guess, burn_in_steps)
    end = time.time()
    multi_time = end - start
    print("Burn-In Phase took {0:.1f} seconds".format(multi_time))
    sampler.reset()

    start = time.time()

    # Initialize the convergence condition flag
    converged = False

    # Run the MCMC until convergence condition is met
    while not converged:
        state = sampler.run_mcmc(state, total_templates_required, store=True)

        try:
            # Compute the autocorrelation time
            tau = sampler.get_autocorr_time(quiet=True)
            max_tau = np.max(tau)

            # Check if the number of samples is greater than 50 times the maximum autocorrelation time
            if sampler.iteration > 50 * max_tau:
                converged = True
        except emcee.autocorr.AutocorrError:
            # If the autocorrelation time cannot be computed, continue running the MCMC
            pass

    end = time.time()
    multi_time = end - start
    print("Main Phase took {0:.1f} seconds".format(multi_time))


    
sampler = emcee.backends.HDFBackend(filename)
flatchain = sampler.get_chain(flat=True)
indices = np.random.choice(flatchain.shape[0], total_templates_required, replace=False)
templates_for_search = flatchain[indices]
with open(args.output_filename + '.csv', 'w') as f:

    for i in range(len(templates_for_search)):


        f.write(str(templates_for_search[i][0]) + ' ' + str(templates_for_search[i][1]) + ' ' + \
                str(templates_for_search[i][2]) + '\n')
    
    ''' Adding an extra template in the end with asini/c=0, to cover isolated pulsars! ''' 
    f.write(str(templates_for_search[0][0]) + ' ' + str(0.0) + ' ' + \
                str(templates_for_search[0][2]) + '\n')
    

# Making Corner plots
tau = sampler.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)

# Emcee requires log posterior in natural log units, changed it to base 10 here

log_prob_samples = np.log10(np.exp(sampler.get_log_prob(discard=burnin, flat=True, thin=thin)))

samples[:,0] = (2 * np.pi/samples[:,0])/3600 # Convert angular velocity to porb in hours.
samples[:,2] = np.degrees(samples[:,2]) # Orbital Phase in Degrees




all_samples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
labels=["Orbital Period \n (hrs)", "Projected Radius \n (lt-s)", "Orbital Phase \n (degrees)"]
labels += [r"log$_{10}$ $\left(\sqrt{|det(\gamma_{\alpha \beta})|}\right)$"]

figure = corner.corner(all_samples, labels=labels, color='black', title_kwargs={"fontsize": 12}, \
                       smooth=True, smooth1d=True, scale_hist=True, levels=(0.1175031 , 0.39346934, 0.67534753, 0.86466472), \
                      );

figure.savefig('corner_plot_' + args.output_filename + '.png', dpi=300)

