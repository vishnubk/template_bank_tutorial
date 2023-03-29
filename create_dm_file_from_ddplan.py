from pulsar_miner import SurveyConfiguration, Observation, get_command_output, get_DD_scheme_from_DDplan_output, Observation
import configparser, sys, subprocess, os, math, argparse
import numpy as np


parser = argparse.ArgumentParser(description='Convert DDplan to Peasoup DM file')
parser.add_argument('-c', '--template_bank_config', help='Template Bank Search Configuration file', type=str,  default="template_bank_search.cfg")
parser.add_argument('-o', '--obs', help='Observation file ', type=str,  required=True)
args = parser.parse_args()

def run_ddplan(low_dm, high_dm, central_frequency, bandwidth, nchans, sampling_time, output_file, out_dir, coherent_dedisp_dm = 0):
    if coherent_dedisp_dm!=0:
        cmd_DDplan = 'DDplan.py -o ddplan_%s -l %.2f -d %.2f -c %.2f -f %.2f -b %d -n %d -t %.6f' % (output_file, low_dm, high_dm, coherent_dedisp_dm, central_frequency, abs(bandwidth), nchans, sampling_time)
    else:
        cmd_DDplan = 'DDplan.py -o ddplan_%s -l %.2f -d %.2f -f %.2f -b %d -n %d -t %.6f' % (output_file, low_dm, high_dm, central_frequency, abs(bandwidth), nchans, sampling_time)
    output_DDplan    = get_command_output(cmd_DDplan, shell_state=False, work_dir=out_dir)
    list_DD_schemes  = get_DD_scheme_from_DDplan_output(output_DDplan)
    
    return list_DD_schemes

#template_bank_config_file = 'template_bank_search.cfg'
#observation_file = '2012-04-12-16:27:35.fil'
template_bank_config_file = args.template_bank_config

observation_file = args.obs
config_template_bank = configparser.ConfigParser()
config_template_bank.read(template_bank_config_file)

coverage = float(config_template_bank['TEMPLATE_BANK_PARAMETERS']['COVERAGE'])
mismatch = float(config_template_bank['TEMPLATE_BANK_PARAMETERS']['MISMATCH'])
fastest_spin_period = float(config_template_bank['TEMPLATE_BANK_PARAMETERS']['FASTEST_SPIN_PERIOD_MS']) * 1e-3
low_dm = float(config_template_bank['DM_SEARCH_RANGE']['DM_MIN'])
high_dm = float(config_template_bank['DM_SEARCH_RANGE']['DM_MAX'])
coherent_dedisp_dm = float(config_template_bank['DM_SEARCH_RANGE']['DM_COHERENT_DEDISPERSION'])

bandwidth = int(subprocess.check_output('''readfile %s  | grep "Total Bandwidth" | awk '{print $NF}' ''' % (observation_file), shell=True))
central_frequency = float(subprocess.check_output('''readfile %s  | grep "Central freq (MHz)" | awk '{print $NF}' ''' % (observation_file), shell=True))
nchans = int(subprocess.check_output('''readfile %s  | grep "Number of channels" | awk '{print $NF}' ''' % (observation_file), shell=True))
sampling_time = float(subprocess.check_output('''readfile %s  | grep "Sample time (us)" | awk '{print $NF}' ''' % (observation_file), shell=True)) * 1e-6
tobs = int(math.ceil(float(subprocess.check_output('''readfile %s  | grep "Time per file (sec)" | awk '{print $NF}' ''' % (observation_file), shell=True))))

ddplan_results = run_ddplan(low_dm, high_dm, central_frequency, bandwidth, nchans, sampling_time, 'ddplan_plot', os.getcwd(), coherent_dedisp_dm)
ndm_trials = []
downsamp = []
dDM = []
low_dm = []
high_dm = []
for i in range(len(ddplan_results)):
    ndm_trials.append(ddplan_results[i]['num_DMs'])
    downsamp.append(ddplan_results[i]['downsamp'])
    dDM.append(ddplan_results[i]['dDM'])
    low_dm.append(ddplan_results[i]['loDM'])
    high_dm.append(ddplan_results[i]['highDM'])

print(low_dm, high_dm, ndm_trials, dDM)
with open ('%s_dm_file.txt' %observation_file[:-4], 'w') as f:
    for i in range(len(low_dm)):
        dm_trials = np.around(np.linspace(low_dm[i], high_dm[i], ndm_trials[i]), 2)
        for j in range(len(dm_trials)):
            f.write(str(dm_trials[j]) + '\n')
         

