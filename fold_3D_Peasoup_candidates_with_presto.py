import xml.etree.ElementTree as ET
import sys, os, subprocess
import argparse
import pandas as pd
import numpy as np
parser = argparse.ArgumentParser(description='Fold Specific 3D Peasoup candidates for quick inspection')
parser.add_argument('-o', '--output_path', help='Output path to save results',  default=os.getcwd(), type=str)
parser.add_argument('-i', '--input_file', help='Name of the input xml file', type=str)
parser.add_argument('-c', '--candidate_id', help='Candidate id to fold', type=int)
parser.add_argument('-m', '--mask_file', help='Mask file for prepfold', type=str)

args = parser.parse_args()

if not args.input_file:
    print("You need to provide an xml file to read")
    sys.exit()

xml_file = args.input_file
candidate_id_to_fold = args.candidate_id
tree = ET.parse(xml_file)
root = tree.getroot()

header_params = root[1]
search_params = root[2]
candidates = root[5]
tstart = float(header_params.find("tstart").text) 
filterbank_file = str(search_params.find("infilename").text)

ignored_entries = ['candidate', 'opt_period', 'folded_snr', 'byte_offset', 'is_adjacent', 'is_physical', 'ddm_count_ratio', 'ddm_snr_ratio']
rows = []
for candidate in candidates:
    cand_dict = {}
    for cand_entry in candidate.iter():
        if not cand_entry.tag in ignored_entries:
            cand_dict[cand_entry.tag] = cand_entry.text
   
    cand_dict['cand_id_in_file'] = candidate.attrib.get("id")
    rows.append(cand_dict)

df = pd.DataFrame(rows)

os.chdir(args.output_path)

for index, row in df.iterrows():
    if row['cand_id_in_file'] == str(candidate_id_to_fold):
        orbital_period_seconds = 2 * np.pi/float(row['omega'])
        orbital_period_days = orbital_period_seconds/(3600 * 24)
        normalised_orbital_phase = float(row['phi'])/(2 * np.pi)
        T0 = tstart + ((1.50 - normalised_orbital_phase) * orbital_period_days)
        output_filename = str(header_params.find("source_name").text) + '_3D_Peasoup_fold_candidate_id_' + str(row['cand_id_in_file'])
        
        if args.mask_file:
            cmds = "prepfold -fixchi -nsub 128 -noxwin -n 64 -mask %s -topo -p %.12f -dm %.5f -bin -pb %.12f -x %.12f -To %.12f -o %s %s" %(args.mask_file, float(row['period']), float(row['dm']), orbital_period_seconds, float(row['tau']), T0, output_filename, filterbank_file)
        else:
            cmds = "prepfold -fixchi -nsub 128 -noxwin -n 64 -topo -p %.12f -dm %.5f -bin -pb %.12f -x %.12f -To %.12f -o %s %s" %(float(row['period']), float(row['dm']), orbital_period_seconds, float(row['tau']), T0, output_filename, filterbank_file)
        subprocess.call(cmds, shell=True)
    

