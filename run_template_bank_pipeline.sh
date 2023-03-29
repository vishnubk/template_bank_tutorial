#!/bin/bash

obs_file='/hercules/scratch/vishnu/MeerKAT_TUTORIAL/2012-04-12-16:27:35.fil'
search_config='template_bank_search.cfg'
pointing=${obs_file::-4}
beam=00
#Parse the slurm config file
get_config_value() {
    local section="$1"
    local option="$2"
    awk -F ':\\s+' -v sec="$section" -v opt="$option" '
        $0 ~ "^\\[" sec "\\]" { f = 1 }
        f && $1 == opt {
            sub(/#.*$/, "", $2)  # remove comments
            gsub(/"/, "", $2)    # remove quotes
            if ($2 ~ /^[0-9]+(\.[0-9]+)?$/) {
                # numeric value
                if ($2 ~ /\./) {
                    # float
                    printf "%.5f", $2
                } else {
                    # integer
                    printf "%d", $2
                }
            } else {
                # string value
                if (opt == "WALL_CLOCK_TIME") {
                    split($2, t, ":")
                    if (length(t) == 2) {
                        # time in HH:MM format
                        printf "%s:00", $2
                    } else {
                        # time in HH:MM:SS format
                        print $2
                    }
                } else {
                    print $2
                }
            }
            exit
        }' "$search_config"
}
presto_image=$(get_config_value "Singularity_Image" "PRESTO_IMAGE_FOLDING")
template_bank_software_image=$(get_config_value "Singularity_Image" "Template_Bank_Singularity_Image_Path")
mount_path=$(get_config_value "Singularity_Image" "MOUNT_PATH")
code_directory=$(get_config_value "Singularity_Image" "CODE_DIRECTORY_ABS_PATH")


singularity exec -H $HOME:/home1 -B $mount_path:$mount_path $presto_image python $code_directory/create_dm_file_from_ddplan.py -o $obs_file -c $search_config 

# Get binary parameter search range
porb_min=$(get_config_value "BINARY_PARAMETER_SEARCH_RANGE" "MIN_PORB")
porb_max=$(get_config_value "BINARY_PARAMETER_SEARCH_RANGE" "MAX_PORB")
pulsar_mass_min=$(get_config_value "BINARY_PARAMETER_SEARCH_RANGE" "MIN_PULSAR_MASS")
companion_mass_max=$(get_config_value "BINARY_PARAMETER_SEARCH_RANGE" "MAX_COMPANION_MASS")
fraction=$(get_config_value "BINARY_PARAMETER_SEARCH_RANGE" "FRAC_COVERAGE_INCLINATION_ANGLE")

# Get template bank parameters
coverage=$(get_config_value "TEMPLATE_BANK_PARAMETERS" "COVERAGE")
mismatch=$(get_config_value "TEMPLATE_BANK_PARAMETERS" "MISMATCH")
fastest_spin_period=$(get_config_value "TEMPLATE_BANK_PARAMETERS" "FASTEST_SPIN_PERIOD_MS")
obs_time=$(get_config_value "TEMPLATE_BANK_PARAMETERS" "OBS_TIME_MIN")

singularity exec -H $HOME:/home1 -B $mount_path:$mount_path $template_bank_software_image python3 $code_directory/create_circular_orbit_search_template_bank_emcee.py -t $obs_time -p $porb_min -P $porb_max -d $pulsar_mass_min -c $companion_mass_max -f $fraction -b $coverage -m $mismatch -s $fastest_spin_period -n 24 -file ${obs_file::-4}_template_bank -o $code_directory

#template_bank=htru_low_lat_seventy_two_minute_full_stochastic_gpu_format
template_bank=sample_3D_template_bank.txt
#singularity exec -H $HOME:/home1 -B $mount_path:$mount_path $template_bank_software_image circular_orbit_template_bank_peasoup -i $obs_file -T ${obs_file::-4}_template_bank.csv --dm_file ${obs_file::-4}_dm_file.txt -limit 100000 -m 9.0 

SEARCH_RESULTS_OUTPUT=$code_directory/$pointing/$beam/01_SEARCH/
mkdir -p $SEARCH_RESULTS_OUTPUT
search=$(sbatch --parsable --output=search_logs_$pointing_$beam.out --error=search_logs_$pointing_$beam.err -J TEMPLATE_BANK_SEARCH --gres=gpu:1 -p short.q -t 4:00:00 --wrap="singularity exec --nv -H $HOME:/home1 -B $mount_path:$mount_path $template_bank_software_image circular_orbit_template_bank_peasoup -i $obs_file -T $template_bank --dm_file ${obs_file::-4}_dm_file.txt --limit 100000 -m 9.0 --ram_limit_gb 10 -o $SEARCH_RESULTS_OUTPUT")

FOLD_RESULTS_OUTPUT=$code_directory/$pointing/$beam/02_FOLD/
mkdir -p $FOLD_RESULTS_OUTPUT

#Fold Candidates with PRESTO
sbatch -J FOLD -p short.q --dependency=afterok:$search --output=fold_logs_$pointing_$beam.out --error=fold_logs_$pointing_$beam.err -t 4:00:00 --wrap="singularity exec -H $HOME:/home1 -B $mount_path:$mount_path $presto_image python $code_directory/fold_all_3D_Peasoup_candidates_with_presto.py -i $SEARCH_RESULTS_OUTPUT/3D_template_bank_peasoup_candidates.xml -o $FOLD_RESULTS_OUTPUT" 



