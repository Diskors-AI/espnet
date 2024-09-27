#!/usr/bin/env bash

set -e
set -u
set -o pipefail

# Function to log messages
log() {
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') $*"
}

# Arguments
src_dir=$1       # Source directory containing the Kaldi-style data files

if [ ! -d "${src_dir}" ]; then
    echo "Source directory ${src_dir} does not exist."
    exit 1
fi

# Split sets you want to create _phn versions for
split_sets="tr_no_dev dev eval1"

log "Starting the preparation of phoneme data sets..."

# Loop over each set and create _phn directories
for dset in ${split_sets}; do
    src_set_dir=${src_dir}/${dset}
    
    if [ ! -d "${src_set_dir}" ]; then
        echo "Source set directory ${src_set_dir} does not exist."
        continue
    fi

    # Copy data directory structure
    utils/copy_data_dir.sh ${src_set_dir}{,_phn}

    # Ensure data directories are validated and properly formatted
    utils/fix_data_dir.sh "${src_set_dir}_phn"
done

log "Phoneme data set preparation complete."