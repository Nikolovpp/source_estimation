#!/bin/bash
# Check which subjects are missing percDiff or prodDiff CSV files in a given folder.
# Usage: bash check_missing.sh /path/to/csv/folder

DIR="${1:-.}"

SUBJECTS=(
    EEGPROD4001 EEGPROD4003 EEGPROD4004
    EEGPROD4005 EEGPROD4006 EEGPROD4007
    EEGPROD4008 EEGPROD4009 EEGPROD4010
    EEGPROD4011 EEGPROD4013 EEGPROD4014
    EEGPROD4015 EEGPROD4016 EEGPROD4018
    EEGPROD4019 EEGPROD4020 EEGPROD4021
    EEGPROD4022 EEGPROD4023
)

missing=0
for subj in "${SUBJECTS[@]}"; do
    perc=$(find "$DIR" -name "${subj}_*_percDiff_*.csv" | head -1)
    prod=$(find "$DIR" -name "${subj}_*_prodDiff_*.csv" | head -1)

    if [[ -z "$perc" && -z "$prod" ]]; then
        echo "MISSING BOTH:    $subj"
        missing=1
    elif [[ -z "$perc" ]]; then
        echo "MISSING percDiff: $subj"
        missing=1
    elif [[ -z "$prod" ]]; then
        echo "MISSING prodDiff: $subj"
        missing=1
    fi
done

if [[ $missing -eq 0 ]]; then
    echo "All subjects have both percDiff and prodDiff CSV files."
fi
