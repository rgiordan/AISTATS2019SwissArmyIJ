#!/bin/bash

# Execute the notebooks and save them in $OUTPUT_DIR as pdf files.  They can
# then be copied to the AISTATS paper directory to be included in the
# appendix.

# Make sure your virtual environment is activated before running this script.

OUTPUT_DIR="."

echo "Converting step 1."
jupyter nbconvert --to pdf --execute --ExecutePreprocessor.timeout=600 \
    fit_model_and_save.ipynb \
    --output ${OUTPUT_DIR}"/"fit_model_and_save

echo "Converting step 2."
jupyter nbconvert --to pdf --execute --ExecutePreprocessor.timeout=600 \
    load_and_refit.ipynb \
    --output ${OUTPUT_DIR}"/"load_and_refit

echo "Converting step 3."
jupyter nbconvert --to pdf --execute --ExecutePreprocessor.timeout=600 \
    calculate_prediction_errors.ipynb \
    --output ${OUTPUT_DIR}"/"calculate_prediction_errors

echo "Converting step 4."
cd R
jupyter nbconvert --to pdf --execute --ExecutePreprocessor.timeout=600 \
    examine_and_save_results.ipynb \
    --output ${OUTPUT_DIR}"/"examine_and_save_results

cd ..
