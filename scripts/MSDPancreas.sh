# export PYTHONNOUSERSITE=1

export nnUNet_raw="/content/nnUNet_raw_data"
export nnUNet_preprocessed="/content/nnUNet_preprocessed"
export nnUNet_results="/content/nnUNet_results"
nnUNetv2_plan_and_preprocess -d 801 -preprocessor_name IdentityPreprocessor --verify_dataset_integrity
