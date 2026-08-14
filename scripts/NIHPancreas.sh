export PYTHONNOUSERSITE=1

export nnUNet_raw="/content/drive/MyDrive/nnUNet_raw_data"
export nnUNet_preprocessed="/content/drive/MyDrive/nnUNet_preprocessed"
export nnUNet_results="/content/drive/MyDrive/nnUNet_results"
nnUNetv2_plan_and_preprocess -d 800 -preprocessor_name IdentityPreprocessor --verify_dataset_integrity