# Structural MRI Harmonization via Disentangled Latent Energy-Based Style Translation (DLEST)
This is the official repository for the [DLEST](https://doi.org/10.1007/978-3-031-45673-2_1) paper.

## Train Site-Invariant Image Generation Module
1. Modify the YAML file in the ```/configs``` directory.
2. Modify the ```dataloader.py``` and ```dataset_util.py``` to accommodate your dataset.
3. Modify ```train_alae.py```, set ```default_config=``` to use a specific YAML configuration file.

## Train Site-Specific Style Translation Module
1. Modify ```train_EBM.py```, set ```default_config=``` to use a specific YAML configuration file.
2. Run ```train_EBM.py``` to perform training.

## Test Site-Specific Style Translation Module
1. Modify ```test_EBM.py```, set ```default_config=``` to use YAML configuration file used during training.
2. Run ```test_EBM.py``` to perform style translation.


## Acknowledgement
Our code was inspired by the [latenet-energy-transport](https://github.com/YangNaruto/latent-energy-transport) repository.
