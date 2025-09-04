# Near-Lossless Astronomical Data Compression via RL-Enhanced PCA

This repository is the Pytorch implementation of the paper "[Near-Lossless Astronomical Data Compression via RL-Enhanced PCA]" (xxx):

## Setup

### Installation
Clone this repository:

```bash
git clone https://github.com/LabShuHangGU/CCA.git
```
Create new environment and install required dependencies
```
conda create --name RL-PCA python=3.8
pip install requirements.txt
```

## DataSet
We have provided some experimental data in the data folder, including int8, int16, int32, and floating-point data. The random seed for all provided data is set to 20. If you need to test other data, you must adjust the random seed according to the data type and resolution.

## Compression/Decompression
After changing the file path (e.g., data/int8), simply run “RL-PCA.py”.