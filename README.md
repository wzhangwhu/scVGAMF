## Installations

# This code neeeds cuda11.8, which can be downloaded from the link
https://developer.nvidia.com/cuda-11-8-0-download-archiv


conda create -n scVGAMF python=3.10
conda activate scVGAMF
# Please make sure to change the version to match the version of your GPU/CPU machine exactly.
pip install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt


## Running demo

import pandas as pd
import scVGAMF

# Each row represents a gene, and each column corresponds to a cell.
data = pd.read_csv("Data/Zeisel/Zeisel.csv", encoding="utf-8", index_col=0)
impute_data = scVGAMF.run_impute(data)
# obtain the imputed matrix
impute_data.to_csv('Data/Zeisel/impute.csv')
