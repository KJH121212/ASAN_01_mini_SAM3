import os
import pandas as pd
from pathlib import Path

#-----------------------------
# path 정리
#-----------------------------
frame_path = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/walking_data/FRAME/frontal__walking__1")
data_path = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
output_path = data_path / "walking_data/sam"

# checkpoint path
bpe_path = data_path / "checkpoints/SAM3/bpe_simple_vocab_16e6.txt.gz"
checkpoint_path = data_path / "checkpoints/SAM3/sam3.pt"

#-----------------------------
# 
#-----------------------------

