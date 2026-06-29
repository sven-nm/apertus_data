#%%
from apertus_data.dataset import Dataset
from pathlib import Path

dataset = Dataset.from_id('starcoderdata___9fc30b5')

dataset.build(force=False)