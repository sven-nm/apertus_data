#%%
from apertus_data.dataset import Dataset
from pathlib import Path

dataset = Dataset.from_id('sven-nm___xet_test___5bc987c')

dataset.build(force=True,)