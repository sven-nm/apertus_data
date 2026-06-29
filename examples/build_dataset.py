#%%
from apertus_data.dataset import Dataset
from pathlib import Path

if __name__ == '__main__':
    dataset = Dataset.from_id('xet_test___5bc987c')

    dataset.build(force=True)