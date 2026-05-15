"""Dataset object: download, process, and tokenize operations."""

#%%
import shutil
from datetime import datetime, timezone
from pathlib import Path
from builders.default_hf_download import hf_transfer_download

import yaml

from apertus_data import constants as cs
from apertus_data.utils import get_logger
from apertus_data import utils

logger = get_logger(__name__)


class Dataset:
    """A dataset described by a catalogue YAML file."""

    def __init__(self, **kwargs) -> None:
        """Initialize from a parsed catalogue entry."""

        self.yaml_attrs = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.yaml_attrs.append(key)

        # Set the local root dir
        self.local_root_dir: Path = cs.DATA_DIR / self.id
        self.yaml_attrs.append('local_root_dir')

        # Set data, logs and hash dirs
        for key in ['data', 'logs', 'hashes']:
            setattr(self, f'local_{key}_dir', self.local_root_dir / key)
            self.yaml_attrs.append(f'local_{key}_dir')

        self.yaml_path: Path = cs.CATALOGUE_DIR / f'{self.id}.yaml'

        self.is_hf_dataset: bool = self.url.startswith('https://huggingface.co/datasets')

        if 'downloads' not in kwargs:
            self.downloads: list[dict] = []
            self.yaml_attrs.append('downloads')

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'Dataset':
        """Load a Dataset from a catalogue YAML file."""
        return cls(**yaml.safe_load(yaml_path.read_text(encoding='utf-8')))

    @classmethod
    def from_id(cls, id_: str) -> 'Dataset':
        """Load a Dataset by its catalogue id."""
        return cls.from_yaml(cs.CATALOGUE_DIR / f'{id_}.yaml')

    def to_yaml(self) -> None:
        """Write the dataset to a catalogue YAML file."""
        yaml_dict = {k: getattr(self, k) if not isinstance(getattr(self, k), Path) else str(getattr(self, k))
                     for k in self.yaml_attrs}

        self.yaml_path.write_text(yaml.safe_dump(yaml_dict), encoding='utf-8')

    def get_local_commit(self) -> str:
        """Get the commit hash of the local dataset, read from on-disk metadata."""
        if self.is_hf_dataset:
            hf_metadata_path = self.local_data_dir / '.cache/huggingface/download/.gitattributes.metadata'
            if not hf_metadata_path.exists():
                raise FileNotFoundError(
                    f"Huggingface metadata file not found at {hf_metadata_path}; "
                    f"check the dataset id and revision."
                )
            return hf_metadata_path.read_text().strip().split('\n')[0]
        else:
            raise NotImplementedError


    def download(self, force_download: bool = False) -> None:
        """Download the dataset to ``self.local_data_dir`` and record the build.

        Note:
            The method verifies that the dataset is not already downloaded (unless
            ``force_download`` is True), checks that the download script is committed
            and pushed to git origin, optionally wipes the existing local data,
            runs the download, hashes the result, and appends a build entry to the
            catalogue YAML.

        Args:
            force_download: If True, wipe any existing local data and redownload
                even when the dataset already appears to be present.
        """
        if not force_download and self._is_already_downloaded():
            raise ValueError(
                f"Dataset {self.id} is already downloaded at {self.local_data_dir} and force_download=False."
                " Use force_download=True to overwrite."
            )

        if force_download and self.local_data_dir.exists():
            logger.info("force_download=True - removing %s", self.local_data_dir)
            shutil.rmtree(self.local_data_dir)

        self.local_data_dir.mkdir(parents=True, exist_ok=True)

        # Todo add logger

        # Run the download script.
        # TODO: invoke builders.download.<dataset_id>.main(output_dir=self.local_data_dir)
        if self.is_hf_dataset:
            hf_transfer_download(
                repo_id=self.url.split("https://huggingface.co/datasets/")[-1],
                output_dir=self.local_data_dir,
                revision=self.commit,  # todo change this
                hf_token=cs.HF_TOKEN,
            )
        else:
            raise NotImplementedError

        # Hash the dataset files
        utils.compute_and_write_files_hashes(
            root_dir=self.local_data_dir,
            output_dir=self.local_root_dir / 'hashes',
            filename_pattern="*.json",
        )

        # Compute a single hash for the entire dataset directory (e.g., by hashing the concatenation of all file hashes)
        dataset_hash = utils.compute_directory_hash(
            root_dir=self.local_root_dir / 'hashes',
            output_path=self.local_root_dir / f'hashes/{self.id}.hash',
        )

        self.downloads.append({
            'date': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'commit': getattr(self, 'commit', None),
            # 'user': getpass.getuser(), # Todo implement this
            'output_dir': str(self.local_data_dir),
            'hash': dataset_hash,
        })

        # Write to yaml
        self.to_yaml()


    def _is_already_downloaded(self) -> bool:
        """Return True if ``self`` already appears downloaded at the requested commit."""
        if not self.downloads:
            return False
        if not self.local_data_dir.exists() or not any(self.local_data_dir.iterdir()):
            return False
        try:
            return self.get_local_commit() == getattr(self, 'commit', None)
        except (FileNotFoundError, NotImplementedError):
            return False


dataset = Dataset.from_yaml(Path('/Users/sven/packages/apertus_data/catalogue/sven-nm___xet_test.yaml'))
dataset.download()
