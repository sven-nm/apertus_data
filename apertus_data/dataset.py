"""Dataset object: download, process, and tokenize operations."""

#%%
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from apertus_data import build
from apertus_data import constants as cs
from apertus_data import utils
from apertus_data.utils import get_logger

logger = get_logger(__name__)


class Dataset:
    """A dataset described by a catalogue YAML file."""

    def __init__(self, **kwargs) -> None:
        """Initialize from a parsed catalogue entry."""

        # Todo: clean this (-> Pydantic BaseModel or dataclass) and validate against schema ?
        for key, value in kwargs.items():
            # Quick and dirty hack for paths
            if key.endswith('_dir') or key.endswith('_path'):
                value = Path(value)
            setattr(self, key, value)

        self.yaml_path: Path = cs.CATALOGUE_DIR / f'{self.id}.yaml'

        self.is_hf_dataset: bool = self.url.startswith('https://huggingface.co/datasets')


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
                     for k in cs.YAML_KEYS if hasattr(self, k)}

        self.yaml_path.write_text(yaml.safe_dump(yaml_dict), encoding='utf-8')

    def get_local_commit(self) -> str:
        """Get the commit hash of the local dataset, read from on-disk metadata."""
        if self.is_hf_dataset:
            hf_metadata_path = self.data_dir / '.cache/huggingface/download/.gitattributes.metadata'
            if not hf_metadata_path.exists():
                raise FileNotFoundError(
                    f"Huggingface metadata file not found at {hf_metadata_path}; "
                    f"check the dataset id and revision."
                )
            return hf_metadata_path.read_text().strip().split('\n')[0]
        else:
            raise NotImplementedError


    def build(self, force: bool = False) -> None:
        """Build the dataset by running its pinned ``main()`` build script.

        Flow:
        1. Verify ``build_requirements.build_script_url`` exists on GitHub at
           ``build_requirements.build_script_commit`` (HEAD on raw URL).
        2. Verify the local script is at that commit (or check it out from
           file history if it isn't); raise if the commit isn't in history.
        3. Import the script and validate ``main()``'s signature.
        4. Check ``force`` and :meth:`_is_already_built`; wipe ``data_dir``
           if ``force`` is set.
        5. Run ``main(output_dir=data_dir, logs_dir=logs_dir)`` under
           :func:`build.run_builder`'s contract (log handler attached,
           non-empty output asserted).
        6. Hash the produced files and append a build_history entry, then
           persist to YAML.

        Args:
            force: If True, wipe ``self.data_dir`` and re-run even when the
                dataset already appears built at the requested commit.
        """
        req = getattr(self, 'build_requirements', None) or {}
        url, commit = req.get('build_script_url'), req.get('build_script_commit')
        if not url or not commit:
            raise AttributeError(
                f"Dataset {self.id!r} has no build_script_url/commit in build_requirements."
            )

        main_fn = build.prepare_builder(url, commit)

        if not force and self._is_already_built():
            raise ValueError(
                f"Dataset {self.id} is already built at {self.data_dir}. "
                f"Use force=True to rebuild."
            )

        if force and self.data_dir.exists():
            logger.warning("force=True - removing %s", self.data_dir)
            shutil.rmtree(self.data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)

        build.run_builder(
            main_fn,
            output_dir=self.data_dir,
            logs_dir=self.logs_dir,
        )

        # Hash the dataset files
        utils.compute_and_write_files_hashes(
            input_dir=self.data_dir,
            output_dir=self.hashes_dir,
            filename_patterns=['*' + f for f in self.file_formats],
        )

        # Single hash for the whole dataset (digest of the per-file hashes)
        dataset_hash = utils.compute_directory_hash(
            input_dir=self.hashes_dir,
            output_path=self.hashes_dir / f'{self.id}.hash',
        )

        self.build_history.append({
            'datetime': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'hash': dataset_hash,
        })

        self.to_yaml()




    def _is_already_built(self) -> bool:
        """Return True if ``self`` already appears downloaded at the requested commit."""
        if not self.build_history:
            return False
        if not self.data_dir.exists() or not any(self.data_dir.iterdir()):
            return False
        try:
            return self.get_local_commit() == self.version
        except (FileNotFoundError, NotImplementedError):
            return False
