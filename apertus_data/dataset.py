"""Dataset object: download, process, and tokenize operations."""

#%%
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import os

import requests
import yaml
from jsonschema import ValidationError, validate

from apertus_data import build
from apertus_data import constants as cs
from apertus_data import utils
from apertus_data.utils import get_logger, log_to_file

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
        """Load a Dataset from a catalogue YAML file.

        Performs two safety checks before creating the object:
        1. Asserts the local YAML matches the upstream (GitHub) version.
        2. Validates the YAML against the dataset schema.
        """
        local_data = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))

        dataset_id = local_data.get('id')
        if not dataset_id:
            raise ValueError(f"YAML at {yaml_path} is missing the required 'id' field.")

        upstream_url = f"{cs.GITHUB_API_BASE}/contents/catalogue/{dataset_id}.yaml"
        headers = {"Accept": "application/vnd.github.raw+json"}
        if token := os.getenv("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = requests.get(upstream_url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to fetch upstream YAML from {upstream_url}: {e}")

        upstream_data = yaml.safe_load(response.text)
        if local_data != upstream_data:
            raise ValueError(
                f"Local YAML at {yaml_path} diverges from upstream {upstream_url}.")

        schema = yaml.safe_load(cs.SCHEMA_PATH.read_text(encoding='utf-8'))
        try:
            validate(instance=local_data, schema=schema)
        except ValidationError as e:
            raise ValueError(
                f"Schema validation failed for {yaml_path}: {e.message} (at {list(e.path)})")

        return cls(**local_data)

    @classmethod
    def from_id(cls, id_: str) -> 'Dataset':
        """Load a Dataset by its catalogue id."""
        return cls.from_yaml(cs.CATALOGUE_DIR / f'{id_}.yaml')

    def to_yaml(self) -> None:
        """Write the dataset to a catalogue YAML file."""
        # Todo 👀: should this include the a YAML validation
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
        with log_to_file(self.logs_dir, name=f'{self.id}_main'):
            source_id = getattr(self, 'source_dataset', None)
            if source_id:
                source = Dataset.from_id(source_id)
                if not source._is_already_built():
                    logger.info("Building source dataset %r before %r", source_id, self.id)
                    source.build(force=False)

            req = getattr(self, 'build_requirements', None) or {}
            url, commit = req.get('build_script_url'), req.get('build_script_commit')
            if not url or not commit:
                raise AttributeError(
                    f"Dataset {self.id!r} has no build_script_url/commit in build_requirements."
                )

            main_fn = build.prepare_builder(url, commit, yaml_path=self.yaml_path)

            if not force and self._is_already_built():
                raise ValueError(
                    f"Dataset {self.id} is already built at {self.data_dir}. "
                    f"Use force=True to rebuild."
                )

            if force and self.data_dir.exists():
                logger.warning("force=True - removing %s", self.data_dir)
                shutil.rmtree(self.root_dir)
                self.root_dir.mkdir(parents=True, exist_ok=True)

            build.run_builder(
                main_fn,
                output_dir=self.data_dir,
                logs_dir=self.logs_dir,
                dataset=self,
            )

            # Hash the dataset files
            utils.compute_and_write_files_hashes(
                input_dir=self.data_dir,
                output_dir=self.hashes_dir,
                filename_patterns=['*' + f for f in self.formats],
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

            try:
                subprocess.run(
                    ['git', 'add', str(self.yaml_path)],
                    check=True, capture_output=True, text=True,
                )
                subprocess.run(
                    ['git', 'commit', '-m', f'Update build_history for {self.id}'],
                    check=True, capture_output=True, text=True,
                )
                subprocess.run(
                    ['git', 'push', 'origin', 'main'],
                    check=True, capture_output=True, text=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Git operation failed for {self.yaml_path}:\n{e.stderr}. Please push the updated yaml manually, so upstream `build_history` is updated."
                ) from e




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
