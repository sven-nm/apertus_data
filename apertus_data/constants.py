"""Centralised constants and paths for apertus_data."""

from pathlib import Path
import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CATALOGUE_DIR: Path = PROJECT_ROOT / 'catalogue'
BUILDERS_DIR: Path = PROJECT_ROOT / 'builders' # Todo change this
DATA_DIR: Path = Path('/Users/sven/Desktop/data')  # Todo: ⚠️ change this

SCHEMA_PATH = PROJECT_ROOT / 'apertus_data' / 'dataset_schema.yml'

GITHUB_OWNER = 'sven-nm'  # Todo: ⚠️ change this
GITHUB_REPO = 'apertus_data'  # Todo: ⚠️ change this
GITHUB_RAW_BASE = f'https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}'

YAML_KEYS = yaml.safe_load(SCHEMA_PATH.read_text(encoding='utf-8'))['properties'].keys()

# Todo: 👀  will this be necessary ? Remove eventually.
# MANUAL_YAML_KEYS = [
#     'url',
#     'owner',
#     'name',
#     'version',
#     'tags',
#     'modalities',
#     'languages',
#     'licenses',
#     'formats',
#     'notes'
# ]
#
# AUTOMATIC_YAML_KEYS = [
#     'id',
#     'build_requirements',
#     'build_history',
#     'usage_history'
# ]
