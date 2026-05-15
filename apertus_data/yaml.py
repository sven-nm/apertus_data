#%%
"""YAML enrichment and validation for apertus_data datasets.

This script performs automatic field cleaning, generates deterministic IDs,
issues warnings for common problems, enforces hard validation rules,
and runs final JSON Schema validation.
"""

import sys
import re
import yaml
from jsonschema import ValidationError, validate
import requests
from pathlib import Path
from typing import Dict, List


from apertus_data import constants as cs
from apertus_data.utils import get_logger

logger = get_logger(__name__)


def is_commit_hash(string: str) -> bool:
    """Check if a string is a valid 40-character git commit hash."""
    return bool(re.fullmatch(r"[0-9a-f]{40}", string))


def clean_version(version: str | None) -> str | None:
    """Clean and normalize version string."""
    if not version:
        return None
    # Remove trailing spaces, collapse whitespace, replace with _
    cleaned = re.sub(r'\s+', ' ', str(version).strip())
    cleaned = cleaned.replace(' ', '_')
    return cleaned


def generate_id(owner: str, name: str, version: str | None) -> str:
    """Generate deterministic dataset ID in the format `owner___name___version`."""
    if version:
        version_part = version[:7] if is_commit_hash(version) else version
    else:
        version_part = "0000000"

    return f"{owner}___{name}___{version_part}"


# ====================== WARNINGS (non-blocking) ======================
def check_url_rules(data: dict) -> list[str]:
    """Perform soft checks on URL field and return warnings."""
    warnings = []

    # Warning: external dataset without URL
    if not data['url'] and data['owner'] != "swiss-ai":
        warnings.append("WARNING: `url` is missing but `owner` is not `swiss-ai`.")

    if not data['url'] or not isinstance(data['url'], str):
        warnings.append("WARNING: `url` is not provided.")
        return warnings

    # HF check
    if "huggingface.co/datasets" in data['url']:
        expected = f"https://huggingface.co/datasets/{data['owner']}/{data['name']}"
        if data['url'] != expected:
            warnings.append(f"WARNING: HF `url` should usually be: {expected}. Please double check the spelling.")

    # GitHub check
    elif "github.com" in data['url']:
        if data['owner'] not in data['url'] or data['name'] not in data['url']:
            warnings.append("WARNING: GitHub `url` should usually contain both data['owner'] and dataset data['name'].")


    # Kaggle check
    elif "kaggle.com/datasets" in data['url']:
        expected = f"https://www.kaggle.com/datasets/{data['owner']}/{data['name']}"
        if data['url'] != expected:
            warnings.append(f"WARNING: Kaggle `url` should usually be: {expected}")


    # General suspicion
    else:
        if data['owner'] not in data['url'] or data['name'] not in data['url']:
            warnings.append("WARNING: `url` does not contain data['owner']/data['name'].")

    return warnings


# ====================== HARD ERRORS (will fail validation) ======================
def check_version_rules(data: Dict) -> List[str]:
    """Enforce strict version rules for version-controlled sources.

    Returns list of error messages (empty if all checks pass).
    """
    errors = []
    url = data.get("url", "")

    if "huggingface.co" in url or "github.com" in url:
        if not data.get('version'):
            errors.append("ERROR: Datasets from HF or GitHub MUST have a 'version' field (full commit hash).")
        elif not is_commit_hash(data['version']):
            errors.append("ERROR: HF/GitHub datasets MUST use a full 40-character commit hash as 'version'.")
    return errors


# Todo: not sure we want to do this first. User may want to require validation on a dataset before using it.
def check_build_requirements(data: Dict) -> List[str]:
    """Enforce existence of source datasets and build script on GitHub.

    Returns list of error messages (empty if all checks pass).
    """
    errors = []
    req = data["build_requirements"]

    # Source datasets must exist
    for src_id in req["source_datasets_ids"]:
        if src_id:
            url = f"{cs.GITHUB_RAW_BASE}/main/apertus_data/catalogue/{src_id}.yaml"
            try:
                if requests.head(url, timeout=10).status_code != 200:
                    errors.append(f"ERROR: Source dataset '{src_id}' not found on GitHub main branch. Please create an issue with the source datasets first.")
            except:
                errors.append(f"ERROR: Could not verify source dataset '{src_id}' on GitHub.")

    # Build script must exist at the given commit
    script_url = req.get("build_script_url")
    commit = req.get("build_script_commit")

    if script_url and commit:
        try:
            # Extract path after /blob/xxxx/
            rel_path = script_url.split("/blob/")[-1].split("/", 1)[-1]
            check_url = f"https://raw.githubusercontent.com/{cs.GITHUB_OWNER}/{cs.GITHUB_REPO}/{commit}/{rel_path}"

            if requests.head(check_url, timeout=10).status_code != 200:
                errors.append(f"ERROR: Build script not found at commit {commit[:7]} on GitHub. You must commit the dataset's build script to GitHub before proceeding.")
        except:
            errors.append("ERROR: Could not verify build script on GitHub.")

    return errors


def main(yaml_path: Path):
    """Main entry point for dataset YAML enrichment and validation."""
    logger.info(f"Processing: {yaml_path.name}")

    # 1. Load YAML # Todo see how yaml path behaves with a yaml coming from an issue
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to parse YAML: {e}")
        sys.exit(1)

    original_data = data.copy()

    # 2. Automatic enhancements
    data["version"] = clean_version(data.get("version"))

    # Generate id (only if missing or empty)
    data["id"] = generate_id(data["owner"], data["name"], data["version"])

    # === Run checks ===
    warnings = check_url_rules(data)
    errors = []
    errors.extend(check_version_rules(data))
    # errors.extend(check_build_requirements(data))  # Todo: We exclude this for now. see above.

    # Log warnings
    for w in warnings:
        logger.warning(w)

    # Log errors
    for e in errors:
        logger.error(e)

    # Fail if any hard errors
    if errors:
        logger.error("❌ Validation failed due to hard errors.")
        sys.exit(1)

    # Final JSON Schema validation
    try:
        schema = yaml.safe_load(cs.SCHEMA_PATH.read_text(encoding="utf-8"))
        validate(instance=data, schema=schema)
        logger.info("✅ YAML schema validation passed.")
    except ValidationError as e:
        logger.error(f"❌ Schema validation failed: {e.message}")
        logger.error(f"   Location: {list(e.path)}")
        sys.exit(1)


main(Path("/Users/sven/packages/apertus_data/catalogue/sven-nm___test_new_yaml.yaml"))