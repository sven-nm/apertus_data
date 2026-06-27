"""YAML enrichment and validation for apertus_data datasets.

Loads a dataset description either from an existing YAML file or from a
GitHub Issue Form body, applies automatic field cleaning, generates a
deterministic ID, issues warnings for common problems, enforces hard
validation rules, and runs final JSON Schema validation.

Run as ``python -m apertus_data.yaml_processing --yaml-path <path>`` to
validate an existing YAML, or with ``--issue-body-file <path>`` /
``--issue-body <body>`` to ingest a GitHub Issue Form body and write the
resulting YAML into the catalogue.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml
from jsonschema import ValidationError, validate

from apertus_data import constants as cs
from apertus_data.utils import get_logger

logger = get_logger(__name__)

NO_RESPONSE = "_No response_"


def is_commit_hash(string: str) -> bool:
    """Check if a string is a valid 40-character git commit hash."""
    return bool(re.fullmatch(r"[0-9a-f]{40}", string))


def clean_version(version: str | None) -> str | None:
    """Clean and normalize version string."""
    if not version:
        return None
    cleaned = re.sub(r'\s+', ' ', str(version).strip())
    cleaned = cleaned.replace(' ', '_')
    return cleaned


# Todo adapt this for processed datasets
def generate_id(data: dict) -> str:
    """Generate deterministic dataset ID in the format ``name___version``."""
    if data['version']:
        version_part = data['version'][:7] if is_commit_hash(data['version']) else data['version']
    else:
        version_part = "0000000"

    return f"{data['name']}___{version_part}"


# ====================== ISSUE-FORM PARSING ======================
def _split_issue_sections(body: str) -> dict[str, str]:
    """Split an issue body into ``{header: raw_text}`` using ``### `` headers."""
    sections: dict[str, str] = {}
    current_header: str | None = None
    current_lines: list[str] = []

    for line in body.splitlines():
        match = re.match(r"^###\s+(.+?)\s*$", line)
        if match:
            if current_header is not None:
                sections[current_header] = "\n".join(current_lines).strip()
            current_header = match.group(1).strip()
            current_lines = []
        elif current_header is not None:
            current_lines.append(line)

    if current_header is not None:
        sections[current_header] = "\n".join(current_lines).strip()

    return sections


def _parse_checkboxes(raw: str) -> list[str]:
    """Extract checked labels from a checkbox section (``- [X] label``)."""
    selected: list[str] = []
    for line in raw.splitlines():
        match = re.match(r"^\s*-\s*\[(.)\]\s*(.+?)\s*$", line)
        if match and match.group(1).strip().lower() == "x":
            selected.append(match.group(2).strip())
    return selected


def _coerce_array(raw: str, issue_type: str) -> list[str]:
    """Coerce a raw issue-form section into a list of strings.

    ``textarea`` fields use one entry per line; ``input`` fields are split on
    commas; ``checkboxes`` fields use ``- [X] label`` rows.
    """
    if issue_type == "checkboxes":
        return _parse_checkboxes(raw)
    if issue_type == "textarea":
        return [item.strip() for item in raw.splitlines() if item.strip()]
    # input (single-line, may carry comma-separated values)
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_issue_body(body: str) -> dict[str, Any]:
    """Parse a GitHub Issue Form body into a dataset dict.

    The body uses ``### <field>`` headers followed by the user's response, or
    ``_No response_`` for empty optional fields. Field coercion (scalar vs
    list, checkbox vs textarea vs input) is driven by the dataset schema's
    ``x-git-issue-type`` annotations so this parser stays in sync with the
    issue template.

    Automatic catalogue fields (``build_requirements``, ``build_history``,
    ``usage_history``) are pre-populated with empty defaults; ``id`` is left
    out and is generated later by :func:`enrich_and_validate`.
    """
    schema = yaml.safe_load(cs.SCHEMA_PATH.read_text(encoding="utf-8"))
    properties = schema["properties"]
    sections = _split_issue_sections(body)

    data: dict[str, Any] = {}
    for field_name, prop in properties.items():
        if field_name not in sections:
            continue

        raw = sections[field_name]
        if raw == "" or raw == NO_RESPONSE:
            data[field_name] = None
            continue

        prop_type = prop.get("type")
        is_array = prop_type == "array" or (isinstance(prop_type, list) and "array" in prop_type)
        if is_array:
            data[field_name] = _coerce_array(raw, prop.get("x-git-issue-type", "textarea"))
        else:
            data[field_name] = raw

    return data


# ====================== WARNINGS (non-blocking) ======================
def check_url_rules(data: dict) -> list[str]:
    """Perform soft checks on URL field and return warnings."""
    warnings = []

    if not data['url'] and data['owner'] != "swiss-ai":
        warnings.append("WARNING: `url` is missing but `owner` is not `swiss-ai`.")

    if not data['url'] or not isinstance(data['url'], str):
        warnings.append("WARNING: `url` is not provided.")
        return warnings

    if "huggingface.co/datasets" in data['url']:
        expected = f"https://huggingface.co/datasets/{data['owner']}/{data['name']}"
        if data['url'] != expected:
            warnings.append(f"WARNING: HF `url` should usually be: {expected}. Please double check the spelling.")

    elif "github.com" in data['url']:
        if data['owner'] not in data['url'] or data['name'] not in data['url']:
            warnings.append("WARNING: GitHub `url` should usually contain both data['owner'] and dataset data['name'].")

    elif "kaggle.com/datasets" in data['url']:
        expected = f"https://www.kaggle.com/datasets/{data['owner']}/{data['name']}"
        if data['url'] != expected:
            warnings.append(f"WARNING: Kaggle `url` should usually be: {expected}")

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
    url = data.get("url") or ""

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
    builders_url = f'https://github.com/{cs.GITHUB_OWNER}/{cs.GITHUB_REPO}/blob/main/builders/'
    build_requirements = data.get("build_requirements")

    # ============================================================
    #                   YAML FORMATTING CHECKS
    # ============================================================

    if build_requirements is None:
        return []

    for field in ["source_datasets_ids", "build_script_url", "build_script_commit"]:
        if build_requirements.get(field) is None:
            errors.append(f"ERROR: build_requirements must contain a '{field}' field.")

    if errors:
        return errors

    try:
        is_list = isinstance(build_requirements['source_datasets_ids'], list)
        is_list_str = all(isinstance(id_, str) for id_ in build_requirements['source_datasets_ids'])
        assert is_list and is_list_str, "ERROR: `build_requirements['source_datasets_ids']` should be a list[str] of dataset IDs."
    except AssertionError as e:
        errors.append(str(e))

    try:
        is_str = isinstance(build_requirements['build_script_url'], str)
        is_github_url = build_requirements['build_script_url'].startswith(builders_url)
        assert is_str and is_github_url, f"ERROR: `build_requirements['build_script_url']` should be a string and start with {builders_url}."
    except AssertionError as e:
        errors.append(str(e))

    try:
        is_str = isinstance(build_requirements['build_script_commit'], str)
        is_commit = is_commit_hash(build_requirements['build_script_commit'])
        assert is_str and is_commit, "ERROR: `build_requirements['build_script_commit']` should be a string and a full 40-character commit hash."
    except AssertionError as e:
        errors.append(str(e))

    if errors:
        return errors

    # ============================================================
    #                   GITHUB CHECKS
    # ============================================================

    for src_id in build_requirements["source_datasets_ids"]:
        url = f"{cs.GITHUB_RAW_BASE}/main/catalogue/{src_id}.yaml"
        try:
            if requests.head(url, timeout=10).status_code != 200:
                errors.append(
                    f"ERROR: Source dataset '{src_id}' not found on GitHub main branch. Please create an issue with the source datasets first.")
        except Exception:
            errors.append(f"ERROR: Could not verify source dataset '{src_id}' on GitHub.")

    script_url = build_requirements.get("build_script_url")
    commit = build_requirements.get("build_script_commit")

    try:
        rel_path = script_url.split(builders_url)[-1]
        check_url = f"https://raw.githubusercontent.com/{cs.GITHUB_OWNER}/{cs.GITHUB_REPO}/{commit}/{rel_path}"

        if requests.head(check_url, timeout=10).status_code != 200:
            errors.append(
                f"ERROR: Build script not found at commit {commit[:7]} on GitHub. You must commit the dataset's build script to GitHub before proceeding.")
    except Exception:
        errors.append("ERROR: Could not verify build script on GitHub.")

    return errors


# ====================== PIPELINE ======================
def enrich_and_validate(data: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    """Run automatic enrichment, soft checks, hard checks, and schema validation.

    Returns ``(data, warnings, errors)``. ``data`` is returned regardless of
    whether validation passed; the caller decides what to do based on
    ``errors``.
    """
    # manual enrichment
    data["version"] = clean_version(data.get("version"))
    data["id"] = generate_id(data)
    data['root_dir'] = cs.DATA_DIR / f'datasets_{data["type"]}' / data['id']
    data['data_dir'] = str(data['root_dir'] / "data")
    data['logs_dir'] = str(data['root_dir'] / "logs")
    data['hashes_dir'] = str(data['root_dir'] / "hashes")
    data['root_dir'] = str(data['root_dir'])


    # out in a follow-up PR.
    data.setdefault("build_requirements", {
        "source_datasets_ids": [],
        "build_script_url": None,
        "build_script_commit": None,
    })
    data.setdefault("build_history", [])
    data.setdefault("usage_history", [])

    warnings = check_url_rules(data)
    errors: list[str] = []
    errors.extend(check_version_rules(data))
    # errors.extend(check_build_requirements(data))  # Todo: re-enable once source-dataset workflow is settled.

    if not errors:
        try:
            schema = yaml.safe_load(cs.SCHEMA_PATH.read_text(encoding="utf-8"))
            validate(instance=data, schema=schema)
        except ValidationError as e:
            errors.append(f"ERROR: Schema validation failed: {e.message} (at {list(e.path)})")

    return data, warnings, errors


def _dump_yaml(data: dict[str, Any]) -> str:
    """Serialize a dataset dict to YAML in catalogue style."""
    return yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=120,
    )


def _emit_github_output(**kv: Any) -> None:
    """Append key/value pairs to ``$GITHUB_OUTPUT`` when running in Actions."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for key, value in kv.items():
            fh.write(f"{key}={value}\n")


def _load_input(args: argparse.Namespace) -> dict[str, Any]:
    """Load the input dataset dict from either a YAML path or an issue body."""
    if args.yaml_path:
        return yaml.safe_load(args.yaml_path.read_text(encoding="utf-8"))
    if args.issue_body_file:
        return parse_issue_body(args.issue_body_file.read_text(encoding="utf-8"))
    if args.issue_body is not None:
        return parse_issue_body(args.issue_body)
    raise SystemExit("Must provide --yaml-path, --issue-body, or --issue-body-file.")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--yaml-path", type=Path, help="Validate an existing catalogue YAML file in place.")
    source.add_argument("--issue-body", type=str, help="Parse a GitHub Issue Form body passed as a string.")
    source.add_argument("--issue-body-file", type=Path, help="Parse a GitHub Issue Form body read from a file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cs.CATALOGUE_DIR,
        help="Where to write the resulting YAML (only used with --issue-body[-file]). Defaults to the catalogue.",
    )
    parser.add_argument(
        "--errors-file",
        type=Path,
        help="If set, write the list of validation errors as JSON to this path (even on success, where it'll be []).",
    )
    args = parser.parse_args(argv)

    data = _load_input(args)
    data, warnings, errors = enrich_and_validate(data)

    for w in warnings:
        logger.warning(w)
    for e in errors:
        logger.error(e)

    if args.errors_file:
        args.errors_file.write_text(json.dumps(errors), encoding="utf-8")

    if errors:
        logger.error("❌ Validation failed.")
        _emit_github_output(success="false", dataset_id=data.get("id", ""))
        sys.exit(1)

    output_path = args.output_dir / f"{data['id']}.yaml"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_dump_yaml(data), encoding="utf-8")

    logger.info("✅ YAML schema validation passed.")
    logger.info("📝 Wrote %s", output_path)

    try:
        relpath = str(output_path.resolve().relative_to(cs.PROJECT_ROOT))
    except ValueError:
        relpath = str(output_path)

    _emit_github_output(
        success="true",
        dataset_id=data["id"],
        yaml_path=str(output_path),
        yaml_relpath=relpath,
    )


if __name__ == "__main__":
    main()
