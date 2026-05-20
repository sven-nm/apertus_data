"""Generate a GitHub Issue Form template from the dataset JSON Schema.

The schema (``dataset_schema.yaml``) is the single source of truth. Properties
that carry ``x-git-issue-*`` extension keywords get rendered as form fields in
``.github/ISSUE_TEMPLATE/new_dataset.yml``. The schema's regular ``description``
is reused as the field description; ``items.enum`` is reused as the option list
for ``checkboxes`` fields.

Extension keywords consumed:
- ``x-git-issue-type``: ``input`` | ``textarea`` | ``checkboxes`` (required to emit).
- ``x-git-issue-placeholder``: placeholder text (ignored for ``checkboxes``).
- ``x-git-issue-required``: bool; when true, ``validations.required: true``.

Run as ``python -m apertus_data.issue_template`` to regenerate the template.
"""

from pathlib import Path
from typing import Any

import yaml

from apertus_data import constants as cs
from apertus_data.utils import get_logger

logger = get_logger(__name__)

ISSUE_TYPE_KEY = 'x-git-issue-type'
PLACEHOLDER_KEY = 'x-git-issue-placeholder'
REQUIRED_KEY = 'x-git-issue-required'
ALLOWED_TYPES = {'input', 'textarea', 'checkboxes', 'dropdown'}


def _checkbox_options(prop_schema: dict[str, Any]) -> list[dict[str, str]]:
    """Return the option list for a ``checkboxes`` field, sourced from ``items.enum``."""
    enum = prop_schema.get('items', {}).get('enum')
    if not enum:
        raise ValueError(
            "Property with x-git-issue-type=checkboxes must declare `items.enum`."
        )
    return [{'label': str(value)} for value in enum]


def _build_field(prop_name: str, prop_schema: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a schema property into a GitHub Issue Form field.

    Returns ``None`` if the property does not carry ``x-git-issue-type``.

    Raises:
        ValueError: If ``x-git-issue-type`` is not one of :data:`ALLOWED_TYPES`,
            or if a ``checkboxes`` field lacks ``items.enum``.
    """
    issue_type = prop_schema.get(ISSUE_TYPE_KEY)
    if issue_type is None:
        return None

    if issue_type not in ALLOWED_TYPES:
        raise ValueError(
            f"Property {prop_name!r}: unsupported {ISSUE_TYPE_KEY}={issue_type!r}; "
            f"allowed: {sorted(ALLOWED_TYPES)}."
        )

    attributes: dict[str, Any] = {'label': prop_name}
    if 'description' in prop_schema:
        attributes['description'] = prop_schema['description']

    if issue_type == 'checkboxes':
        attributes['options'] = _checkbox_options(prop_schema)
    elif PLACEHOLDER_KEY in prop_schema:
        attributes['placeholder'] = prop_schema[PLACEHOLDER_KEY]

    field: dict[str, Any] = {
        'type': issue_type,
        'id': prop_name,
        'attributes': attributes,
    }
    if prop_schema.get(REQUIRED_KEY, False):
        field['validations'] = {'required': True}
    return field


def build_issue_template(schema: dict[str, Any]) -> dict[str, Any]:
    """Build the in-memory GitHub Issue Form representation from ``schema``."""
    properties = schema.get('properties', {})

    body: list[dict[str, Any]] = []
    if 'description' in schema:
        body.append({
            'type': 'markdown',
            'attributes': {'value': schema['description']},
        })

    for prop_name, prop_schema in properties.items():
        field = _build_field(prop_name, prop_schema)
        if field is not None:
            body.append(field)

    return {
        'name': '➕ New Dataset',
        'description': 'Submit a new dataset to the Apertus catalogue',
        'title': '[dataset]: <owner>/<name>',
        'labels': ['dataset'],
        'body': body,
    }


class _BlockStringDumper(yaml.SafeDumper):
    """SafeDumper that renders multi-line strings as ``|`` block scalars."""


def _str_representer(dumper: _BlockStringDumper, data: str):
    if '\n' in data:
        normalized = '\n'.join(line.rstrip() for line in data.splitlines()) + '\n'
        return dumper.represent_scalar('tag:yaml.org,2002:str', normalized, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


_BlockStringDumper.add_representer(str, _str_representer)


def generate(
    schema_path: Path = cs.SCHEMA_PATH,
    output_path: Path = cs.PROJECT_ROOT / '.github' / 'ISSUE_TEMPLATE' / 'new_dataset.yml',
) -> Path:
    """Generate the GitHub Issue Form template and write it to ``output_path``."""
    schema = yaml.safe_load(schema_path.read_text(encoding='utf-8'))
    template = build_issue_template(schema)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(
            template,
            Dumper=_BlockStringDumper,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=120,
        ),
        encoding='utf-8',
    )
    logger.info("✅ Issue template written to %s", output_path)
    return output_path


if __name__ == '__main__':
    generate()
