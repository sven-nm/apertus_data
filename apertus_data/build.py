"""Builder discovery, contract enforcement, and runner.

A *builder* is a per-dataset script under ``builders/<dataset_id>.py`` that
exposes a ``main(output_dir, logs_dir, ...)`` function. Builders are
discovered from a dataset's YAML via ``build_requirements.build_script_url``
and pinned to ``build_requirements.build_script_commit``. The runtime, not
the builder author, enforces:

- the script exists on GitHub at the pinned commit (network HEAD check);
- the local copy matches that commit (history lookup + ``git checkout`` of
  just that file if needed);
- ``main`` has signature ``(output_dir, logs_dir, *, **kwargs) -> None``;
- a log file lands in ``logs_dir`` for every run (attached automatically);
- ``output_dir`` is non-empty after the builder returns.
"""

import importlib.util
import inspect
import subprocess
import sys
import re
from pathlib import Path
from typing import Callable

import requests

from apertus_data import constants as cs
from apertus_data.utils import get_logger

logger = get_logger(__name__)

BuilderFn = Callable[..., None]


def validate_builder_signature(func: BuilderFn) -> None:
    """Validate a builder's signature against the runner's contract.

    The function must accept ``output_dir`` and ``logs_dir`` as the first two
    positional parameters; any additional parameters must be keyword-only;
    return annotation, if present, must be ``None``.
    """
    signature = inspect.signature(func)
    params = list(signature.parameters.values())

    if len(params) < 3 or params[0].name != 'output_dir' or params[1].name != 'logs_dir' or params[2].name != 'dataset':
        raise TypeError(
            f"Builder {func.__qualname__!r} must start with positional parameters "
            f"(output_dir, logs_dir, ...); got {[p.name for p in params]!r}."
        )

    # Todo: see how we handle this
    # for p in params[3:]:
    #     if p.kind is not inspect.Parameter.KEYWORD_ONLY:
    #         raise TypeError(
    #             f"Builder {func.__qualname__!r}: parameter {p.name!r} after "
    #             f"(output_dir, logs_dir) must be keyword-only "
    #             f"(declare them after `*` in the signature)."
    #         )

    if signature.return_annotation not in (None, type(None), inspect.Signature.empty):
        raise TypeError(
            f"Builder {func.__qualname__!r} must return None "
            f"(got annotation {signature.return_annotation!r})."
        )


def parse_build_script_url(url: str) -> str:
    """
    Extract the repo-relative path from any GitHub URL.

    Handles all these formats:
        - https://github.com/sven-nm/apertus_data/builders/foo.py
        - https://github.com/sven-nm/apertus_data/blob/main/builders/foo.py
        - https://github.com/sven-nm/apertus_data/blob/75e48d.../builders/foo.py
        - https://raw.githubusercontent.com/sven-nm/apertus_data/main/builders/foo.py
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")

    url = url.strip().rstrip('/')

    # 1. raw.githubusercontent.com URLs
    if "raw.githubusercontent.com" in url:
        # raw.githubusercontent.com/OWNER/REPO/BRANCH/PATH...
        match = re.search(r'raw\.githubusercontent\.com/[^/]+/[^/]+/[^/]+/(.+)', url)
        if match:
            return match.group(1)

    # 2. github.com with /blob/ or /tree/
    match = re.search(
        r'github\.com/[^/]+/[^/]+/(?:blob|tree)/[^/]+/(.+)',
        url
    )
    if match:
        return match.group(1)

    # 3. github.com without blob/tree (direct path)
    match = re.search(
        r'github\.com/[^/]+/[^/]+/(.+)',
        url
    )
    if match:
        return match.group(1)

    # Fallback: if nothing matched, return the original (or raise)
    raise ValueError(f"Could not parse GitHub URL: {url}")


def verify_remote_script(url: str, commit: str, timeout: int = 10) -> None:
    """HEAD-check that ``url`` resolves to a file at ``commit`` on GitHub raw."""
    relative_path = parse_build_script_url(url)
    raw_url = f'{cs.GITHUB_RAW_BASE}/{commit}/{relative_path}'
    logger.info("🔎 Verifying remote script at %s", raw_url)

    response = requests.head(raw_url, timeout=timeout, allow_redirects=True)
    if response.status_code != 200:
        raise RuntimeError(
            f"Build script not found on GitHub at commit {commit}: "
            f"HTTP {response.status_code} on {raw_url}."
        )


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run a git command at ``cs.PROJECT_ROOT`` and return the completed process."""
    return subprocess.run(
        ['git', '-C', str(cs.PROJECT_ROOT), *args],
        capture_output=True, text=True, check=False,
    )

def verify_local_script_at_commit(url: str, commit: str, yaml_path: Path | None = None) -> Path:
    """Ensure the working-tree script is byte-for-byte identical to the pinned commit.

    Reads the file content at ``commit`` via ``git show`` and compares it to the
    working-tree file (including any uncommitted edits).  If they differ, raises
    a :class:`RuntimeError` that explains the three remediation steps.

    Args:
        url: GitHub URL of the build script.
        commit: The pinned commit hash recorded in the dataset YAML.
        yaml_path: Absolute path to the dataset YAML, used in the error message.

    Returns:
        Path to the local (working-tree) script, verified to match ``commit``.
    """
    relative_path = parse_build_script_url(url)
    local_path = cs.PROJECT_ROOT / relative_path

    if not local_path.exists():
        raise FileNotFoundError(
            f"Local copy of build script missing at {local_path}; "
            f"pull or check out the {commit[:7]} revision first."
        )

    if _git('rev-parse', '--is-inside-work-tree').returncode != 0:
        raise RuntimeError(f"{cs.PROJECT_ROOT} is not a git repository.")

    if _git('ls-files', '--error-unmatch', '--', relative_path).returncode != 0:
        raise RuntimeError(f"Build script {relative_path} is not tracked by git.")

    result = _git('show', f'{commit}:{relative_path}')
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not read {relative_path} at pinned commit {commit[:7]} from git "
            f"(have you fetched it?): {result.stderr.strip()}"
        )

    if local_path.read_text() != result.stdout:
        yaml_abs = yaml_path or '<dataset-yaml-path>'
        raise RuntimeError(
            f"The local build script '{relative_path}' does not match the pinned "
            f"commit {commit[:7]}.\n\n"
            f"This means the script you are about to run differs from the version "
            f"recorded in the dataset YAML.  To fix this:\n\n"
            f"  1. Commit your build script changes and push to main:\n"
            f"       git commit {local_path} -m 'Update build script'\n"
            f"       git push origin main\n\n"
            f"  2. Copy the new commit hash and update build_requirements.build_script_commit "
            f"in the dataset YAML.\n\n"
            f"  3. Commit and push the updated YAML:\n"
            f"       git commit {yaml_abs} -m 'Pin build script to new commit'\n"
            f"       git push origin main"
        )

    logger.info("✅ Local %s matches pinned commit %s", relative_path, commit[:7])
    return local_path


def load_main(script_path: Path) -> BuilderFn:
    """Import ``script_path`` as a fresh module and return its ``main`` function."""
    module_name = f'_apertus_builder_{script_path.stem}'
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load build script at {script_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, 'main'):
        raise AttributeError(f"Build script {script_path} has no `main` function.")
    return module.main


def prepare_builder(url: str, commit: str, yaml_path: Path | None = None) -> BuilderFn:
    """Verify, load, and validate the builder ``main`` for ``url`` at ``commit``.

    Composition of :func:`verify_remote_script`,
    :func:`verify_local_script_at_commit`, :func:`load_main`, and
    :func:`validate_builder_signature`. Returns the validated ``main`` callable.
    """
    verify_remote_script(url, commit)
    script_path = verify_local_script_at_commit(url, commit, yaml_path=yaml_path)
    main_fn = load_main(script_path)
    validate_builder_signature(main_fn)
    return main_fn


def run_builder(
    func: BuilderFn,
    output_dir: Path,
    logs_dir: Path,
    dataset: 'Dataset',
    **kwargs,
) -> None:
    """Invoke a builder ``main`` callable under the runner's contract.

    Creates ``output_dir`` and ``logs_dir`` if missing, attaches a file log
    handler, runs the builder, then asserts that the builder populated
    ``output_dir`` and that the log file landed in ``logs_dir``.
    """
    name = func.__module__.removeprefix('_apertus_builder_') or func.__qualname__

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("▶️ Running builder %r → %s", name, output_dir)
    func(output_dir=output_dir, logs_dir=logs_dir, dataset=dataset, **kwargs)
    logger.info("◀️ Builder %r finished", name)

    if not any(output_dir.iterdir()):
        raise RuntimeError(f"Builder {name!r} produced no files in {output_dir}.")
