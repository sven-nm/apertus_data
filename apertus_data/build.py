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
from pathlib import Path
from typing import Callable

import requests

from apertus_data import constants as cs
from apertus_data.utils import get_logger, log_to_file

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

    if len(params) < 2 or params[0].name != 'output_dir' or params[1].name != 'logs_dir':
        raise TypeError(
            f"Builder {func.__qualname__!r} must start with positional parameters "
            f"(output_dir, logs_dir, ...); got {[p.name for p in params]!r}."
        )

    for p in params[2:]:
        if p.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(
                f"Builder {func.__qualname__!r}: parameter {p.name!r} after "
                f"(output_dir, logs_dir) must be keyword-only "
                f"(declare them after `*` in the signature)."
            )

    if signature.return_annotation not in (None, type(None), inspect.Signature.empty):
        raise TypeError(
            f"Builder {func.__qualname__!r} must return None "
            f"(got annotation {signature.return_annotation!r})."
        )


def parse_build_script_url(url: str) -> str:
    """Extract the repo-relative path from a ``build_script_url``.

    Example::

        >>> parse_build_script_url(
        ...     'https://github.com/sven-nm/apertus_data/builders/foo.py'
        ... )
        'builders/foo.py'
    """
    expected_prefix = f'https://github.com/{cs.GITHUB_OWNER}/{cs.GITHUB_REPO}/'
    return url.split(expected_prefix)[-1]


def verify_remote_script(url: str, commit: str, timeout: int = 10) -> None:
    """HEAD-check that ``url`` resolves to a file at ``commit`` on GitHub raw."""
    relative_path = parse_build_script_url(url)
    raw_url = f'{cs.GITHUB_RAW_BASE}/{commit}/{relative_path}'
    logger.info("🔎 Verifying remote script at %s", raw_url)

    response = requests.head(raw_url, timeout=timeout, allow_redirects=True)
    if response.status_code != 200:
        raise RuntimeError(
            f"Build script not found on GitHub at commit {commit[:7]}: "
            f"HTTP {response.status_code} on {raw_url}."
        )


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run a git command at ``cs.PROJECT_ROOT`` and return the completed process."""
    return subprocess.run(
        ['git', '-C', str(cs.PROJECT_ROOT), *args],
        capture_output=True, text=True, check=False,
    )


def verify_local_script_at_commit(url: str, commit: str) -> Path:
    """Ensure the local copy of the script is at the recorded commit.

    Strategy:
    - Derive the local path from the URL.
    - Resolve the file's most recent commit. If it matches ``commit``, return.
    - Otherwise look ``commit`` up in the file's history; if absent, raise.
    - If present, ``git checkout <commit> -- <path>`` to update the working
      tree to exactly that revision, then return the path.

    Returns:
        Path to the (now commit-aligned) local script file.
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

    current = _git('log', '-1', '--pretty=format:%H', '--', relative_path).stdout.strip()
    if current == commit:
        logger.info("✅ Local %s already at recorded commit %s", relative_path, commit[:7])
        return local_path

    logger.warning(
        "Local %s is at %s, not the recorded %s; searching file history…",
        relative_path, current[:7] or '<none>', commit[:7],
    )
    history = _git('log', '--pretty=format:%H', '--', relative_path).stdout.split()
    if commit not in history:
        raise RuntimeError(
            f"Recorded commit {commit[:7]} not found in history of {relative_path}. "
            f"History tip: {current[:7] or '<none>'}."
        )

    logger.info("⏪ Checking out %s @ %s", relative_path, commit[:7])
    checkout = _git('checkout', commit, '--', relative_path)
    if checkout.returncode != 0:
        raise RuntimeError(
            f"git checkout {commit[:7]} -- {relative_path} failed: {checkout.stderr.strip()}"
        )
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


def prepare_builder(url: str, commit: str) -> BuilderFn:
    """Verify, load, and validate the builder ``main`` for ``url`` at ``commit``.

    Composition of :func:`verify_remote_script`,
    :func:`verify_local_script_at_commit`, :func:`load_main`, and
    :func:`validate_builder_signature`. Returns the validated ``main`` callable.
    """
    verify_remote_script(url, commit)
    script_path = verify_local_script_at_commit(url, commit)
    main_fn = load_main(script_path)
    validate_builder_signature(main_fn)
    return main_fn


def run_builder(
    func: BuilderFn,
    output_dir: Path,
    logs_dir: Path,
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

    with log_to_file(logs_dir, name=name) as log_path:
        logger.info("▶️ Running builder %r → %s", name, output_dir)
        func(output_dir=output_dir, logs_dir=logs_dir, **kwargs)
        logger.info("◀️ Builder %r finished", name)

    if not any(output_dir.iterdir()):
        raise RuntimeError(f"Builder {name!r} produced no files in {output_dir}.")
