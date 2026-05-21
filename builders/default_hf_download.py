from apertus_data import private
from pathlib import Path
from huggingface_hub import snapshot_download
import os

from apertus_data.utils import get_logger, log_to_file

logger = get_logger(__name__)


def main(
    output_dir: Path,
    logs_dir: Path,
    dataset: 'Dataset',
    hf_token: str = private.HF_TOKEN,
    max_workers: int = 16
) -> None:
    """Download the ``default_hf_download`` snapshot into ``output_dir``."""

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    repo_id = f'{dataset.owner}/{dataset.name}'

    with log_to_file(logs_dir, name=__file__.name) as log_path:
        logger.info(f"🚀 Starting ultra-fast download of: {repo_id}, version: {dataset.version[:8]}")
        logger.info(f"📁 Saving to: {output_dir}")
        logger.info(f"⚡ hf_transfer enabled (Rust downloader)")

        snapshot_download(repo_id=repo_id,
                          repo_type="dataset",
                          revision=dataset.version,
                          local_dir=output_dir,
                          token=hf_token,
                          max_workers=max_workers,  # Adjust based on your connection
                          allow_patterns=["*"],  # Download everything
                          )

        logger.info(f"✅ Download completed successfully!")
        logger.info(f"📂 Files saved in: {output_dir}")
        logger.info(f'📃 Log saved in: {log_path}')
