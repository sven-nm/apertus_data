from pathlib import Path
from huggingface_hub import snapshot_download
import os

from apertus_data.utils import get_logger

logger = get_logger(__name__)

def hf_transfer_download(
    repo_id: str,
    output_dir: str | Path,
    revision: str,
    hf_token: str,
    max_workers: int = 16
) -> Path:
    """Ultra-fast dataset download using hf_transfer (Rust backend)."""
    # Enable hf_transfer (the fastest downloader)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    output_dir = Path(output_dir) if output_dir else Path(repo_id.replace("/", "_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"🚀 Starting ultra-fast download of: {repo_id}")
    logger.info(f"📁 Saving to: {output_dir}")
    logger.info(f"⚡ hf_transfer enabled (Rust downloader)")

    local_dir = snapshot_download(repo_id=repo_id,
                                  repo_type="dataset",
                                  revision=revision,
                                  local_dir=output_dir,
                                  token=hf_token,
                                  max_workers=max_workers,  # Adjust based on your connection
                                  allow_patterns=["*"],  # Download everything
                                  )

    logger.info(f"✅ Download completed successfully!")
    logger.info(f"📦 Files saved in: {local_dir}")
    return Path(local_dir)