import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Iterator
import csv
import yaml
import zlib
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from datetime import datetime, timezone


# ========================= LOGGING ============================
ROOT_FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
ROOT_STREAM_HANDLER = logging.StreamHandler()
ROOT_STREAM_HANDLER.setLevel(logging.DEBUG)
ROOT_STREAM_HANDLER.setFormatter(ROOT_FORMATTER)
ROOT_LOGGER = logging.getLogger()
ROOT_LOGGER.setLevel(logging.INFO)
ROOT_LOGGER.addHandler(ROOT_STREAM_HANDLER)

def get_logger(name: str):
    """Custom logging wraper, called each time a logger is declared in the package.

    Note:
        Please make sure to call ``get_logger`` rather than ``logging.getLogger``, so as to centralize the logging configuration and make \
        sure ``ROOT_LOGGER`` is defined. To change the logging level, please import change ``ROOT_LOGGER`` from here directly.

    """
    return logging.getLogger(name)

@contextmanager
def log_to_file(logs_dir: Path, name: str) -> Iterator[Path]:
    """Attach a timestamped FileHandler to ``ROOT_LOGGER`` for the context's lifetime."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log_path = logs_dir / f'{name}_{timestamp}.log'

    handler = logging.FileHandler(log_path, encoding='utf-8')
    handler.setFormatter(ROOT_FORMATTER)
    ROOT_LOGGER.addHandler(handler)
    try:
        yield log_path
    finally:
        ROOT_LOGGER.removeHandler(handler)
        handler.close()
#================================================================


logger = get_logger(__name__)


def safe_move_directory(source: str, dest: str, exclude_suffix: str = '.hash'):
    """This function uses ``mv`` to move a directory from source to destination, skipping files that already exist in the destination."""

    src_dir = Path(source).resolve()
    dst_dir = Path(dest).resolve()

    if not src_dir.exists():
        raise FileNotFoundError(f"Source not found: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Get all files recursively
    files = sorted([f for f in src_dir.rglob('*')
                    if not f.name.endswith(exclude_suffix)])

    print(f"Moving {len(files)} files from {src_dir} to {dst_dir}")

    for src_file in tqdm(files, desc="Moving files"):
        dst_file = dst_dir / src_file.relative_to(src_dir)

        # Skip if already moved
        try:
            if not src_file.is_file():
                continue

            if dst_file.exists():
                continue
        except Exception as e:
            print(f"Error checking if {dst_file} exists: {e}")
            continue

        dst_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            src_file.rename(dst_file)  # atomic move on same filesystem
        except Exception as e:
            print(f"Failed to move {src_file}: {e}")

    print("Move completed successfully!")


def compute_adler32(file_path: Path | str) -> str:
    """Compute fast Adler32 hash of a file."""
    hash_value = 1
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):  # 1MB chunks
            hash_value = zlib.adler32(chunk, hash_value)
    return f"{hash_value:08x}"


def compute_and_write_file_hash(
    file_path: Path,
    output_path: Path,
    hash_function=compute_adler32,
) -> bool:
    """Compute the hash of ``file_path`` and write it to ``output_path``."""
    try:
        hash_hex = hash_function(file_path)
        output_path.write_text(hash_hex)
        return True
    except Exception:
        return False


def compute_and_write_files_hashes(
    root_dir: Path,
    output_dir: Path,
    filename_pattern: str = "*",
    num_cpus: int = 1,
) -> None:
    """Recursively compute Adler32 hash for every file under ``root_dir`` and write
    each hash to ``output_dir``, mirroring the file's relative path under ``root_dir``.
    """
    logger.info(
        f"""⚙️ Computing hashes for files in {root_dir} with pattern {filename_pattern}, writing hash files to {output_dir}""")

    if not root_dir.is_dir():
        raise FileNotFoundError(f"Error: {root_dir} is not a valid directory")

    # Get all files recursively
    files = [f for f in root_dir.rglob(filename_pattern) if f.is_file()]
    logger.info(f"⚙️ Found {len(files)} files in {root_dir}")

    # Pre-create destination subdirectories so workers don't need to mkdir
    for parent in {(output_dir / f.relative_to(root_dir)).parent for f in files}:
        parent.mkdir(parents=True, exist_ok=True)

    # Define output_path for each file
    output_paths = [(output_dir / f.relative_to(root_dir).with_suffix('.hash')) for f in files]

    # Set number of workers
    num_cpus = min(num_cpus, cpu_count())

    # Parallel processing with progress bar
    with Pool(processes=num_cpus) as pool:
        results = list(tqdm(
            pool.starmap(compute_and_write_file_hash, zip(files, output_paths)),
            total=len(files),  # or len(my_output_files)
            desc=f"⚙️ Computing hashes using {num_cpus} CPU(s)",
            unit="file"
        ))

    # Summary
    success = sum(results)
    if success < len(files):
        logger.warning(f"⚠️  Failed to hash {len(files) - success} out of {len(files)} files.")
    else:
        logger.info(f"✅ Hashed {success} out of {len(files)} files successfully.")


def compute_directory_hash(
    root_dir: Path,
    output_path: Path,
    hash_function=zlib.adler32,
    filename_pattern: str = '*.hash'
) -> str:
    """Compute the hash of all hash files in a directory recursively and write it to a file.

    Note:
        This function sorts the hash files before computing the hash for maximum reproducibility.

    Args:
        root_dir (Path): The root directory to compute the hash for.
        output_path (Path): The file to write the computed hash to.
        hash_function (callable, optional): The hash function to use. Should take a binary string as input. Defaults to zlib.adler32.
        filename_pattern (str, optional): The pattern to match hash files. Defaults to '*.hash'.
    """
    logger.info(f"⚙️ Computing directory hash for {root_dir} using files matching {filename_pattern}")

    # Get existing hashes
    hashes = [f.read_text(encoding='utf-8').strip() for f in root_dir.rglob(filename_pattern)]

    # Sort the hashes
    hashes.sort()

    # Combine to binaries and compute hash value
    combined_data = ''.join(hashes).encode('utf-8')
    hash_value = hash_function(combined_data)
    hash_value = f"{hash_value:08x}"
    logger.info(f'✅ Computed directory hash for {root_dir}: {hash_value}')

    # Write to output_path
    output_path.write_text(hash_value)
    logger.info(f'⬇️ Hash written to {output_path}')

    return hash_value


def compare_files_hashes(dir1: str, dir2: str):
    """Compare all the hashes of two directories."""
    dir1 = Path(dir1).resolve()
    dir2 = Path(dir2).resolve()

    if not dir1.is_dir() or not dir2.is_dir():
        print("One or both directories do not exist")
        return

    # Get all .hash files in both directories
    hash_files1 = {f.relative_to(dir1): f for f in dir1.rglob("*.hash")}
    hash_files2 = {f.relative_to(dir2): f for f in dir2.rglob("*.hash")}

    all_rel_paths = sorted(set(hash_files1.keys()) | set(hash_files2.keys()))

    print(f"Comparing .hash files between:\n  {dir1}\n  {dir2}\n")
    print(f"Found {len(all_rel_paths)} unique .hash files.\n")

    differences = 0

    for rel_path in tqdm(all_rel_paths):
        file1 = hash_files1.get(rel_path)
        file2 = hash_files2.get(rel_path)

        if not file1 or not file1.exists():
            print(f"❌ Missing in dir1: {rel_path}")
            differences += 1
            continue
        if not file2 or not file2.exists():
            print(f"❌ Missing in dir2: {rel_path}")
            differences += 1
            continue

        hash1 = file1.read_text().strip()
        hash2 = file2.read_text().strip()

        if hash1 == hash2:
            # print(f"✓ Match: {rel_path}")
            pass
        else:
            print(f"✗ DIFFERENT: {rel_path}")
            differences += 1

    print(f"\nComparison finished. Found {differences} difference(s).")


def print_files_by_date(directory: str, target_date: str):
    """
    Print all files created on the given date.

    Args:
        directory (str): Root directory to search
        target_date (str): Date in 'YYYY-MM-DD' format (e.g. '2025-04-18')
    """
    target = datetime.strptime(target_date, '%Y-%m-%d').date()

    print(f"Files created on {target_date}:\n")

    for file_path in Path(directory).rglob('*'):
        # print(file_path)
        if file_path.is_file():
            # Use creation time if available, otherwise fall back to modification time
            stat = file_path.stat()
            try:
                # st_birthtime = true creation time (macOS, some Linux)
                creation_time = stat.st_birthtime
            except AttributeError:
                # st_mtime = modification time (most reliable fallback on Linux)
                creation_time = stat.st_mtime

            file_date = datetime.fromtimestamp(creation_time).date()

            if file_date == target:
                if input(f"Found log: {file_path}\nPrint?") == "":
                    print(file_path.read_text(encoding="utf-8"))


def sanity_check_tokenized_dataset(tokenized_dir: str):
    tokenized_dir = Path(tokenized_dir).resolve()

    print(f"Sanity checking tokenized dataset at {tokenized_dir}")
    print("==============================================================")
    print(f"Total files: {len(list(tokenized_dir.rglob('*.bin')))}")
    print("==============================================================")

    print("Checking .bin files without corresponding .idx:")
    print("==============================================================")
    for bin_path in tokenized_dir.rglob("*.bin"):
        if not bin_path.with_suffix('.idx').exists():
            print(bin_path)

    print("Checking .idx files without corresponding .bin:")
    print("==============================================================")
    for bin_path in tokenized_dir.rglob("*.idx"):
        if not bin_path.with_suffix('.bin').exists():
            print(bin_path)

    print("Checking .bin files with size 0:")
    print("==============================================================")
    for bin_path in tokenized_dir.rglob("*.bin"):
        if bin_path.stat().st_size == 0:
            print(bin_path)

    print("Checking .idx files with size 0:")
    print("==============================================================")
    for idx_path in tokenized_dir.rglob("*.idx"):
        if idx_path.stat().st_size == 0:
            print(idx_path)


def tsv_to_yaml_files(tsv_path: str | Path, output_dir: str | Path) -> None:
    """
    Reads a TSV file and creates one YAML file per row in the output directory.
    Each YAML file is named after the value in the 'id' column.

    Args:
        tsv_path: Path to the input TSV file (must contain an 'id' column)
        output_dir: Directory where individual YAML files will be saved
    """
    tsv_path = Path(tsv_path)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    with tsv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            row_dict: Dict[str, Any] = dict(row)  # convert to plain dict

            id_value = str(row_dict['Dataset ID']).strip()
            if not id_value:
                print(f"⚠️  Skipping row with empty 'id': {row_dict}")
                continue

            # Create safe filename
            safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in id_value)
            yaml_path = output_dir / f"{safe_id}.yaml"

            # Write single dict as YAML
            with yaml_path.open('w', encoding='utf-8') as f_out:
                yaml.dump(
                    row_dict,
                    f_out,
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                    indent=2
                )

            print(f"✅ Created: {yaml_path.name}")

    print(f"\n Finished! YAML files written to: {output_dir}")

