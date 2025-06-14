import re
from typing import Optional


def get_num_blocks_from_filename(filename_base: str) -> Optional[int]:
    """Extracts number of blocks from a filename like 'blocks_3_problem_1'."""
    match = re.search(r"blocks_(\d+)_problem", filename_base)
    if match:
        return int(match.group(1))
    return None
