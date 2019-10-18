import json
import fire
import re
from pathlib import Path

def is_export(cell):
    if cell["cell_type"] != "code":
        return False

    src = cell["source"]
    if len(src) == 0 or len(src[0]) < 7:
        return False

    ### import pdb
    ### pdb.set_trace()

    return re.match(r"^\s*#\s*export\s*$", src[0], re.IGNORECASE) is not None


def get_sorted_files(all_files, up_to=True):

