import os
import subprocess
from glob import glob

if __name__ == "__main__":
    if os.path.exists("data"):
        assert os.path.islink("data")
    else:
        os.symlink("/pilot_data_raw", "data")
    pack_cfgs = glob("configs/*/train.py")

    for cfg in pack_cfgs:
        subprocess.check_call(f"python3 pack.py --config {cfg} --visualize", shell=True)