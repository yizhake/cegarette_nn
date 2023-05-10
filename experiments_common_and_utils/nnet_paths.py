import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

if MARABOU_DIR := os.getenv("MARABOU_DIR", None):
    MARABOU_DIR = Path(MARABOU_DIR)
else:
    import inspect, maraboupy
    maraboupy_dir = Path(inspect.getabsfile(maraboupy)).resolve()
    MARABOU_DIR = maraboupy_dir.parent.parent
    if not MARABOU_DIR.exists():
        raise Exception(f"MARABOU_DIR not set and could not find maraboupy dir: {maraboupy_dir}")

NNET_ROOT_PATH = MARABOU_DIR / "resources/nnet/acasxu"

all_nnet_paths = [
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_1_1.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_1_2.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_1_3.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_1_4.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_1_5.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_1_6.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_1_7.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_1_8.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_1_9.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_2_1.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_2_2.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_2_3.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_2_4.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_2_5.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_2_6.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_2_7.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_2_8.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_2_9.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_3_1.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_3_2.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_3_3.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_3_4.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_3_5.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_3_6.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_3_7.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_3_8.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_3_9.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_4_1.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_4_2.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_4_3.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_4_4.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_4_5.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_4_6.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_4_7.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_4_8.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_4_9.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_5_1.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_5_2.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_5_3.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_5_4.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_5_5.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_5_6.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_5_7.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_5_8.nnet",
    NNET_ROOT_PATH / "ACASXU_experimental_v2a_5_9.nnet",
]
