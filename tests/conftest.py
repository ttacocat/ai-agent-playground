"""共享 fixtures。"""

import sys
from pathlib import Path

# 让 pytest 无论从哪个目录启动都能 import 到 src layout 的包
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
