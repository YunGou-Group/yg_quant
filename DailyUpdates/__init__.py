"""行情入库、因子增量更新、Bin/SQLite 存储。

从仓库根导入：``from DailyUpdates.storage import SQLiteStorage``。
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
