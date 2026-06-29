"""Per-SU progress reporting for the dashboard volume scripts.

Each snip script calls ``report(processed, total, label)`` as it works. The
dashboard's volume_runner reads the resulting ``progress.json`` (written
atomically into the cwd, which is the script's working dir) on each status poll
to drive a determinate progress bar. Failures here are swallowed — progress
reporting must never break the actual pipeline.
"""

import json
import os


def report(processed: int, total: int, label: str = "") -> None:
    try:
        payload = json.dumps({"processed": processed, "total": total, "label": label})
        cwd = os.getcwd()
        tmp = os.path.join(cwd, "progress.json.tmp")
        with open(tmp, "w") as f:
            f.write(payload)
        os.replace(tmp, os.path.join(cwd, "progress.json"))  # atomic on same fs
    except OSError:
        pass
