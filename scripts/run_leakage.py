# run_leakage.py

Run Study 1 (target leakage) and print the result as JSON.

---

## Source Code

```python
#!/usr/bin/env python3

"""Run Study 1 (target leakage) and print the result as JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgme import leakage   # noqa: E402


def main() -> int:
    print(json.dumps(leakage.run().as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## Description

This script:

1. Locates the project's `src` directory.
2. Adds it to the Python import path if necessary.
3. Imports the `leakage` module from the `dgme` package.
4. Executes `leakage.run()`.
5. Converts the result to a dictionary via `as_dict()`.
6. Prints the result as formatted JSON.
7. Exits with status code `0`.

### Expected Usage

```bash
python run_leakage.py
```

### Output

The script outputs a JSON object containing the results of the target leakage study.