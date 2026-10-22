# run_uplift.py

Run Study 2 (propensity vs uplift) and print the result as JSON.

---

## Source Code

```python
#!/usr/bin/env python3
"""Run Study 2 (propensity vs uplift) and print the result as JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgme import uplift   # noqa: E402


def main() -> int:
    print(json.dumps(uplift.run().as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## Description

This script executes **Study 2: Propensity vs Uplift** and outputs the results in JSON format.

### Processing Flow

1. Determines the project's `src` directory.
2. Adds the directory to Python's import path (`sys.path`) if needed.
3. Imports the `uplift` module from the `dgme` package.
4. Runs the study using:

```python
uplift.run()
```

5. Converts the returned result to a dictionary:

```python
.as_dict()
```

6. Serializes the result as formatted JSON:

```python
json.dumps(..., indent=2)
```

7. Prints the JSON output to the console.
8. Exits with status code `0`.

### Example Usage

```bash
python run_uplift.py
```

### Expected Output

A JSON document containing the metrics and findings from the uplift-modeling study.