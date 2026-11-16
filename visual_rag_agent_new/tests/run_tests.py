#!/usr/bin/env python3
"""Run every tests/test_*.py without pytest: each test_* function is one check.
    python tests/run_tests.py            # all suites
    python tests/run_tests.py protocol   # suites whose name contains 'protocol'
"""
import importlib.util
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl"))


def main() -> int:
    needles = sys.argv[1:]
    passed = failed = 0
    for path in sorted(HERE.glob("test_*.py")):
        if needles and not any(n in path.stem for n in needles):
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"== {path.stem} ==")
        for name in dir(module):
            if not name.startswith("test_"):
                continue
            fn = getattr(module, name)
            if not callable(fn):
                continue
            try:
                fn()
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
            else:
                passed += 1
                print(f"  ok  {name}")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
