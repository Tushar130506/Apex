import sys, pathlib
root = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(root / "src")); sys.path.insert(0, str(root / "tests"))
