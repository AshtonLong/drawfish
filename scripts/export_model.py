"""Copy a trained checkpoint into models/drawfish.pt (the file the GUI and UCI engine load).

  python scripts/export_model.py --run main            # latest EMA weights of a run
  python scripts/export_model.py --file path/to/net.pt
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drawfish.model import load_model  # noqa: E402
from drawfish.paths import CKPT_DIR, DEFAULT_MODEL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--file", default=None)
    args = ap.parse_args()
    src = Path(args.file) if args.file else CKPT_DIR / args.run / "net_latest.pt"
    load_model(src)  # validate it loads
    DEFAULT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, DEFAULT_MODEL)
    print(f"exported {src} -> {DEFAULT_MODEL}")


if __name__ == "__main__":
    main()
