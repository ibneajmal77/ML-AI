from pathlib import Path
import sys

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    uvicorn.run(
        "world_cricket_ml.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        app_dir=str(PROJECT_ROOT / "src"),
    )


if __name__ == "__main__":
    main()
