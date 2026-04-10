from __future__ import annotations

import logging
from pathlib import Path

from world_cricket_ml.utils import write_json

log = logging.getLogger(__name__)


def export_onnx_stub(artifact_root: Path) -> None:
    """Attempt to export the best classifier pipeline to ONNX format.

    Requires the ``onnx`` optional dependency group:
        pip install world-cricket-ml-foundations[onnx]

    The export uses per-column type declarations (FloatTensorType for numeric
    features, StringTensorType for categorical features) so that skl2onnx can
    correctly trace the ColumnTransformer → OneHotEncoder path.
    """
    try:
        import joblib
        import pandas as pd
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType
    except ImportError:
        log.info("ONNX export skipped — install skl2onnx and onnxruntime to enable.")
        write_json(
            artifact_root / "onnx" / "status.json",
            {"status": "skipped", "reason": "Install skl2onnx and onnxruntime (pip install world-cricket-ml-foundations[onnx])."},
        )
        return

    try:
        from world_cricket_ml.training.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

        model = joblib.load(artifact_root / "classification" / "model.joblib")

        # Build per-column initial_type matching exactly what the Pipeline expects:
        # numeric features → FloatTensorType, categorical features → StringTensorType.
        initial_type = (
            [(col, FloatTensorType([None, 1])) for col in NUMERIC_FEATURES]
            + [(col, StringTensorType([None, 1])) for col in CATEGORICAL_FEATURES]
        )

        onnx_model = convert_sklearn(model, initial_types=initial_type)
        output_path = artifact_root / "onnx" / "dominance_model.onnx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(onnx_model.SerializeToString())
        log.info("ONNX export complete: %s", output_path)
        write_json(
            artifact_root / "onnx" / "status.json",
            {"status": "completed", "path": str(output_path)},
        )
    except Exception as exc:
        log.warning("ONNX export failed: %s", exc, exc_info=True)
        write_json(
            artifact_root / "onnx" / "status.json",
            {"status": "failed", "reason": str(exc)},
        )
