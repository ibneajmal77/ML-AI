from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from p1_customer_health.utils import ensure_dir, write_json


def export_classifier_to_onnx(bundle: dict, sample_df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    try:
        from skl2onnx import to_onnx
        import onnxruntime as ort
    except ImportError:
        write_json(output_dir / "status.json", {"status": "skipped", "reason": "skl2onnx and/or onnxruntime not installed"})
        return

    try:
        model = bundle["model"]
        onnx_model = to_onnx(model, sample_df, target_opset=12)
        onnx_path = output_dir / "classifier.onnx"
        onnx_path.write_bytes(onnx_model.SerializeToString())

        session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        runtime_outputs = session.run(None, {input_name: sample_df.to_dict(orient="list")})
        write_json(output_dir / "status.json", {
            "status": "completed",
            "onnx_path": str(onnx_path),
            "runtime_output_shapes": [list(np.array(output).shape) for output in runtime_outputs],
        })
    except Exception as exc:
        write_json(output_dir / "status.json", {"status": "skipped", "reason": f"onnx export failed: {type(exc).__name__}"})


def export_dense_classifier_to_onnx(model, sample_array: np.ndarray, output_dir: Path) -> None:
    ensure_dir(output_dir)
    try:
        from skl2onnx import to_onnx
        import onnxruntime as ort
    except ImportError:
        write_json(output_dir / "dense_status.json", {"status": "skipped", "reason": "skl2onnx and/or onnxruntime not installed"})
        return

    try:
        sample_array = np.asarray(sample_array, dtype=np.float32)
        onnx_model = to_onnx(model, sample_array[:1], target_opset=12)
        onnx_path = output_dir / "dense_classifier.onnx"
        onnx_path.write_bytes(onnx_model.SerializeToString())

        session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        runtime_outputs = session.run(None, {input_name: sample_array[:5]})
        write_json(output_dir / "dense_status.json", {
            "status": "completed",
            "onnx_path": str(onnx_path),
            "runtime_output_shapes": [list(np.array(output).shape) for output in runtime_outputs],
        })
    except Exception as exc:
        write_json(output_dir / "dense_status.json", {"status": "skipped", "reason": f"dense onnx export failed: {type(exc).__name__}"})
