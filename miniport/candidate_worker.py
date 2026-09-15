"""Trusted transport entry point, mounted read-only in the torch-only container.

No reference implementation, hidden seeds, or expected outputs enter this process.
Its results are untrusted and independently graded outside the container.
"""
import contextlib
import importlib.util
import json
import sys
import traceback

import numpy as np
import torch


class OutputTypeError(TypeError):
    """A returned value violates the tensor interface, not a candidate API call."""
    def __init__(self, field, value):
        self.field = field
        self.actual_type = type(value).__name__
        super().__init__(f"{field} must return torch.Tensor, got {self.actual_type}")


def exception_details(exc, stage):
    """Preserve runtime errors verbatim; output validation has its own category."""
    frames = [frame for frame in traceback.extract_tb(exc.__traceback__) if frame.filename == sys.argv[1]]
    detail = {"code": "runtime_error", "stage": stage, "message": str(exc),
              "exception_type": type(exc).__name__, "line": frames[-1].lineno if frames else None}
    if isinstance(exc, OutputTypeError):
        detail.update(code="output_type_mismatch", field=exc.field,
                      required_type="Tensor", actual_type=exc.actual_type)
    elif isinstance(exc, AttributeError):
        detail.update(code="missing_attribute", attribute=getattr(exc, "name", None),
                      object_type=type(getattr(exc, "obj", None)).__name__)
    elif isinstance(exc, KeyError):
        detail.update(code="missing_key", key=exc.args[0] if exc.args else None)
    elif isinstance(exc, TypeError):
        detail["code"] = "type_error"
    elif isinstance(exc, NotImplementedError):
        detail["code"] = "not_implemented"
    return detail


def tensor_payload(value, field):
    if not isinstance(value, torch.Tensor):
        raise OutputTypeError(field, value)
    value = value.detach().cpu()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "values": value.tolist()}


def main():
    request = json.load(sys.stdin)
    replies = []
    torch.set_num_threads(1)
    # Candidate prints go to stderr; diagnostics from this process are never
    # trusted as grading evidence, only returned tensors are compared externally.
    with contextlib.redirect_stdout(sys.stderr):
        spec = importlib.util.spec_from_file_location("candidate", sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            replies = [{"topology": False, "error_type": type(exc).__name__,
                        "diagnostic": exception_details(exc, "import")} for _ in request["cases"]]
            print(json.dumps({"cases": replies}), file=sys.__stdout__)
            return
        for case in request["cases"]:
            reply = {"parameters": None, "layer": None, "output": None,
                     "repeat": None, "cached": None, "error_type": None}
            try:
                stage = "build"
                model = module.build(case["config"])
                stage = "topology"
                if model.topology() != case["config"]["template"]:
                    raise ValueError("Topology mismatch")
                for name in ("load_parameters", "export_parameters", "run"):
                    if not callable(getattr(model, name, None)):
                        raise TypeError("Missing method")
                if case["config"]["template"] == "rope_cache" and not callable(getattr(model, "run_cached", None)):
                    raise TypeError("Missing cached runner")
                reply["topology"] = True
                stage = "load_parameters"
                model.load_parameters({k: np.asarray(v, dtype=np.float32) for k, v in case["parameters"].items()})
                stage = "export_parameters"
                reply["parameters"] = {k: tensor_payload(v, f"export_parameters.{k}") for k, v in model.export_parameters().items()}
                x = torch.tensor(case["input"], dtype=torch.float32)
                draws = torch.tensor(case["draws"], dtype=torch.float32)
                with torch.no_grad():
                    stage = "run"
                    result = model.run(x.clone(), draws.clone())
                    stage = "layer"
                    reply["layer"] = tensor_payload(result["layer"], "run.layer")
                    stage = "output"
                    reply["output"] = tensor_payload(result["output"], "run.output")
                    stage = "repeat"
                    reply["repeat"] = tensor_payload(model.run(x.clone(), draws.clone())["output"], "run.output")
                    if case["config"]["template"] == "rope_cache":
                        stage = "run_cached"
                        reply["cached"] = tensor_payload(model.run_cached(x.clone(), draws.clone()), "run_cached")
            except Exception as exc:
                reply["error_type"] = type(exc).__name__
                reply["diagnostic"] = exception_details(exc, stage)
            replies.append(reply)
    print(json.dumps({"cases": replies}, allow_nan=False))


if __name__ == "__main__":
    main()
