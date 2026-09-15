"""Ordered parity verifier. Run candidate code only in a trusted local workspace."""
import importlib.util
import math
from pathlib import Path

from .tasks import GATES, VERIFIER_VERSION


def verify(task, candidate, seeds):
    import numpy as np
    import torch
    from .reference import fixture, mapped_parameters, reference

    report = {"version": VERIFIER_VERSION, "task_id": task.id, "gates": [], "passed": False}
    seeds = list(seeds)
    if not seeds:
        raise ValueError("At least one verifier seed is required")
    stage = "topology"
    metric = None

    def array(value):
        if not isinstance(value, torch.Tensor):
            raise TypeError("Candidate outputs and exported parameters must be torch tensors")
        return value.detach().cpu().numpy()

    def compare(actual, expected):
        nonlocal metric
        actual = array(actual)
        if actual.shape != expected.shape:
            raise AssertionError(f"shape {actual.shape} != {expected.shape}")
        if not np.isfinite(actual).all():
            raise AssertionError("nonfinite output")
        error = float(np.max(np.abs(actual.astype(float) - expected.astype(float))))
        metric = max(metric or 0.0, error)
        if np.issubdtype(expected.dtype, np.integer):
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)

    try:
        path = Path(candidate).resolve()
        spec = importlib.util.spec_from_file_location("miniport_candidate", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.build(task.to_dict())
        assert model.topology() == task.template, "Incorrect topology identifier"
        for method in ("load_parameters", "export_parameters", "run"):
            assert callable(getattr(model, method, None)), f"Missing {method}"
        if task.template == "rope_cache":
            assert callable(getattr(model, "run_cached", None)), "Missing run_cached"
        report["gates"].append({"name": stage, "passed": True})
        for stage in GATES[1:]:
            metric = None
            for seed in seeds:
                p, x, draws = fixture(task, seed)
                model.load_parameters({k: v.copy() for k, v in p.items()})
                with torch.no_grad():
                    if stage == "parameters":
                        actual = model.export_parameters()
                        expected = mapped_parameters(task, p)
                        assert set(actual) == set(expected), "Parameter keys differ"
                        for key in expected:
                            compare(actual[key], expected[key])
                    else:
                        expected = reference(task, p, x, draws)
                        tx, td = torch.from_numpy(x.copy()), torch.from_numpy(draws.copy())
                        result = model.run(tx, td)
                        key = "layer" if stage == "layer" else "output"
                        compare(result[key], expected[key])
                        if task.template == "rope_cache" and stage != "layer":
                            compare(model.run_cached(torch.from_numpy(x.copy()), torch.from_numpy(draws.copy())), expected["output"])
                        if stage == "determinism":
                            second = model.run(torch.from_numpy(x.copy()), torch.from_numpy(draws.copy()))
                            assert np.array_equal(array(result["output"]), array(second["output"])), "Repeated call changed output"
            report["gates"].append({"name": stage, "passed": True, "error": metric})
        report["passed"] = True
    except Exception as exc:
        report["gates"].append({"name": stage, "passed": False,
                                "error": metric if metric is None or math.isfinite(metric) else None,
                                "diagnostic": f"{type(exc).__name__}: {exc}"[:2000]})
    return report
