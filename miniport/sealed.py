"""External parity grading with a fail-closed Docker execution boundary."""
from dataclasses import replace
import json
from pathlib import Path
import secrets
import random
import shutil
import subprocess
import tempfile
import threading
import uuid
import os
import signal
import sys

from .integrity import scan, submission_source, audit_passes
from .tasks import GATES, VERIFIER_VERSION

MAX_OUTPUT_BYTES = 8_000_000


class BackendUnavailable(RuntimeError):
    pass


class CandidateFailure(RuntimeError):
    pass


def bounded_run(command, payload, timeout=60, *, env=None, cwd=None, process_group=False):
    """Drain both pipes concurrently, bound aggregate output and execution time."""
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=env, cwd=cwd, start_new_session=process_group)
    output, errors = bytearray(), bytearray()
    overflow = threading.Event()

    def kill():
        try:
            if process_group:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass

    def read(pipe, target):
        while chunk := pipe.read(65536):
            if len(target) + len(chunk) > MAX_OUTPUT_BYTES:
                overflow.set()
                kill()
                break
            target.extend(chunk)

    def write():
        try:
            process.stdin.write(json.dumps(payload).encode())
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    threads = [threading.Thread(target=read, args=(process.stdout, output), daemon=True),
               threading.Thread(target=read, args=(process.stderr, errors), daemon=True),
               threading.Thread(target=write, daemon=True)]
    for thread in threads:
        thread.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        kill()
        process.wait()
        raise CandidateFailure("timeout") from exc
    finally:
        if process_group:
            kill()  # Clean up children even if the leader exited normally.
        for thread in threads:
            thread.join(timeout=1)
        process.stdout.close()
        process.stderr.close()
    if overflow.is_set():
        raise CandidateFailure("output_limit")
    if process.returncode:
        raise CandidateFailure("execution_failed")
    try:
        return json.loads(output)
    except (ValueError, UnicodeError) as exc:
        raise CandidateFailure("invalid_response") from exc


class DockerBackend:
    isolation = "docker"
    def __init__(self, image="miniport-verifier:local", timeout=60):
        self.image = image
        self.timeout = timeout

    def run(self, source, request):
        docker = shutil.which("docker")
        bundled = Path("/Applications/Docker.app/Contents/Resources/bin/docker")
        if not docker and bundled.is_file():
            docker = str(bundled)
        if not docker:
            raise BackendUnavailable("Docker is required; host execution is disabled")
        try:
            info = subprocess.run([docker, "image", "inspect", self.image, "--format", "{{.Id}}"],
                                  capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BackendUnavailable("Docker daemon unavailable") from exc
        if info.returncode:
            raise BackendUnavailable("Verifier image or Docker daemon unavailable")
        image_id = info.stdout.strip()
        self.image_id = image_id
        name = "miniport-" + uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="miniport-sealed-") as directory:
            root = Path(directory)
            (root / "submission").mkdir(mode=0o755)
            (root / "runner").mkdir(mode=0o755)
            (root / "submission" / "candidate.py").write_text(source)
            shutil.copyfile(Path(__file__).with_name("candidate_worker.py"), root / "runner" / "candidate_worker.py")
            command = [docker, "run", "--rm", "--name", name, "--interactive",
                       "--network", "none", "--read-only", "--cap-drop", "ALL",
                       "--security-opt", "no-new-privileges", "--pids-limit", "64",
                       "--memory", "2g", "--memory-swap", "2g", "--cpus", "2",
                       "--user", "65534:65534", "--ulimit", "nofile=128:128",
                       "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
                       "--mount", f"type=bind,src={root / 'submission'},dst=/workspace/submission,readonly",
                       "--mount", f"type=bind,src={root / 'runner'},dst=/runner,readonly", image_id]
            try:
                return bounded_run(command, request, self.timeout)
            finally:
                # Killing the Docker client alone does not guarantee container termination.
                try:
                    subprocess.run([docker, "rm", "-f", name], capture_output=True, timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass


class SubprocessBackend:
    """User-approved, non-isolated CPU execution for bare-metal GPU training."""
    isolation = "subprocess"

    def __init__(self, python=None, timeout=60):
        self.python = python or sys.executable
        self.timeout = timeout

    def run(self, source, request):
        with tempfile.TemporaryDirectory(prefix="miniport-cpu-worker-") as directory:
            root = Path(directory)
            candidate = root / "candidate.py"
            candidate.write_text(source)
            candidate.chmod(0o444)
            worker = root / "candidate_worker.py"
            shutil.copyfile(Path(__file__).with_name("candidate_worker.py"), worker)
            env = dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            try:
                return bounded_run([self.python, "-I", str(worker), str(candidate)], request,
                                   self.timeout, env=env, cwd=root, process_group=True)
            except OSError as exc:
                raise BackendUnavailable("CPU verifier interpreter unavailable") from exc


def make_cases(task, seeds, randomized=False):
    """Create inputs and ground truth in the trusted process; never export targets."""
    from .reference import fixture, reference, mapped_parameters
    cases, expected = [], []
    for i, seed in enumerate(seeds):
        rng = random.Random(seed)
        variant = replace(task, batch=rng.choice([b for b in (1, 2, 3, 4) if b != task.batch]),
                          length=task.length + rng.randint(1, 4)) if randomized else task
        p, x, draws = fixture(variant, seed)
        cases.append({"config": variant.to_dict(), "parameters": {k: v.tolist() for k, v in p.items()},
                      "input": x.tolist(), "draws": draws.tolist()})
        expected.append({"parameters": mapped_parameters(variant, p), **reference(variant, p, x, draws)})
    if not cases:
        raise ValueError("Verifier requires at least one case")
    return {"cases": cases}, expected


def grade(task, response, expected):
    """Treat worker reports as untrusted. Only external comparisons advance gates."""
    import numpy as np
    from .diagnostics import VerificationIssue, runtime_diagnostic
    passed = []
    location = "interface"
    reply = {}

    def compare(payload, target):
        if not isinstance(payload, dict):
            raise VerificationIssue("missing_tensor", message="Expected a returned tensor at this stage.")
        if payload.get("shape") != list(target.shape):
            shape = payload.get("shape")
            shape = shape if isinstance(shape, list) and len(shape) <= 8 and all(type(n) is int for n in shape) else None
            raise VerificationIssue("shape_mismatch", expected_shape=list(target.shape), actual_shape=shape)
        actual = np.asarray(payload["values"])
        if actual.shape != target.shape or actual.dtype.kind not in "fiu" or not np.isfinite(actual).all():
            raise VerificationIssue("nonfinite_or_invalid_tensor", message="Return finite numeric tensors with consistent shapes.")
        integer = np.issubdtype(target.dtype, np.integer)
        dtype = payload.get("dtype")
        dtype = dtype if isinstance(dtype, str) and dtype in {"torch.float16", "torch.float32", "torch.float64",
                 "torch.bfloat16", "torch.int8", "torch.int16", "torch.int32", "torch.int64", "torch.bool"} else "unknown"
        if integer and payload.get("dtype") not in {"torch.int32", "torch.int64"}:
            raise VerificationIssue("dtype_mismatch", required_dtype="torch.int32 or torch.int64", actual_dtype=dtype)
        if not integer and payload.get("dtype") != "torch.float32":
            raise VerificationIssue("dtype_mismatch", required_dtype="torch.float32", actual_dtype=dtype)
        error = float(np.max(np.abs(actual.astype(float) - target.astype(float))))
        valid = np.array_equal(actual, target) if integer else np.allclose(actual, target, atol=2e-5, rtol=2e-4)
        stats = None
        if not valid:
            delta = np.abs(actual.astype(float) - target.astype(float))
            outside = actual != target if integer else delta > (2e-5 + 2e-4 * np.abs(target.astype(float)))
            stats = {"shape": list(target.shape), "actual_dtype": dtype,
                     "expected_dtype": "torch.int32 or torch.int64" if integer else "torch.float32",
                     "max_abs_error": error, "mean_abs_error": float(delta.mean()),
                     "fraction_outside_tolerance": float(outside.mean()),
                     "actual_range": {"min": float(actual.min()), "max": float(actual.max())},
                     "expected_range": {"min": float(target.min()), "max": float(target.max())}}
        return valid, error, stats

    def layer_sign_hint(stats):
        """Identify clipping of a returned intermediate without exposing values."""
        if not stats:
            return None
        actual_min = stats["actual_range"]["min"]
        expected_min = stats["expected_range"]["min"]
        if actual_min >= -2e-5 and expected_min < -2e-5:
            return {
                "code": "possible_early_sign_clipping",
                "message": "Candidate intermediate values are nonnegative while the expected "
                           "intermediate includes negative values. Check whether a clamp, ReLU, "
                           "absolute value, or sign-changing mask was applied before the returned "
                           "intermediate. Use the task's returned-tensor contract to place such "
                           "operations.",
            }
        return None

    try:
        replies = response["cases"]
        if not isinstance(replies, list) or len(replies) != len(expected):
            raise VerificationIssue("case_count", message="Candidate response has the wrong number of cases.")
        for gate in GATES:
            location = "interface" if gate == "topology" else f"{task.template}.{gate}"
            max_error = None
            valid = True
            first_failure = None
            numerical = None
            for case_index, (reply, target) in enumerate(zip(replies, expected)):
                if gate == "topology":
                    if reply.get("topology") is not True and runtime_diagnostic(reply):
                        return {**summary(passed, error_type="candidate_exception", location=location),
                                "diagnostic": runtime_diagnostic(reply)}
                    valid &= reply.get("topology") is True
                    continue
                if gate == "parameters":
                    actual = reply["parameters"]
                    if not isinstance(actual, dict) or set(actual) != set(target["parameters"]):
                        if isinstance(actual, dict):
                            raise VerificationIssue("parameter_keys", missing_keys=sorted(set(target["parameters"]) - set(actual)),
                                                    message="Export exactly the required parameter names.")
                        raise VerificationIssue("parameter_keys", message="export_parameters must return a dictionary of torch.Tensor values.")
                    pairs = [(f"{task.template}.parameters.{k}", actual[k], v) for k, v in target["parameters"].items()]
                elif gate == "layer":
                    pairs = [(f"{task.template}.layer", reply["layer"], target["layer"])]
                else:
                    pairs = [(f"{task.template}.output", reply["output"], target["output"])]
                    if task.template == "rope_cache":
                        pairs.append((f"{task.template}.cached_output", reply["cached"], target["output"]))
                    if gate == "determinism":
                        pairs.append((f"{task.template}.repeat_output", reply["repeat"], target["output"]))
                        if reply["repeat"] != reply["output"]:
                            valid = False
                            first_failure = first_failure or f"{task.template}.repeat_output"
                for location, actual, value in pairs:
                    matches, error, stats = compare(actual, value)
                    if not matches:
                        first_failure = first_failure or location
                        if numerical is None and gate in {"layer", "end_to_end"}:
                            numerical = {"tensor": location, "case_index": case_index,
                                         "scope": "first failing tensor and case", **stats}
                            if gate == "layer":
                                hint = layer_sign_hint(stats)
                                if hint:
                                    numerical["inference"] = hint
                    valid &= matches
                    max_error = max(max_error or 0.0, error)
            if not valid:
                code = "nondeterministic_output" if gate == "determinism" else "value_mismatch"
                return {**summary(passed, max_error, "parity_mismatch", first_failure or location),
                        "diagnostic": {"code": code, "message": "Repeated runs must match exactly." if gate == "determinism"
                                       else "Returned values do not match the reference within tolerance.",
                                       "atol": 2e-5, "rtol": 2e-4,
                                       **({"numerical": numerical} if numerical else {})}}
            passed.append(gate)
        return summary(passed)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        missing = not isinstance(exc, VerificationIssue) or exc.diagnostic["code"] in {"missing_tensor", "parameter_keys"}
        diagnostic = runtime_diagnostic(reply) if missing and isinstance(reply, dict) else None
        diagnostic = diagnostic or (exc.diagnostic if isinstance(exc, VerificationIssue) else
                                   {"code": "invalid_result_structure", "message": "Check the required return keys and tensor types."})
        return {**summary(passed, error_type="invalid_candidate_result", location=location), "diagnostic": diagnostic}


def summary(passed=(), error=None, error_type=None, location=None, outage=False):
    return {"version": VERIFIER_VERSION, "passed_gates": list(passed),
            "current_gate": GATES[len(passed)] if len(passed) < len(GATES) else "complete",
            "max_abs_error": error, "error_type": error_type, "failure_location": location,
            "passed": len(passed) == len(GATES), "environment_outage": outage}


def check_source(task, source, seeds, backend, randomized=False):
    request, expected = make_cases(task, seeds, randomized)
    try:
        return grade(task, backend.run(source, request), expected)
    except BackendUnavailable:
        return summary(error_type="backend_unavailable", outage=True)
    except CandidateFailure as exc:
        hints = {"timeout": "Candidate exceeded the execution time limit.",
                 "output_limit": "Candidate exceeded the output-size limit; avoid printing tensors or debug dumps.",
                 "invalid_response": "Candidate execution did not return the required structured response.",
                 "execution_failed": "Candidate process exited unsuccessfully."}
        return {**summary(error_type=str(exc), location="execution"),
                "diagnostic": {"code": str(exc), "message": hints.get(str(exc), "Candidate execution failed.")}}


def fresh_seeds(task, challenge_path=None):
    """Persist private post-submission challenges for replay across paired conditions."""
    if challenge_path is None:
        return [secrets.randbits(32) for _ in range(4)]
    path = Path(challenge_path)
    if not path.exists():
        payload = {"version": VERIFIER_VERSION, "task_id": task.id,
                   "seeds": [secrets.randbits(32) for _ in range(4)]}
        with path.open("x") as stream:
            json.dump(payload, stream)
        path.chmod(0o600)
    payload = json.loads(path.read_text())
    if payload.get("version") != VERIFIER_VERSION or payload.get("task_id") != task.id:
        raise ValueError("Fresh challenge task/version mismatch")
    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 4 or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Invalid challenge seeds")
    return seeds


def verify_submission(task, directory, seeds=(11, 23, 47), *, backend=None, final=False, audit=None, challenge_path=None, randomized=False):
    try:
        source = submission_source(directory)
    except (OSError, ValueError, UnicodeError):
        return {**summary(error_type="invalid_submission", location="integrity"), "valid": False}
    static = scan(source)
    result = {"submission_sha256": static["sha256"], "static_scan": static, "valid": False}
    if not static["passed"]:
        return {**result, **summary(error_type="static_scan_failed", location="integrity")}
    backend = backend or DockerBackend()
    result.update(check_source(task, source, seeds, backend, randomized=randomized))
    if final:
        # Seeds are sampled only after the exact scanned submission is snapshotted.
        fresh = check_source(task, source, fresh_seeds(task, challenge_path), backend, randomized=True)
        result["fresh_randomization"] = fresh
        result["hidden_parity_passes"] = result["passed"]
        result["fresh_randomization_passes"] = fresh["passed"]
        result["pytorch_only_passes"] = static["passed"]
        result["integrity_audit_passes"] = audit_passes(audit, static["sha256"])
        result["valid"] = all(result[k] for k in ("hidden_parity_passes", "fresh_randomization_passes",
                                                "pytorch_only_passes", "integrity_audit_passes"))
        result["evaluation_terminal_reward"] = 10 if result["valid"] else 0
    result["verifier_image_id"] = getattr(backend, "image_id", None)
    result["isolation"] = getattr(backend, "isolation", "test")
    return result
