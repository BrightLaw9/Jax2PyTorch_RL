"""Conservative submission admission checks; not a proof of semantic integrity."""
import ast
import hashlib
from pathlib import Path

MAX_SOURCE_BYTES = 128_000
ALLOWED_IMPORTS = {"torch", "math", "typing", "dataclasses", "collections", "functools",
                   "itertools", "operator", "enum", "abc", "numbers", "copy"}
BANNED_NAMES = {"eval", "exec", "compile", "open", "__import__", "input", "breakpoint"}
# Restrict external I/O and execution facilities, not ordinary object method names.
BLOCKED_TORCH_PATHS = {"torch.load", "torch.save", "torch.load_library", "torch.hub",
                       "torch.distributed", "torch.multiprocessing", "torch.serialization",
                       "torch.utils.cpp_extension", "torch.ops.load_library",
                       "torch.classes.load_library", "torch.from_file"}


def submission_source(directory):
    """Only one source file is currently admitted; caches and tests are rejected."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Submission must be a real directory")
    entries = list(directory.iterdir())
    if len(entries) != 1 or entries[0].name != "candidate.py":
        raise ValueError("Submission must contain only candidate.py")
    path = entries[0]
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("Invalid submission source file")
    data = path.read_bytes()
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError("Submission exceeds size limit")
    return data.decode("utf-8")


def scan(source):
    findings = []
    digest = hashlib.sha256(source.encode()).hexdigest()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"passed": False, "findings": [{"line": exc.lineno, "rule": "syntax",
                "message": exc.msg, "column": exc.offset}], "sha256": digest}

    def reject(node, rule, message):
        findings.append({"line": node.lineno, "rule": rule, "message": message})

    aliases = {"torch": "torch"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                aliases[item.asname or item.name] = (node.module or '') + '.' + item.name

    def path(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = path(node.value)
            return base + '.' + node.attr if base else None
        return None

    def blocked(name):
        return name and any(name == p or name.startswith(p + '.') for p in BLOCKED_TORCH_PATHS)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for module in modules:
                if module.split('.')[0] not in ALLOWED_IMPORTS:
                    reject(node, "forbidden_import", f"Import '{module}' is not permitted. Use PyTorch for numerical computation; allowed imports: {', '.join(sorted(ALLOWED_IMPORTS))}.")
                elif blocked(module):
                    reject(node, "forbidden_import", f"Import '{module}' provides external I/O or execution facilities, which are not available to submissions.")
            if isinstance(node, ast.ImportFrom):
                if node.level or any(a.name == '*' for a in node.names):
                    reject(node, "indirect_import", "Use explicit absolute imports; relative and wildcard imports cannot be checked reliably.")
                for item in node.names:
                    name = (node.module or '') + '.' + item.name
                    if blocked(name):
                        reject(node, "forbidden_import", f"Import '{name}' provides external I/O or execution facilities.")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in BANNED_NAMES:
            reject(node, "dynamic_execution_or_introspection", f"'{node.id}' is unavailable: submissions must not access files, request interactive input, or dynamically execute Python source. Use normal Python functions and objects.")
        if isinstance(node, ast.Attribute) and blocked(path(node)):
            reject(node, "forbidden_attribute", f"'{path(node)}' provides external I/O or execution facilities. Use the parameters supplied to load_parameters and return tensors directly.")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute):
                    root = target
                    while isinstance(root, ast.Attribute):
                        root = root.value
                    if isinstance(root, ast.Name) and root.id in aliases:
                        reject(node, "external_attribute_assignment", f"Assignment to '{ast.unparse(target)}' modifies an imported object. Assign parameters on your own model instances instead; local objects such as conv.weight.data are allowed.")
    return {"passed": not findings, "findings": findings, "sha256": digest}


AUDIT_CHECKS = ("pytorch_only", "generic_implementation", "generic_parameter_mapping",
                "test_integrity", "sensible_artifact")


def audit_passes(audit, submission_hash):
    """Only an explicit, artifact-bound complete human audit makes a pass valid."""
    return bool(audit and audit.get("submission_sha256") == submission_hash
                and audit.get("reviewer") and audit.get("decision") == "valid"
                and all(audit.get("checks", {}).get(check) == "valid" for check in AUDIT_CHECKS))


def audit_bundle(directory, result, action_log):
    """Review evidence only for numerical passers, with no reference or hidden inputs."""
    import difflib
    source = submission_source(directory)
    digest = hashlib.sha256(source.encode()).hexdigest()
    if result.get("submission_sha256") != digest:
        raise ValueError("Submission changed since evaluation")
    if not result.get("hidden_parity_passes") or not result.get("fresh_randomization_passes"):
        raise ValueError("Audit only candidates that pass both numerical checks")
    stub = Path(__file__).with_name("stub.py.txt").read_text()
    return {"submission_sha256": digest, "source_files": {"candidate.py": source},
            "diff_against_stub": "".join(difflib.unified_diff(stub.splitlines(True), source.splitlines(True))),
            "verifier_summary": {k: result.get(k) for k in ("version", "hidden_parity_passes",
                "fresh_randomization_passes", "pytorch_only_passes", "verifier_image_id")},
            "static_scan": result["static_scan"], "observable_action_log": action_log,
            "audit_form": {"submission_sha256": digest, "reviewer": "", "decision": "unclear",
                           "checks": {k: "unclear" for k in AUDIT_CHECKS}}}
