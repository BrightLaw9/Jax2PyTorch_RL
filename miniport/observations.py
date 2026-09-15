"""Fixed visible feedback shared by every reward/credit-assignment condition."""
import math


def visible_feedback(report):
    if report.get("environment_outage"):
        return {"status": "environment_unavailable"}
    if report["passed"]:
        return {"status": "pass"}
    error = report.get("max_abs_error")
    feedback = {"status": "fail", "failed_gate": report["current_gate"],
            "max_abs_error": float(f"{error:.6g}") if isinstance(error, (int, float)) and math.isfinite(error) else None,
            "failure_location": report.get("failure_location"), "error_type": report.get("error_type")}
    if report.get("error_type") == "static_scan_failed":
        feedback["static_scan_findings"] = [
            {"line": finding["line"], "rule": finding["rule"],
             **({"message": finding["message"]} if "message" in finding else {})}
            for finding in report["static_scan"]["findings"]
        ]
    if report.get("diagnostic"):
        feedback["diagnostic"] = report["diagnostic"]
    return feedback
