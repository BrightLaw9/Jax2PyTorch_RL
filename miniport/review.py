"""Local notebook workflow review; capability correctness stays with the verifier."""
import hashlib
from html import escape
import json
from pathlib import Path

from .requirements import EQUATIONS
from .trajectory import select_checkpoints

LABELS = ("useful diagnosis", "genuine implementation progress", "repetition/waste", "regression",
          "premature optimization", "premature termination", "unclear")


def record_id(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def save_annotation(path, record, reviewer, label, ordinal, stop_decision=None):
    if not reviewer.strip() or label not in LABELS or ordinal not in (-2, -1, 0, 1, 2):
        raise ValueError("Reviewer, workflow label and ordinal score are required")
    is_stop = record["checkpoint"]["action"] == "stop"
    if is_stop and stop_decision not in ("appropriate stop", "should continue", "unclear"):
        raise ValueError("Stop actions require a stop decision")
    row = {"checkpoint_id": record_id(record), "reviewer": reviewer.strip(), "label": label,
           "ordinal": ordinal, "stop_decision": stop_decision if is_stop else None}
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    return row


def create_notebook(log_paths, destination, count=36):
    """Produce a notebook plus selected observable records, without executable user data."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Review directory already exists")
    records = []
    for path in log_paths:
        from .resume import read_jsonl
        records.extend(read_jsonl(path))
    selected = select_checkpoints(records, count)
    if not selected:
        raise ValueError("No checkpoint records to review")
    destination.mkdir(parents=True)
    (destination / "checkpoints.json").write_text(json.dumps(selected, indent=2))
    code = "from miniport.review import show_review\nshow_review('checkpoints.json', 'annotations.jsonl')\n"
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [{"cell_type": "markdown", "id": "instructions", "metadata": {},
                           "source": ["# MiniPort process review\n", "Run with the MiniPort environment and ipywidgets installed. Label workflow quality only; choose unclear when judging requires framework expertise. No chain-of-thought or hidden-test data is included.\n"]},
                          {"cell_type": "code", "id": "review", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": code.splitlines(True)}]}
    (destination / "review.ipynb").write_text(json.dumps(notebook, indent=2))
    return len(selected)


def show_review(records_path, annotations_path):
    import ipywidgets as widgets
    from IPython.display import display
    records = json.loads(Path(records_path).read_text())
    index = widgets.IntSlider(min=0, max=len(records) - 1, description="Checkpoint")
    reviewer = widgets.Text(description="Reviewer")
    label = widgets.Dropdown(options=LABELS, value="unclear", description="Workflow")
    ordinal = widgets.Dropdown(options=(-2, -1, 0, 1, 2), value=0, description="Ordinal")
    stop = widgets.Dropdown(options=("unclear", "appropriate stop", "should continue"), description="Stop")
    panels = [widgets.HTML(layout=widgets.Layout(width="33%", overflow="auto")) for _ in range(3)]
    bottom, status = widgets.HTML(), widgets.HTML()
    save = widgets.Button(description="Save annotation", button_style="primary")

    def pre(value):
        text = value if isinstance(value, str) else json.dumps(value, indent=2)
        return '<pre style="white-space:pre-wrap;max-height:450px;overflow:auto">' + escape(text) + "</pre>"

    def render(change=None):
        r = records[index.value]
        panels[0].value = "<h3>Requirements & gate</h3>" + pre({"task": r["task_id"],
            "requirements": EQUATIONS[r["task_template"]], "gate": r["protected_summary"]["current_gate"]})
        panels[1].value = "<h3>Changed files & diff</h3>" + pre(r["changed_files"]) + pre(r["diff"])
        panels[2].value = "<h3>Protected verification</h3>" + pre({"before": r["before_summary"], "after": r["protected_summary"]})
        bottom.value = "<h3>Observable actions</h3>" + pre({"recent": r["recent_actions"],
            "action": r["checkpoint"]["action"], "command": r["checkpoint"]["command"],
            "proposed_next_action": r["proposed_next_action"], "remaining_actions": r["checkpoint"]["remaining_actions"]})
        label.value, ordinal.value, stop.value = "unclear", 0, "unclear"
        stop.disabled = r["checkpoint"]["action"] != "stop"
        status.value = ""

    def commit(button):
        try:
            save_annotation(annotations_path, records[index.value], reviewer.value, label.value, ordinal.value, stop.value)
            status.value = "Saved."
        except ValueError as exc:
            status.value = escape(str(exc))

    index.observe(render, names="value")
    save.on_click(commit)
    render()
    display(widgets.VBox([index, widgets.HBox(panels), bottom, reviewer, label, ordinal, stop, save, status]))
