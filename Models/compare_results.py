"""Create cross-validation reports from runs/cv/*/*/cv_summary.json.

Usage: python compare_results.py [--runs-dir runs] [--output-dir Results]
Automatically includes normal and weighted runs stored in the same runs/cv tree,
as well as earlier weighted runs saved in sibling runs_weighted/cv.
Only complete five-fold runs are included. No model training or scoring is performed.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path

LABELS = (
    "Cardiovascular-Pulmonary", "Dermatology", "Gastroenterology",
    "General Medicine", "Neurology", "Neurosurgery", "Obstetrics-Gynecology",
    "Ophthalmology", "Orthopedic", "Psychiatry-Psychology", "Radiology", "Surgery",
)
MODELS = ("naive_bayes", "birnn", "birnn_attention", "bilstm", "bilstm_attention")
DISPLAY_MODELS = {
    "naive_bayes": "TF-IDF + Multinomial NB",
    "birnn": "BiRNN", "birnn_attention": "BiRNN + Attention",
    "bilstm": "BiLSTM", "bilstm_attention": "BiLSTM + Attention",
}
COLUMNS = ("accuracy", "macro_f1", "weighted_f1", "min_class_f1")
COL_NAMES = {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "weighted_f1": "Weighted-F1",
             "min_class_f1": "Min class F1"}
STYLE = """
:root{color-scheme:light;font-family:Inter,system-ui,-apple-system,Segoe UI,Arial,sans-serif;
--text:#13233a;--muted:#64748b;--bg:#f3f6fb;--edge:#dbe3ef;--navy:#163451;--cyan:#1c8eaa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text)}
.container{max-width:1250px;margin:auto;padding:32px 28px 70px}
.header{display:flex;align-items:center;justify-content:space-between;gap:15px;flex-wrap:wrap}
.kicker{text-transform:uppercase;letter-spacing:.12em;font-size:12px;font-weight:800;color:#527189}
h1{font-size:clamp(27px,4vw,40px);letter-spacing:-.035em;margin:8px 0 6px}
h2{font-size:20px;margin:28px 0 14px}h3{font-size:17px;margin:18px 0 10px}
p{color:var(--muted);line-height:1.55;margin:10px 0 18px}
a{color:#066989;text-decoration:none}a:hover{text-decoration:underline}
.btn{border:1px solid var(--edge);border-radius:11px;background:white;padding:10px 13px;color:var(--navy);font-weight:650;display:inline-block}
.panel{border:1px solid var(--edge);background:white;border-radius:16px;padding:20px 22px;margin:18px 0;box-shadow:0 3px 14px #142f4810}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:13px;margin:20px 0}
.card{border:1px solid var(--edge);background:white;border-radius:14px;padding:18px}
.card .number{font-size:27px;font-weight:750;color:#103b58;margin:7px 0 3px}
.caption,small{color:var(--muted);font-size:12px}.muted{color:var(--muted)}
.tablewrap{overflow-x:auto;width:100%}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:13px}
th{text-align:left;color:#3b5871;background:#eef3f8;font-weight:800;font-size:12px;letter-spacing:.02em}
th,td{padding:11px 12px;border-bottom:1px solid #e9edf4}tr:last-child td{border-bottom:0}
tbody tr:hover{background:#f6fbfe}td.num{text-align:right;font-variant-numeric:tabular-nums}
.b{font-weight:800;color:#12415d}.bar{height:6px;width:90px;background:#e8edf4;border-radius:4px;overflow:hidden;margin-top:5px}
.bar span{display:block;background:#3283a4;height:100%;border-radius:4px}
.badge{display:inline-block;border:1px solid #cddfe9;border-radius:7px;background:#e8f5f9;color:#19637d;padding:3px 8px;font-size:12px;font-weight:750}
.notice{padding:12px 15px;border-radius:10px;background:#edf5f8;color:#31536c;border:1px solid #d6e5ee;font-size:13px}
.warn{background:#fff7ec;border-color:#f7e0b6;color:#81521c}
.links{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}
.linkcard{display:block;background:#fff;border:1px solid var(--edge);padding:15px 17px;border-radius:12px}
.linkcard strong{display:block;color:#204664}.linkcard small{display:block;margin-top:5px}
.heat{font-variant-numeric:tabular-nums;text-align:center;min-width:52px}
.mono{font-family:ui-monospace,monospace;font-size:12px;white-space:normal;overflow-wrap:anywhere}
.footer{border-top:1px solid var(--edge);padding-top:18px;margin-top:32px;font-size:12px;color:#7b8fa2}
"""


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, title: str, body: str, depth: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    back = "../" * depth + "index.html"
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(title)} · Clinical NLP</title>
<style>{STYLE}</style></head><body><main class="container">
<div class="header"><span class="kicker">Clinical NLP · 5-Fold Cross-Validation</span>
<a class="btn" href="{esc(back)}">← Dashboard</a></div><h1>{esc(title)}</h1>{body}
<div class="footer">All evaluations are out-of-fold on labelled development data. The official unlabelled test set is not evaluated.</div>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def title_config(slug: str) -> str:
    tokens = slug.split("__")
    base = tokens[0]
    if base == "all_four_fields":
        title = "All four fields"
    elif base.endswith("_only"):
        title = base[:-5].replace("_", " ").title() + " only"
    else:
        title = " + ".join(word.replace("_", " ").title() for word in tokens
                           if not word.startswith("prefix_") and word != "full_text")
    if "prefix_removed" in tokens:
        title += " · Prefix removed"
    elif "prefix_only" in tokens:
        title += " · Specialty prefix only*"
    if "full_text" in tokens:
        title += " · Full text (no truncation)"
    return title


def title_model(model: str) -> str:
    return DISPLAY_MODELS.get(model, model)


def val(run, metric):
    return float(run["metrics"][metric]["mean"])


def sd(run, metric):
    return float(run["metrics"][metric]["sd"])


def pooled(run, metric):
    return float(run["pooled_oof"][metric])


def f3(x):
    return f"{float(x):.3f}"


def percent(x):
    return f"{100 * float(x):.1f}%"


def metric_html(x, std, *, pct=False, bar=False):
    value = percent(x) if pct else f3(x)
    error = percent(std) if pct else f3(std)
    graph = f'<div class="bar"><span style="width:{max(0,min(100,float(x)*100)):.2f}%"></span></div>' if bar else ""
    return f'<div class="b">{value} ± {error}</div>{graph}'


def info_box(text: str, warning=False):
    return f'<div class="notice {"warn" if warning else ""}">{esc(text)}</div>'


def weighting_label(item: dict) -> str:
    weighting = item["weighting"]
    return "Normal" if weighting == "none" else f"Weighted ({weighting})"


def stored_model_name(item: dict) -> str:
    return item["model"] if item["weighting"] == "none" else f'{item["model"]}__{item["weighting"]}'


def collect(runs: Path, *, legacy_weighted_dir: bool = False):
    found, warnings = [], []
    cv_dir = runs / "cv"
    if not cv_dir.exists():
        return found, warnings
    for path in sorted(cv_dir.glob("*/*/cv_summary.json")):
        try:
            data = read_json(path)
            model_dir = path.parent.name
            configuration = path.parent.parent.name
            config = data["configuration"]
            model = config["model"]
            weight = config.get("class_weighting", "none")
            expected_dir = model if weight == "none" else f"{model}__{weight}"
            # Previous weighted commands used `--runs-dir runs_weighted` and
            # saved unsuffixed model directory names. Continue recognizing them.
            legacy_path = legacy_weighted_dir and model_dir == model
            if (data["cv_version"] != 1 or data["n_folds"] != 5 or
                    model not in MODELS or weight not in ("none", "sqrt_inv") or
                    (model == "naive_bayes" and weight != "none") or
                    (model_dir != expected_dir and not legacy_path) or
                    config["config_name"] != configuration):
                raise ValueError("Invalid CV schema/model name/class weighting")
            if not all(metric in data["metrics"] for metric in COLUMNS):
                raise ValueError("Missing performance metrics")
            if not all(label in data["per_class_f1"] for label in LABELS):
                raise ValueError("Missing a specialty")
            if not all((path.parent / f"fold_{n:02d}" / "outer_metrics.json").exists()
                       for n in range(1, 6)):
                raise ValueError("Missing outer-fold data")
            found.append({"data": data, "model": model, "configuration": configuration,
                          "weighting": weight, "source": path.parent, "runs_root": runs})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"Skipped {path}: {exc}")
    # Report unfinished experiments without showing misleading incomplete averages.
    for model_path in sorted(cv_dir.glob("*/*")):
        if model_path.is_dir() and (model_path / "fold_01").exists() and not (model_path / "cv_summary.json").exists():
            warnings.append(f"Incomplete five-fold run: {model_path.relative_to(runs)}")
    return found, warnings


def collect_all(runs_dir: Path):
    # Both `--runs-dir runs` and older `--runs-dir runs_weighted` workflows
    # generate one unified dashboard. The normal directory always takes precedence
    # if identical runs were saved in both places.
    if runs_dir.name.endswith("_weighted"):
        normal_dir = runs_dir.with_name(runs_dir.name[:-len("_weighted")])
        roots = [(normal_dir, False), (runs_dir, True)]
    else:
        roots = [(runs_dir, False)]
        legacy_dir = runs_dir.with_name(runs_dir.name + "_weighted")
        if legacy_dir.is_dir():
            roots.append((legacy_dir, True))
    result, warnings, seen = [], [], set()
    manifests = []
    for root, legacy in roots:
        entries, problems = collect(root, legacy_weighted_dir=legacy)
        warnings.extend(problems)
        manifest_file = root / "cv_splits.json"
        if manifest_file.is_file() and entries:
            manifests.append((manifest_file, read_json(manifest_file)))
        for entry in entries:
            # Avoid duplicated Naive Bayes baselines (it has no class weighting)
            # and duplicate experiments saved in both the old/new directory layout.
            key = json.dumps(entry["data"]["configuration"], sort_keys=True)
            if key not in seen:
                seen.add(key)
                result.append(entry)
    if len(manifests) > 1:
        reference = manifests[0][1]
        for path, other in manifests[1:]:
            if (other.get("fingerprint"), other.get("seed"), other.get("records")) != (reference.get("fingerprint"), reference.get("seed"), reference.get("records")):
                warnings.append(f"WARNING: CV fold assignments in {path} differ from {manifests[0][0]}; weighted/unweighted comparisons are not paired.")
    return result, warnings


def summary_row(item: dict, label: str):
    data = item["data"]
    values = {"name": label, "config": item["configuration"], "model": item["model"],
              "training": weighting_label(item), "class_weighting": item["weighting"],
              "n": data["n_examples"], "passing_folds": data["folds_passing_both_thresholds"],
              "parameters_mean": round(data["parameter_count"]["mean"]),
              "best_epoch_mean": ("" if item["model"] == "naive_bayes" else data["best_epoch"]["mean"]),
              "best_epoch_sd": ("" if item["model"] == "naive_bayes" else data["best_epoch"]["sd"])}
    for field in COLUMNS:
        values[field + "_mean"] = val(data, field)
        values[field + "_sd"] = sd(data, field)
        values[field + "_pooled"] = pooled(data, field)
    return values


MAIN_COLUMNS = ["name", "config", "model", "training", "class_weighting", "n"] + [k + suffix for k in COLUMNS
              for suffix in ("_mean", "_sd", "_pooled")] + ["passing_folds", "parameters_mean", "best_epoch_mean", "best_epoch_sd"]


def comparison_table(items: list[dict], name_fn):
    ordered = sorted(items, key=lambda item: val(item["data"], "macro_f1"), reverse=True)
    heading = "<tr><th>Experiment</th>" + "".join(
        f"<th>{COL_NAMES[key]}<br><small>fold mean ± SD</small></th>" for key in COLUMNS
    ) + "<th>Macro-F1<br><small>pooled OOF</small></th><th>Pass folds</th><th>Avg. best epoch</th></tr>"
    rows = []
    for item in ordered:
        data = item["data"]
        columns = [f"<td><strong>{esc(name_fn(item))}</strong></td>"]
        for metric in COLUMNS:
            columns.append(f'<td class="num">{metric_html(val(data,metric),sd(data,metric), pct=metric=="accuracy",bar=metric=="macro_f1")}</td>')
        columns.extend([
            f'<td class="num b">{f3(pooled(data,"macro_f1"))}</td>',
            f'<td class="num">{data["folds_passing_both_thresholds"]}/5</td>',
            (f'<td class="num">{f3(data["best_epoch"]["mean"])} ± {f3(data["best_epoch"]["sd"])}</td>'
             if item["model"] != "naive_bayes" else '<td class="num muted">—</td>'),
        ])
        rows.append("<tr>" + "".join(columns) + "</tr>")
    return f'<div class="tablewrap"><table><thead>{heading}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def per_class_comparison(items, name_fn):
    ordered = sorted(items, key=lambda item: val(item["data"], "macro_f1"), reverse=True)
    heading = "<tr><th>Medical specialty</th>" + "".join(f"<th>{esc(name_fn(item))}</th>" for item in ordered) + "</tr>"
    rows = []
    for label in LABELS:
        cells = [f"<td>{esc(label)}</td>"]
        for item in ordered:
            data = item["data"]
            metric = data["per_class_f1"][label]
            cells.append(f'<td class="num"><div>{f3(metric["mean"])} ± {f3(metric["sd"])}</div>'
                         f'<small>pooled: {f3(data["pooled_oof"]["per_class_f1"][label])}</small></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tablewrap"><table><thead>{heading}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def create_comparisons(results: Path, group_name: str, sets: dict, name_fn, title_fn):
    output = results / group_name
    output.mkdir(parents=True, exist_ok=True)
    created = []
    for key, items in sorted(sets.items()):
        if not items:
            continue
        # Suppress unsafe comparisons with different training datasets / frozen folds.
        fingerprints = {item["data"]["data_fingerprint"] for item in items}
        split_seeds = {item["data"]["configuration"]["seed"] for item in items}
        body = f"<p>{esc(len(items))} completed five-fold experiment(s). Fold mean ± sample standard deviation; pooled OOF is computed over all labelled examples.</p>"
        if len(fingerprints) != 1 or len(split_seeds) != 1:
            body += info_box("WARNING: These results have differing datasets/seeds and are not directly paired comparisons.", True)
        if any(item["model"] == "naive_bayes" for item in items) and any(item["model"] != "naive_bayes" for item in items):
            body += info_box("Training-data difference: Naive Bayes uses the full outer-training partition (~80%); recurrent models reserve an inner-validation partition for early stopping (~64% actually fitted).")
        body += '<section class="panel"><h2>Aggregate performance</h2>' + comparison_table(items, name_fn) + '</section>' 
        body += '<section class="panel"><h2>Per-class F1</h2><p>Top line: fold mean ± SD. Second line: pooled out-of-fold F1.</p>' + per_class_comparison(items, name_fn) + '</section>'
        title = title_fn(key)
        write_html(output / (key + ".html"), title, body, 1)
        write_csv(output / (key + ".csv"), [summary_row(item, name_fn(item)) for item in items], MAIN_COLUMNS)
        created.append((title, f"{group_name}/{key}.html", len(items)))
    return created


def create_metrics(results: Path, item: dict):
    data = item["data"]
    model, config = item["model"], item["configuration"]
    out = results / "Metrics" / config / stored_model_name(item)
    out.mkdir(parents=True, exist_ok=True)
    pooled_result = data["pooled_oof"]
    report = pooled_result["classification_report"]
    records = []
    for label in LABELS:
        fold = data["per_class_f1"][label]
        record = {
            "specialty": label, "f1_mean": fold["mean"], "f1_sd": fold["sd"],
            "f1_pooled_oof": pooled_result["per_class_f1"][label],
            "precision_pooled": report[label]["precision"], "recall_pooled": report[label]["recall"],
            "support": report[label]["support"], "f1_at_least_025": int(pooled_result["per_class_f1"][label] >= .25),
        }
        for i, f1 in enumerate(fold["folds"], start=1):
            record[f"fold_{i:02d}_f1"] = f1
        records.append(record)
    write_csv(out / "f1_per_class.csv", records,
              ["specialty", "f1_mean", "f1_sd", "f1_pooled_oof"] + [f"fold_{i:02d}_f1" for i in range(1, 6)])
    write_csv(out / "per_class_metrics.csv", records, list(records[0]))
    matrix = pooled_result["confusion_matrix"]
    write_csv(out / "confusion_matrix.csv", [dict(zip(LABELS, row), **{"actual\\predicted": label})
              for label, row in zip(LABELS, matrix)], ["actual\\predicted", *LABELS])
    normalized = [[float(value) / sum(row) if sum(row) else 0.0 for value in row] for row in matrix]
    write_csv(out / "confusion_matrix_normalized.csv", [dict(zip(LABELS, row), **{"actual\\predicted": label})
              for label, row in zip(LABELS, normalized)], ["actual\\predicted", *LABELS])
    write_json(out / "summary.json", data)
    fold_rows = []
    for i in range(5):
        fold_rows.append({"fold": i + 1, **{field: data["metrics"][field]["folds"][i] for field in COLUMNS},
                          "best_epoch": (None if model == "naive_bayes" else data["best_epoch"]["folds"][i]), "outer_size": data["fold_sizes"][i]})
    write_csv(out / "fold_metrics.csv", fold_rows, list(fold_rows[0]))

    cells = []
    for record in records:
        value = record["f1_pooled_oof"]
        cells.append(f'<tr><td>{esc(record["specialty"])}</td>'
                     f'<td class="num">{metric_html(record["f1_mean"], record["f1_sd"], bar=True)}</td>'
                     f'<td class="num b">{f3(value)}</td>'
                     f'<td class="num">{f3(record["precision_pooled"])}</td>'
                     f'<td class="num">{f3(record["recall_pooled"])}</td>'
                     f'<td class="num">{int(record["support"])}</td>'
                     f'<td><span class="badge">{"PASS" if value >= .25 else "BELOW"}</span></td></tr>')
    table = ("<tr><th>Specialty</th><th>F1 mean ± SD</th><th>Pooled F1</th>"
             "<th>Pooled precision</th><th>Pooled recall</th><th>Support</th><th>F1 ≥ 0.25</th></tr>")
    body = (f'<p>{esc(title_config(config))} / {esc(title_model(model))} / {esc(weighting_label(item))}. '
            f'5 outer folds; {data["n_examples"]} held-out predictions, one per example.</p>')
    if model == "naive_bayes":
        body += info_box("Professor-specified baseline: description only, lowercase, Porter stemming, TF-IDF word unigrams, Multinomial Naive Bayes (alpha=1). Each fold fits on the full outer-training partition; no epochs or early stopping.")
    body += '<div class="cards">' + "".join(
        f'<div class="card"><small>{esc(COL_NAMES[field])}</small><div class="number">{f3(val(data,field))}</div>'
        f'<small>SD {f3(sd(data,field))} · pooled {f3(pooled(data,field))}</small></div>'
        for field in COLUMNS) + '</div>'
    body += f'<section class="panel"><h2>All 12 medical specialties</h2><div class="tablewrap"><table><thead>{table}</thead><tbody>{"".join(cells)}</tbody></table></div></section>'
    body += '<section class="panel"><h2>Individual outer folds</h2>'
    body += '<div class="tablewrap"><table><tr><th>Fold</th>' + "".join(f"<th>{esc(COL_NAMES[k])}</th>" for k in COLUMNS) + '<th>Best epoch</th></tr>'
    for fold in fold_rows:
        body += f'<tr><td>Fold {fold["fold"]}</td>' + "".join(f'<td class="num">{f3(fold[k])}</td>' for k in COLUMNS) + f'<td class="num">{int(fold["best_epoch"]) if fold["best_epoch"] is not None else "—"}</td></tr>'
    body += '</table></div></section>'
    body += '<section class="panel"><h2>Pooled confusion matrix</h2><p>Rows = correct labels; columns = predicted labels. Values are summed from all five outer folds.</p>'
    body += '<div class="tablewrap"><table><thead><tr><th>Actual ↓ / Predicted →</th>' + "".join(f'<th>{esc(k)}</th>' for k in LABELS) + '</tr></thead><tbody>'
    for label, row, ratios in zip(LABELS, matrix, normalized):
        body += f'<tr><td><strong>{esc(label)}</strong></td>'
        for number, ratio in zip(row, ratios):
            body += f'<td class="heat" style="background:rgba(45,132,164,{ratio*0.58:.3f})">{number}</td>'
        body += '</tr>'
    body += '</tbody></table></div></section>'
    body += f'<p>Raw predictions and misclassifications: <span class="mono">{esc(item["source"])}</span>.</p>'
    write_html(out / "metrics.html", f"{title_model(model)} · {title_config(config)} · {weighting_label(item)}", body, 3)
    return (f"{title_config(config)} / {title_model(model)} / {weighting_label(item)}",
            f"Metrics/{config}/{stored_model_name(item)}/metrics.html", 1)



def augmentation_runs(augmentation_root: Path):
    """Read only complete method results from the existing augmentation runner.

    No model is trained and no predictions are recomputed. Unlike the normal
    dashboard, these runs may quarantine source rows and use different folds.
    """
    groups, warnings = [], []
    base = augmentation_root / "original_models"
    if not base.is_dir():
        return groups, warnings
    for cfg in sorted(base.glob("*/*/*/run_config.json")):
        folder = cfg.parent
        try:
            config = read_json(cfg)
            model = config["model"]
            weight = config.get("class_weighting", "none")
            if model not in (*MODELS, "linear_svm") or weight not in ("none", "sqrt_inv"):
                raise ValueError("Unknown model or weighting")
            if folder.parent.parent.parent != base or folder.parent.parent.name != model:
                raise ValueError("Unexpected output folder")
            if folder.parent.name != ("weighted_sqrt_inv" if weight == "sqrt_inv" else "normal"):
                raise ValueError("Weighting and folder disagree")
            if not isinstance(config.get("fields"), list) or not config["fields"]:
                raise ValueError("Missing input fields")
            comparison = folder / "comparison.csv"
            if not comparison.is_file():
                continue
            with comparison.open(encoding="utf-8-sig", newline="") as stream:
                summary = list(csv.DictReader(stream))
            with (folder / "fold_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
                fold_rows = list(csv.DictReader(stream))
            methods, seen = [], set()
            for record in summary:
                method = record.get("method", "")
                if method not in ("baseline", "rwd", "llm", "combined") or method in seen:
                    warnings.append(f"Skipped invalid augmentation method in {comparison}: {method}")
                    continue
                seen.add(method)
                try:
                    perclass_file = folder / f"per_class_{method}.csv"
                    confusion_file = folder / f"confusion_{method}.csv"
                    oof_file = folder / f"oof_{method}.csv"
                    if not all(path.exists() for path in (perclass_file, confusion_file, oof_file)):
                        raise ValueError("Missing method-specific metrics/predictions")
                    with perclass_file.open(encoding="utf-8-sig", newline="") as stream:
                        classes = list(csv.DictReader(stream))
                    with confusion_file.open(encoding="utf-8-sig", newline="") as stream:
                        confusion = list(csv.DictReader(stream))
                    with oof_file.open(encoding="utf-8-sig", newline="") as stream:
                        oof = list(csv.DictReader(stream))
                    folds = sorted((r for r in fold_rows if r.get("method") == method),
                                   key=lambda row: int(row["fold"]))
                    if ([int(f["fold"]) for f in folds] != [1, 2, 3, 4, 5]
                            or sorted(set(int(r["fold"]) for r in oof)) != [1, 2, 3, 4, 5]):
                        raise ValueError("Missing or duplicated fold metrics")
                    if (len(classes) != len(LABELS) or {r["specialty"] for r in classes} != set(LABELS)
                            or len(confusion) != len(LABELS) or {r["true_label"] for r in confusion} != set(LABELS)):
                        raise ValueError("Missing or repeated medical specialties")
                    if (not oof or len(oof) != int(record["validation_originals"])
                            or len({r["source_row_id"] for r in oof}) != len(oof)):
                        raise ValueError("Incomplete or duplicated out-of-fold predictions")
                    for key in ("accuracy", "macro_f1", "weighted_f1", "minimum_class_f1"):
                        number = float(record[key])
                        if not 0 <= number <= 1:
                            raise ValueError(f"Metric out of bounds: {key}")
                    if abs(min(float(row["f1"]) for row in classes)
                           - float(record["minimum_class_f1"])) > 0.00001:
                        raise ValueError("Per-class F1 does not match aggregate")
                    summaries = []
                    if model not in ("naive_bayes", "linear_svm"):
                        for i in range(1, 6):
                            path = folder / method / f"fold_{i:02d}" / "training_summary.json"
                            if not path.is_file():
                                raise ValueError(f"Missing best-epoch summary for fold {i}")
                            summaries.append(read_json(path))
                    methods.append({"method": method, "summary": record, "classes": classes,
                                    "confusion": confusion, "folds": folds,
                                    "best_epochs": [int(v["best_epoch"]) for v in summaries],
                                    "count": len(oof)})
                except (KeyError, ValueError, TypeError, OSError) as exc:
                    warnings.append(f"Incomplete augmentation {folder.name}/{method}: {exc}")
            if methods:
                groups.append({"config": config, "folder": folder, "methods": methods})
        except (KeyError, ValueError, TypeError, OSError) as exc:
            warnings.append(f"Skipped augmentation run {folder}: {exc}")
    return groups, warnings


AUGMENTATION_NAMES = {
    "baseline": "Original data", "rwd": "Random Word Dropout",
    "llm": "LLM paraphrasing", "combined": "Combined (RWD + LLM)",
}


def augmentation_title(group):
    config = group["config"]
    fields = " + ".join(f.replace("_", " ").title() for f in config["fields"])
    if config.get("strip_keyword_prefix"):
        fields += " · Prefix removed"
    fields += " · Full text"
    weighting = config.get("class_weighting", "none")
    model = config["model"]
    label = title_model(model) if model != "linear_svm" else "Linear SVM"
    label += " · Normal" if weighting == "none" else f" · Weighted ({weighting})"
    return f"{label} · {fields}"


def augmentation_table(rows, *, link_prefix=None):
    head = ("<tr><th>Augmentation</th><th>Accuracy</th><th>Macro-F1</th>"
            "<th>Weighted-F1</th><th>Minimum class F1</th><th>Records evaluated</th>"
            "<th>Mean best epoch</th></tr>")
    cells = []
    for item in sorted(rows, key=lambda i: -float(i["summary"]["macro_f1"])):
        values = item["summary"]
        name = AUGMENTATION_NAMES[item["method"]]
        if link_prefix is not None:
            name = f'<a href="{esc(link_prefix + "/" + item["method"] + ".html")}">{esc(name)}</a>'
        else:
            name = esc(name)
        epochs = item["best_epochs"]
        average = f3(sum(epochs) / len(epochs)) if epochs else "—"
        cells.append("<tr>" + f"<td>{name}</td>" + "".join(
            f'<td class="num">{f3(values[key])}</td>'
            for key in ("accuracy", "macro_f1", "weighted_f1", "minimum_class_f1")
        ) + f'<td class="num">{item["count"]}</td><td class="num">{average}</td></tr>')
    return '<div class="tablewrap"><table><thead>' + head + '</thead><tbody>' + "".join(cells) + '</tbody></table></div>'


def create_augmentation_dashboard(output: Path, augmentation_root: Path):
    """Generate Reports/Augmentation only after normal reports are regenerated."""
    groups, warnings = augmentation_runs(augmentation_root)
    if not groups:
        return 0, warnings
    aug_output = output / "Augmentation"
    aug_output.mkdir(parents=True, exist_ok=True)
    links, flat_summary = [], []
    for group in groups:
        config = group["config"]
        # Names use input-configuration slugs generated by the training script.
        key = (group["folder"].parent.parent.name + "__" + group["folder"].parent.name + "__"
               + group["folder"].name)
        title = augmentation_title(group)
        methods = group["methods"]
        for item in methods:
            r = item["summary"]
            epochs = item["best_epochs"]
            flat_summary.append({
                "model": config["model"], "class_weighting": config.get("class_weighting", "none"),
                "fields": " ".join(config["fields"]),
                "prefix_removed": bool(config.get("strip_keyword_prefix")),
                "augmentation": item["method"], "n": item["count"],
                **{k: r[k] for k in ("accuracy", "macro_f1", "weighted_f1", "minimum_class_f1")},
                "mean_best_epoch": (sum(epochs) / len(epochs) if epochs else ""),
            })
            classes = {r["specialty"]: r for r in item["classes"]}
            lines = []
            for label in LABELS:
                entry = classes[label]
                lines.append("<tr>" + f"<td>{esc(label)}</td>" + "".join(
                    f'<td class="num">{f3(entry[metric])}</td>'
                    for metric in ("precision", "recall", "f1")
                ) + f'<td class="num">{int(entry["support"])}</td></tr>')
            detail = (f'<p>{esc(title)} · {esc(AUGMENTATION_NAMES[item["method"]])}. '
                      'Five held-out outer folds; synthetic rows were excluded from evaluation.</p>'
                      '<p><a class="btn" href="../../index.html">← Augmentation overview</a></p>'
                      '<section class="panel"><h2>Aggregate performance</h2>'
                      + augmentation_table([item]) + '</section>'
                      '<section class="panel"><h2>All 12 medical specialities</h2>'
                      '<div class="tablewrap"><table><thead><tr><th>Speciality</th>'
                      '<th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>'
                      '<tbody>' + "".join(lines) + '</tbody></table></div></section>')
            detail += '<section class="panel"><h2>Individual folds</h2><div class="tablewrap"><table><thead>'
            detail += '<tr><th>Fold</th><th>Accuracy</th><th>Macro-F1</th><th>Held-out originals</th><th>Best epoch</th></tr></thead><tbody>'
            for fold in item["folds"]:
                i = int(fold["fold"])
                best = str(item["best_epochs"][i - 1]) if item["best_epochs"] else "—"
                detail += (f'<tr><td>{i}</td><td class="num">{f3(fold["accuracy"])}</td>'
                           f'<td class="num">{f3(fold["macro_f1"])}</td>'
                           f'<td class="num">{int(fold["validation_originals"])}</td>'
                           f'<td class="num">{best}</td></tr>')
            detail += '</tbody></table></div></section>'
            detail += '<section class="panel"><h2>Confusion matrix</h2>'
            detail += '<p>Rows are actual specialities; columns are predicted specialities. Counts aggregate all five folds.</p>'
            detail += '<div class="tablewrap"><table><thead><tr><th>Actual / Predicted</th>'
            detail += "".join(f"<th>{esc(label)}</th>" for label in LABELS) + '</tr></thead><tbody>'
            matrix = {r["true_label"]: r for r in item["confusion"]}
            for label in LABELS:
                line = matrix[label]
                numbers = [int(line[col]) for col in LABELS]
                maximum = sum(numbers)
                detail += f'<tr><td>{esc(label)}</td>'
                for number in numbers:
                    frac = number / maximum if maximum else 0
                    detail += f'<td class="heat" style="background:rgba(45,132,164,{frac*0.58:.3f})">{number}</td>'
                detail += '</tr>'
            detail += '</tbody></table></div></section>'
            detail += f'<p>Original outputs: <span class="mono">{esc(group["folder"])}</span></p>'
            write_html(aug_output / "Details" / key / (item["method"] + ".html"),
                       f"{AUGMENTATION_NAMES[item['method']]} · {title}", detail, 3)
        body = (f'<p>{esc(title)}. All metrics are pooled out-of-fold results on original labelled records. '
                'Each augmentation uses the same five outer fold assignments within this configuration.</p>'
                '<section class="panel"><h2>Augmentation comparison</h2>'
                + augmentation_table(methods, link_prefix="Details/" + key) + '</section>')
        body += ('<section class="panel"><h2>Per-speciality F1</h2><div class="tablewrap">'
                 '<table><thead><tr><th>Speciality</th>'
                 + "".join(f'<th>{esc(AUGMENTATION_NAMES[m["method"]])}</th>' for m in methods)
                 + '</tr></thead><tbody>')
        class_dicts = [{r["specialty"]: r for r in method["classes"]} for method in methods]
        for label in LABELS:
            body += '<tr><td>' + esc(label) + '</td>' + ''.join(
                f'<td class="num">{f3(d[label]["f1"])}</td>' for d in class_dicts) + '</tr>'
        body += '</tbody></table></div></section>'
        if config.get("quarantine_suspicious"):
            body += info_box("This experiment excluded suspicious original records, so its scores are not directly comparable to the main dashboard's unquarantined experiments.", True)
        write_html(aug_output / (key + ".html"), title, body, 1)
        links.append((title, key + ".html", len(methods)))
    columns = ["model", "class_weighting", "fields", "prefix_removed", "augmentation", "n",
               "accuracy", "macro_f1", "weighted_f1", "minimum_class_f1", "mean_best_epoch"]
    write_csv(aug_output / "summary.csv", flat_summary, columns)
    panel = (f'<div class="cards"><div class="card"><small>Completed augmentation results</small>'
             f'<div class="number">{len(flat_summary)}</div></div>'
             f'<div class="card"><small>Model/input configurations</small>'
             f'<div class="number">{len(groups)}</div></div></div>'
             '<p>Performance uses only held-out original records. Augmented records are used only for training. '
             'Do not compare directly with main CV scores if datasets, exclusions or fold assignments differ.</p>'
             '<p><a class="btn" href="summary.csv">Download summary CSV</a></p>'
             '<h2>Model and input comparisons</h2><div class="links">')
    panel += "".join(f'<a class="linkcard" href="{esc(path)}"><strong>{esc(label)}</strong>'
                     f'<small>{count} augmentation methods · Detailed tables</small></a>'
                     for label, path, count in links)
    panel += '</div>'
    if warnings:
        panel += '<h2>Incomplete entries</h2>' + ''.join(info_box(message, True) for message in warnings)
    write_html(aug_output / "index.html", "Augmentation experiment dashboard", panel, 1)
    return len(flat_summary), warnings

def generate(runs_dir: Path, output: Path):
    runs, warnings = collect_all(runs_dir)
    if output.exists():
        # Replace only the generated results. Never delete the runs folder.
        shutil.rmtree(output)
    output.mkdir(parents=True)
    by_config, by_model, by_weighting_pair = defaultdict(list), defaultdict(list), defaultdict(list)
    metric_links = []
    for entry in runs:
        by_config[entry["configuration"]].append(entry)
        by_model[entry["model"]].append(entry)
        by_weighting_pair[f'{entry["configuration"]}--{entry["model"]}'].append(entry)
        metric_links.append(create_metrics(output, entry))
    by_config_links = create_comparisons(output, "Models_Comparison", by_config,
                                         lambda item: f'{title_model(item["model"])} · {weighting_label(item)}', title_config)
    by_model_links = create_comparisons(output, "TrainingSet_Comparison", by_model,
                                        lambda item: f'{title_config(item["configuration"])} · {weighting_label(item)}', title_model)
    paired = {key: items for key, items in by_weighting_pair.items()
              if {item["weighting"] for item in items} >= {"none", "sqrt_inv"}}
    pair_links = create_comparisons(
        output, "Weighting_Comparison", paired,
        lambda item: weighting_label(item),
        lambda key: f'{title_config(key.split("--", 1)[0])} · {title_model(key.split("--", 1)[1])}',
    )
    augmentation_count, augmentation_warnings = create_augmentation_dashboard(
        output, runs_dir.parent / "reports" / "augmentation_cv")
    panels = (('<div class="cards">'
              f'<div class="card"><small>Completed experiments</small><div class="number">{len(runs)}</div><small>Every experiment has 5 outer folds</small></div>'
              f'<div class="card"><small>Input configurations</small><div class="number">{len(by_config)}</div></div>'
              f'<div class="card"><small>Models</small><div class="number">{len(by_model)}</div></div>'
              '</div>'))
    panels += '<p>Results represent five independent held-out folds per experiment, with separate inner validation for early stopping. ± denotes fold-to-fold sample standard deviation, not a confidence interval.</p>'
    if augmentation_count:
        panels += '<h2>Data augmentation experiments</h2>'
        panels += ('<div class="links"><a class="linkcard" href="Augmentation/index.html">'
                   '<strong>RWD, LLM and combined augmentation</strong>'
                   f'<small>{augmentation_count} completed method result(s) · View HTML report</small>'
                   '</a></div>')
    for heading, links in (("Normal vs weighted (same model and inputs)", pair_links),
                           ("Models comparison", by_config_links),
                           ("Training set comparison", by_model_links),
                           ("Per-class metrics", metric_links)):
        panels += f'<h2>{esc(heading)}</h2>'
        if links:
            panels += '<div class="links">' + "".join(
                f'<a class="linkcard" href="{esc(href)}"><strong>{esc(title)}</strong>'
                f'<small>{count} completed {"experiments" if count!=1 else "experiment"} · View HTML report</small></a>'
                for title, href, count in links) + '</div>'
        else:
            panels += '<p>No complete cross-validation data available for this section.</p>'
    warnings.extend(augmentation_warnings)
    if warnings:
        panels += '<h2>Warnings</h2>' + "".join(info_box(w, True) for w in warnings)
    if not runs:
        panels += info_box("No completed main-model cross-validation runs found. Augmentation results, if available, are shown above.")
    write_html(output / "index.html", "Model evaluation dashboard", panels, 0)
    # Remove same-page link from the index header, optional cosmetic only.
    (output / "index.html").write_text((output / "index.html").read_text(encoding="utf-8").replace(
        '<a class="btn" href="index.html">← Dashboard</a>', ''), encoding="utf-8")
    normal_count = sum(item["weighting"] == "none" for item in runs)
    weighted_count = len(runs) - normal_count
    print(f"Generated {output / 'index.html'}: {normal_count} normal + {weighted_count} weighted complete CV experiment(s)")
    for warning in warnings:
        print("WARNING:", warning)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--output-dir", default="Results")
    args = p.parse_args()
    runs = Path(args.runs_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output == runs or runs in output.parents or output in runs.parents:
        raise ValueError("Results and runs directories must be separate (not nested)")
    generate(runs, output)


if __name__ == "__main__":
    main()
