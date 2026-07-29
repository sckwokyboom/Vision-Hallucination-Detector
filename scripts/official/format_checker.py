import argparse
import json
import math
import sys
from pathlib import Path


VALID_LABELS = {"invention", "mischaracterization", "OCR", "miscounting", "other"}
VALID_LANGUAGES = {"en", "fr", "it", "zh"}
REQUIRED_ROW_KEYS = {"id", "labels"}
REQUIRED_SPAN_KEYS = {"start", "end", "prob", "label"}


def check_submission_files(submission_files, reference_files=(), reference_dirs=()):
    errors, warnings = [], []
    stats = {"files": 0, "rows": 0, "spans": 0, "languages": set()}
    submission_files = [Path(path) for path in submission_files]

    if not submission_files:
        errors.append("Submit at least one .jsonl prediction file.")
        return errors, warnings, stats
    if len(submission_files) > len(VALID_LANGUAGES):
        errors.append(f"Submit at most one file per language; got {len(submission_files)} files.")

    # References are optional. When present, they let us check exact id coverage
    # and span bounds even if participant predictions only contain id+labels.
    references = _load_references(reference_files, reference_dirs, errors)
    if not references:
        warnings.append(
            "No reference files provided; id coverage and response-length checks "
            "will be skipped unless a submission row contains response."
        )

    seen_languages, seen_splits = {}, set()
    for path in submission_files:
        file_languages, file_splits = _check_submission_file(path, references, errors, stats)
        seen_splits.update(file_splits)

        # The platform scores one language at a time, so a file should not mix languages.
        if len(file_languages) > 1:
            _err(errors, path, None, f"Each file must contain exactly one language; found {sorted(file_languages)}.")
        elif file_languages:
            language = next(iter(file_languages))
            stats["languages"].add(language)
            if language in seen_languages:
                _err(errors, path, None, f"At most one file per language is allowed; {language} also appears in {seen_languages[language]}.")
            seen_languages.setdefault(language, path)

        if len(file_splits) > 1:
            _err(errors, path, None, f"Each file must contain ids from exactly one split; found {sorted(file_splits)}.")

    if len(seen_splits) > 1:
        errors.append(f"All submitted files should correspond to the same split; found {sorted(seen_splits)}.")
    return errors, warnings, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check SHROOM Vision submission JSONL files before uploading them.")
    parser.add_argument("submission_files", nargs="+", type=Path, help="Submission .jsonl files; use at most one per language.")
    parser.add_argument("--reference-file", action="append", default=[], type=Path, help="Reference/test JSONL file. Can be passed multiple times.")
    parser.add_argument("--reference-dir", action="append", default=[], type=Path, help="Directory containing reference/test JSONL files.")
    parser.add_argument("--max-errors", default=40, type=int, help="Maximum errors to print; 0 prints all.")
    args = parser.parse_args(argv)

    errors, warnings, stats = check_submission_files(args.submission_files, args.reference_file, args.reference_dir)
    _print_report(errors, warnings, stats, args.max_errors)
    return 1 if errors else 0


def _check_submission_file(path, references, errors, stats):
    stats["files"] += 1
    file_ids, file_languages, file_splits = {}, set(), set()

    if path.suffix != ".jsonl":
        _err(errors, path, None, "Submission files must use the .jsonl extension.")

    rows = _read_jsonl(path, errors)
    if not rows:
        _err(errors, path, None, "Submission file contains no predictions.")
        return file_languages, file_splits

    for line_no, row in rows:
        stats["rows"] += 1
        parsed_id = _check_row(row, path, line_no, errors)
        if parsed_id:
            split, language, item_id = parsed_id
            file_splits.add(split)
            file_languages.add(language)
            if item_id in file_ids:
                _err(errors, path, line_no, f"Duplicate prediction id {item_id!r}; first seen on line {file_ids[item_id]}.")
            file_ids.setdefault(item_id, line_no)

        # Empty labels are valid: that means "no hallucination spans predicted".
        labels = row.get("labels")
        if isinstance(labels, list):
            text_len = _text_len(row, parsed_id, references)
            for index, span in enumerate(labels):
                stats["spans"] += 1
                _check_span(span, index, text_len, row.get("id"), path, line_no, errors)

    if references and len(file_languages) == 1 and len(file_splits) == 1:
        _check_reference_ids(path, next(iter(file_splits)), next(iter(file_languages)), set(file_ids), references, errors)

    return file_languages, file_splits


def _load_references(reference_files, reference_dirs, errors):
    paths = [Path(path) for path in reference_files]
    for directory in map(Path, reference_dirs):
        if not directory.exists():
            _err(errors, directory, None, "Reference directory does not exist.")
            continue
        if not directory.is_dir():
            _err(errors, directory, None, "Reference path is not a directory.")
            continue
        test_files = [p for p in sorted(directory.glob("shroom-vision.test.*.jsonl")) if ".all_metadata." not in p.name]
        paths.extend(test_files or sorted(directory.glob("*.jsonl")))

    references = {}
    for path in paths:
        if path.suffix != ".jsonl":
            _err(errors, path, None, "Reference files must use the .jsonl extension.")
        for line_no, row in _read_jsonl(path, errors):
            parsed_id = _parse_id(row.get("id"), path, line_no, errors)
            text_len = _reference_len(row, path, line_no, errors)
            if not parsed_id or text_len is None:
                continue
            split, language, item_id = parsed_id
            language_refs = references.setdefault((split, language), {})
            if item_id in language_refs:
                _err(errors, path, line_no, f"Duplicate reference id {item_id!r}.")
            language_refs.setdefault(item_id, text_len)
    return references


def _read_jsonl(path, errors):
    if not path.exists():
        _err(errors, path, None, "File does not exist.")
        return []
    if not path.is_file():
        _err(errors, path, None, "Path is not a file.")
        return []

    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    _err(errors, path, line_no, "Empty lines are not valid JSONL records.")
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    _err(errors, path, line_no, f"Invalid JSON: {exc.msg}.")
                    continue
                if isinstance(row, dict):
                    rows.append((line_no, row))
                else:
                    _err(errors, path, line_no, "Each JSONL line must be a JSON object.")
    except OSError as exc:
        _err(errors, path, None, f"Could not read file: {exc}.")
    return rows


def _check_row(row, path, line_no, errors):
    missing = REQUIRED_ROW_KEYS - set(row)
    if "labels" in missing and "predictions" in row:
        _err(errors, path, line_no, "Found predictions, but the scorer expects labels. Rename/convert the baseline output before submitting.")
    if missing:
        _err(errors, path, line_no, f"Missing required key(s): {', '.join(sorted(missing))}.")
    if "labels" in row and not isinstance(row["labels"], list):
        _err(errors, path, line_no, "labels must be a list; use [] when there are no spans.")
    return _parse_id(row.get("id"), path, line_no, errors)


def _parse_id(value, path, line_no, errors):
    if not isinstance(value, str):
        _err(errors, path, line_no, "id must be a string.")
        return None
    parts = value.split("-")
    if len(parts) != 3 or not all(parts):
        _err(errors, path, line_no, "id must look like <split>-<language>-<item>, for example test-en-42.")
        return None
    split, language, _ = parts
    split, language = split.lower(), language.lower()
    if language not in VALID_LANGUAGES:
        _err(errors, path, line_no, f"Unsupported language {language!r}; expected one of {sorted(VALID_LANGUAGES)}.")
        return None
    return split, language, value


def _reference_len(row, path, line_no, errors):
    if isinstance(row.get("response"), str):
        return len(row["response"])
    if type(row.get("text_len")) is int and row["text_len"] >= 0:
        return row["text_len"]
    _err(errors, path, line_no, "Reference rows need a string response or a non-negative integer text_len.")
    return None


def _text_len(row, parsed_id, references):
    if parsed_id:
        split, language, item_id = parsed_id
        text_len = references.get((split, language), {}).get(item_id)
        if text_len is not None:
            return text_len
    return len(row["response"]) if isinstance(row.get("response"), str) else None


def _check_span(span, index, text_len, item_id, path, line_no, errors):
    name = f"labels[{index}]"
    if not isinstance(span, dict):
        _err(errors, path, line_no, f"{name} must be an object.")
        return

    missing = REQUIRED_SPAN_KEYS - set(span)
    if missing:
        _err(errors, path, line_no, f"{name} is missing key(s): {', '.join(sorted(missing))}.")

    start, end = span.get("start"), span.get("end")
    prob, label = span.get("prob"), span.get("label")
    if type(start) is not int:
        _err(errors, path, line_no, f"{name}.start must be an int.")
    if type(end) is not int:
        _err(errors, path, line_no, f"{name}.end must be an int.")
    if type(prob) is not float or not math.isfinite(prob):
        _err(errors, path, line_no, f"{name}.prob must be a finite float, for example 1.0.")
    if label not in VALID_LABELS:
        _err(errors, path, line_no, f"{name}.label must be one of {sorted(VALID_LABELS)}; got {label!r}.")
    if type(start) is int and type(end) is int:
        if not 0 <= start < end:
            _err(errors, path, line_no, f"{name} must satisfy 0 <= start < end; got {start}, {end}.")
        if text_len is not None and end > text_len:
            _err(errors, path, line_no, f"{name}.end={end} exceeds response length {text_len} for {item_id!r}.")


def _check_reference_ids(path, split, language, prediction_ids, references, errors):
    reference_ids = set(references.get((split, language), {}))
    if not reference_ids:
        _err(errors, path, None, f"No reference ids found for split={split!r}, language={language!r}.")
        return

    missing = sorted(reference_ids - prediction_ids)
    extra = sorted(prediction_ids - reference_ids)
    if missing:
        _err(errors, path, None, f"Missing {len(missing)} prediction id(s): {_preview(missing)}.")
    if extra:
        _err(errors, path, None, f"Found {len(extra)} id(s) not present in references: {_preview(extra)}.")


def _preview(values, limit=10):
    shown = ", ".join(values[:limit])
    return shown + (f", ... (+{len(values) - limit} more)" if len(values) > limit else "")


def _err(errors, path, line_no, message):
    location = path if line_no is None else f"{path}:{line_no}"
    errors.append(f"{location}: {message}")


def _print_report(errors, warnings, stats, max_errors):
    print(f"Checked {stats['files']} file(s), {stats['rows']} row(s), {stats['spans']} span(s).")
    if stats["languages"]:
        print("Languages: " + ", ".join(sorted(stats["languages"])))
    for warning in warnings:
        print("WARNING: " + warning, file=sys.stderr)
    for error in (errors if max_errors == 0 else errors[:max_errors]):
        print("ERROR: " + error, file=sys.stderr)
    if max_errors and len(errors) > max_errors:
        print(f"ERROR: ... {len(errors) - max_errors} more error(s) not shown.", file=sys.stderr)
    print(f"FAILED: found {len(errors)} error(s).", file=sys.stderr) if errors else print("OK: submission format looks valid.")


if __name__ == "__main__":
    raise SystemExit(main())
