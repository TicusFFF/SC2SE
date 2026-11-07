from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


GOLD_KEYS = {
    "answer",
    "A",
    "A_original",
    "A_helpful",
    "solution",
    "reference",
    "gold",
    "gold_answer",
    "official_solution",
}

PROBLEM_KEYS = (
    "q",
    "problem",
    "question",
    "prompt",
    "query",
)

SAMPLE_ID_KEYS = (
    "id",
    "qid",
    "uid",
    "sample_id",
)


def get_problem_text(item: Dict[str, Any]) -> str:
    for key in PROBLEM_KEYS:
        value = item.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    raise KeyError(
        f"Cannot find problem text. Expected one of {PROBLEM_KEYS}, "
        f"but got keys={list(item.keys())}"
    )


def validate_sample_id(value: Any) -> str:
    """
    Validate an id before using it as a prediction directory name.

    The id is preserved exactly. It is not silently stripped or rewritten.
    """
    if value is None:
        raise ValueError("Sample id must not be None.")

    sample_id = str(value)

    if not sample_id:
        raise ValueError("Sample id must not be empty.")

    if sample_id != sample_id.strip():
        raise ValueError(
            f"Unsafe sample id {sample_id!r}: "
            "leading or trailing whitespace is not allowed."
        )

    if sample_id in {".", ".."}:
        raise ValueError(f"Unsafe sample id: {sample_id!r}")

    if "/" in sample_id or "\\" in sample_id:
        raise ValueError(
            f"Unsafe sample id {sample_id!r}: "
            "path separators are not allowed."
        )

    if any(ord(char) < 32 for char in sample_id):
        raise ValueError(
            f"Unsafe sample id {sample_id!r}: "
            "control characters are not allowed."
        )

    return sample_id


def get_sample_id(
    item: Dict[str, Any],
    index: int | None = None,
    dataset_key: str | None = None,
) -> str:
    """
    Return a stable sample id.

    When dataset_key is provided, mirror the S3 hard-test runner and
    normalize ids to "{dataset_key}_{row_index}" so shared initial
    responses can be matched directly.

    Do not use:

        item.get("id") or ...

    because integer id 0 is valid but evaluates to False.
    """
    if dataset_key:
        if index is None:
            raise ValueError(
                "index is required when dataset_key is provided."
            )

        return validate_sample_id(
            f"{dataset_key}_{index}"
        )

    for key in SAMPLE_ID_KEYS:
        if key not in item:
            continue

        value = item[key]

        if value is None:
            continue

        sample_id = str(value)

        if sample_id:
            return validate_sample_id(sample_id)

    if index is not None:
        return validate_sample_id(
            f"sample_{index:06d}"
        )

    raise KeyError("Cannot determine sample id.")


def sanitize_item(
    item: Dict[str, Any],
    index: int | None = None,
    dataset_key: str | None = None,
) -> Dict[str, Any]:
    """
    Keep only blind problem information.

    Gold fields must not enter:
      - A0
      - A8
      - A11/A12/A13/A14/A15/A16/A17
      - SkillOpt rollout evidence
      - SkillOpt optimizer
      - SkillOpt validation gate
    """
    if not isinstance(item, dict):
        raise TypeError(
            "Dataset item must be a JSON object, "
            f"got {type(item).__name__}."
        )

    sample_id = get_sample_id(
        item,
        index,
        dataset_key=dataset_key,
    )
    problem = get_problem_text(item)

    return {
        "id": sample_id,
        "q": problem,
        "problem": problem,
        "query": problem,
    }


def has_gold_fields(item: Dict[str, Any]) -> bool:
    return bool(
        GOLD_KEYS.intersection(item.keys())
    )


def gold_fields_in(item: Dict[str, Any]) -> list[str]:
    return sorted(
        GOLD_KEYS.intersection(item.keys())
    )


def assert_blind_item(item: Dict[str, Any]) -> None:
    if not isinstance(item, dict):
        raise TypeError(
            "S5 blind item must be a dict, "
            f"got {type(item).__name__}."
        )

    leaked = gold_fields_in(item)

    if leaked:
        raise ValueError(
            f"Gold fields leaked into S5 blind item: {leaked}"
        )

    if "id" not in item:
        raise KeyError("S5 blind item is missing id.")

    validate_sample_id(item["id"])


def assert_unique_sample_ids(
    items: List[Dict[str, Any]],
) -> None:
    seen: dict[str, int] = {}

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(
                f"Dataset row {index} must be a JSON object."
            )

        if "id" not in item:
            raise KeyError(
                f"Dataset row {index} is missing id."
            )

        sample_id = validate_sample_id(item["id"])

        if sample_id in seen:
            raise ValueError(
                f"Duplicate sample id {sample_id!r} "
                f"at rows {seen[sample_id]} and {index}."
            )

        seen[sample_id] = index


def _finalize_items(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    assert_unique_sample_ids(items)
    return items


def _sanitize_rows(
    rows: list[Any],
    *,
    dataset_key: str | None = None,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(
                f"Dataset row {index} must be a JSON object, "
                f"got {type(row).__name__}."
            )

        items.append(
            sanitize_item(
                row,
                index,
                dataset_key=dataset_key,
            )
        )

    return _finalize_items(items)


def load_json_or_jsonl(
    path: str | Path,
    *,
    dataset_key: str | None = None,
) -> List[Dict[str, Any]]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset does not exist: {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        items: List[Dict[str, Any]] = []

        with path.open("r", encoding="utf-8-sig") as file:
            for line_index, line in enumerate(file):
                line = line.strip()

                if not line:
                    continue

                item = json.loads(line)

                if not isinstance(item, dict):
                    raise TypeError(
                        f"JSONL row {line_index + 1} must be an object."
                    )

                items.append(
                    sanitize_item(
                        item,
                        line_index,
                        dataset_key=dataset_key,
                    )
                )

        return _finalize_items(items)

    if suffix == ".json":
        data = json.loads(
            path.read_text(encoding="utf-8-sig")
        )

        if isinstance(data, list):
            return _sanitize_rows(
                data,
                dataset_key=dataset_key,
            )

        if isinstance(data, dict):
            for key in (
                "data",
                "items",
                "examples",
                "records",
                "logs",
            ):
                rows = data.get(key)

                if isinstance(rows, list):
                    return _sanitize_rows(
                        rows,
                        dataset_key=dataset_key,
                    )

            return _finalize_items(
                [
                    sanitize_item(
                        data,
                        0,
                        dataset_key=dataset_key,
                    )
                ]
            )

    raise ValueError(
        f"Unsupported dataset file: {path}. "
        "Expected a .json or .jsonl file."
    )


def split_same_pool(
    items: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    First S5 version:

    Keep SkillOpt's train / validation / test interface,
    but use the same blind pool for all split aliases.

    This closes the first S5 optimization loop. It is not a held-out
    generalization evaluation.
    """
    assert_unique_sample_ids(items)

    return {
        "train": list(items),
        "valid_seen": list(items),
        "valid_unseen": list(items),
        "selection": list(items),
        "val": list(items),
        "test": list(items),
    }
