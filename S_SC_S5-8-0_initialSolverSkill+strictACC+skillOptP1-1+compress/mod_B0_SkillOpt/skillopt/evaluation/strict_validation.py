from __future__ import annotations


FORWARD_PASS_RATE = "forward_pass_rate"
REAL_ACC = "real_acc"
DEFAULT_STRICT_VALIDATION_METRIC = (
    FORWARD_PASS_RATE
)

_ALLOWED = {
    FORWARD_PASS_RATE,
    REAL_ACC,
}

_SELECTION_PROTOCOLS = {
    FORWARD_PASS_RATE: (
        "s5_student_direct_first_forward_pass_rate_v1"
    ),
    REAL_ACC: "s5_student_direct_strict_acc_v1",
}

_EVALUATION_FLOWS = {
    FORWARD_PASS_RATE: (
        "student_direct_first_forward_only"
    ),
    REAL_ACC: "student_direct_only",
}


def normalize_strict_validation_metric(
    value,
) -> str:
    metric = str(
        value
        if value is not None
        else DEFAULT_STRICT_VALIDATION_METRIC
    ).strip().lower()

    if metric not in _ALLOWED:
        raise ValueError(
            "evaluation.strict_validation_metric must be "
            "'forward_pass_rate' or 'real_acc'."
        )

    return metric


def selection_eval_protocol(
    metric,
) -> str:
    return _SELECTION_PROTOCOLS[
        normalize_strict_validation_metric(metric)
    ]


def evaluation_flow(
    metric,
) -> str:
    return _EVALUATION_FLOWS[
        normalize_strict_validation_metric(metric)
    ]
