"""Role-consistency validation and targeted repair for candidate Skills."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable
from typing import Any

from skillopt.model import chat_optimizer
from skillopt.optimizer.skill import apply_patch_with_report
from skillopt.optimizer.structured_skills import (
    ROLE_FIELDS,
    StructuredSkillError,
    parse_atomic_skill_draft,
)
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json


_ROLE_PROFILE_PLACEHOLDER = "{role_profile}"

_CHECK_KEYS = {
    "passed",
    "issues",
}

_ISSUE_KEYS = {
    "quoted_text",
    "problem",
}

_REPAIR_KEYS = {
    "repairs",
}

_REPAIR_ITEM_KEYS = {
    "issue_index",
    "replacement_text",
}

_LEGACY_REPAIR_ITEM_KEYS = {
    "quoted_text",
    "replacement_text",
}

_EXPLICITLY_ALLOWED_ROLE_ISSUE_QUOTES = frozenset({
    "consider numerical verification",
})

_EXPLICIT_ROLE_CONTAMINATION_RE = re.compile(
    r"(?:"
    r"\bForward\b|\bPythonTool\b|"
    r"\b(?:validator|grader|evaluator|tool agent)\b|"
    r"\b(?:reference answer|ground truth|hidden state|system prompt)\b|"
    r"\binternal (?:tool|workflow|record|state|component)\b|"
    r"(?:subsequent|later|downstream).{0,32}"
    r"(?:validation|verdict|feedback|revision)|"
    r"验证器|纠错代理|评测器|参考答案|隐藏状态|系统提示词|"
    r"内部(?:工具|流程|记录|状态|组件)|反馈记录|修订记录|"
    r"后续.{0,16}(?:验证|判定|反馈|修订)|"
    r"\?\s*(?:No|Wait|Wrong|Incorrect)\b|"
    r"(?:^|\n)[ \t]*(?:[-*+][ \t]+)?"
    r"(?:Wait(?:[ \t]*[,!?…]|[ \t]+(?:maybe|perhaps|try)\b)|"
    r"Actually,[ \t]+no\b)|"
    r"Let's re-evaluate\b|"
    r"？\s*(?:不对|等等|也许|再试试)|"
    r"(?:^|\n)[ \t]*(?:[-*+][ \t]+)?"
    r"(?:不对|等等|再试试)[，。！？…]"
    r")",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)

_ROLE_DECISION_KEYS = {
    "decisions",
}

_ROLE_DECISION_ITEM_KEYS = {
    "anchor_id",
    "is_problem",
}

_ROLE_REWRITE_KEYS = {
    "repairs",
}

_ROLE_REWRITE_ITEM_KEYS = {
    "unit_id",
    "decision",
    "replacement_text",
}

_ROLE_REWRITE_DECISIONS = {
    "replace",
    "no_commit",
}

STRUCTURED_ATOMIC_SKILL_FIELDS = tuple(
    ROLE_FIELDS
)

_STRUCTURED_ROLE_REPAIR_KEYS = {
    "decision",
    "replacement_skill",
}

_STRUCTURED_ROLE_REPAIR_DECISIONS = {
    "replace",
    "no_commit",
}

_ROLE_POSTCHECK_BOOLEAN_KEYS = (
    "role_clean",
    "settled_claims_preserved",
    "polarity_preserved",
    "conditions_preserved",
    "no_status_promotion",
    "no_new_math",
    "markdown_coherent",
)

_ROLE_POSTCHECK_KEYS = {
    "unit_reviews",
    "candidate_role_clean",
    "candidate_violations",
}

_ROLE_POSTCHECK_ITEM_KEYS = {
    "unit_id",
    "violations",
} | set(_ROLE_POSTCHECK_BOOLEAN_KEYS)

_ROLE_IMMUTABLE_TOKEN_RE = re.compile(
    r"⟦ROLE_IMMUTABLE_\d+_U\d{4}_\d{4}⟧"
)

_ROLE_IMMUTABLE_TOKEN_BROAD_RE = re.compile(
    r"⟦ROLE_IMMUTABLE_[^⟧]*⟧"
)

_ROLE_PROCESS_TRACE_RE = re.compile(
    r"(?:"
    r"\brejected attempt\b|"
    r"\bfinal revision\b|"
    r"\b(?:analysis |reasoning )?draft\b|"
    r"\binternal analysis\b|"
    r"\btool output\b|"
    r"\bevidence answer\b|"
    r"已否定的候选|最终修订|分析草稿|工具输出|证据答案"
    r")",
    re.IGNORECASE,
)

_COGNITIVE_REJECTED_ANCHOR_KINDS = frozenset({
    "question_rejection",
    "question_rejection_zh",
})

_COGNITIVE_UNRESOLVED_ANCHOR_KINDS = frozenset({
    "question_reconsider",
    "question_check",
})

_COGNITIVE_FAIL_CLOSED_ANCHOR_KINDS = frozenset({
    "actually_no",
    "standalone_no",
    "look_at_evidence",
    "look_at_answer",
    "answer_label",
    "trust_pattern",
    "re_evaluate",
    "process_history_label",
    "wait_transition_zh",
    "try_alternative_zh",
})

_COGNITIVE_STATUS_BOUNDARY_RE = re.compile(
    r"(?:"
    r"[.?!;。！？；\n]+|"
    r"(?<!\w)(?:but|however|whereas|yet)(?!\w)|"
    r"但是|然而|不过|却"
    r")",
    re.IGNORECASE,
)

_COGNITIVE_UNRESOLVED_CUE_RE = re.compile(
    r"(?:"
    r"\b(?:may|might|could|perhaps|possibly|uncertain|"
    r"unresolved|unconfirmed|candidate)\b|"
    r"\b(?:requires?|needs?)\b[^.?!;\n]{0,32}"
    r"\b(?:proof|derivation|verification|checking)\b|"
    r"\b(?:is|are|remains?)\b[^.?!;\n]{0,16}"
    r"\bnot\b[^.?!;\n]{0,16}"
    r"\b(?:established|confirmed|determined)\b|"
    r"\b(?:cannot|should\s+not)\s+be\s+"
    r"(?:assumed|treated\s+as\s+established)\b|"
    r"可能|也许|尚未(?:确定|证实|证明)|待(?:验证|证明|推导)|"
    r"不能假定|不应假定"
    r")",
    re.IGNORECASE,
)

_COGNITIVE_REJECTED_CUE_RE = re.compile(
    r"(?:"
    r"\bavoid\b|"
    r"\b(?:is|are|was|were|be|remains?)\b"
    r"(?:[ \t]+(?:also|clearly|plainly|definitely|"
    r"mathematically|simply|indeed))?[ \t]+"
    r"(?:incorrect|wrong|false|invalid|"
    r"inapplicable|unsuitable)\b|"
    r"\b(?:is|are|was|were|be|does|do|can|cannot|should|must)\b"
    r"[^.?!;\n]{0,12}\bnot\b[^.?!;\n]{0,12}"
    r"\b(?:correct|valid|applicable|suitable|usable|direct|"
    r"established|the\s+correct)\b|"
    r"\b(?:do|does|should|must)\s+not\s+"
    r"(?:use|apply|assume|adopt|take|treat)\b|"
    r"\bcannot\s+be\s+(?:used|applied|assumed|adopted|treated)\b|"
    r"\bcannot\s+serve\b|"
    r"\bshould\s+be\s+(?:discarded|rejected|excluded)\b|"
    r"\b(?:fails?|does\s+not\s+(?:hold|work|apply|yield))\b|"
    r"不正确|错误|不成立|不适用|不可用|不能采用|不要使用|避免"
    r")",
    re.IGNORECASE,
)

_COGNITIVE_DOUBLE_NEGATION_RE = re.compile(
    r"(?:"
    r"\bnot(?:\s+[A-Za-z]+){0,3}\s+(?:incorrect|wrong|false|invalid|"
    r"inapplicable|unsuitable)\b|"
    r"\b(?:hardly|scarcely|rarely|barely)\b[^.?!;\n]{0,16}"
    r"\b(?:incorrect|wrong|false|invalid|inapplicable|unsuitable)\b|"
    r"并非(?:不正确|错误|不成立|不适用|不可用)"
    r")",
    re.IGNORECASE,
)

_COGNITIVE_PREFIX_REJECTION_RE = re.compile(
    r"^(?:avoid|(?:do|does|should|must)\s+not\s+"
    r"(?:use|apply|assume|adopt|take|treat)|避免|不要使用)$",
    re.IGNORECASE,
)

_COGNITIVE_PREFIX_UNRESOLVED_RE = re.compile(
    r"^(?:perhaps|possibly|maybe|也许|可能)$",
    re.IGNORECASE,
)

_COGNITIVE_COORDINATOR_RE = re.compile(
    r"^[ \t]*(?:(?:,[ \t]*)?(?:and|or|以及|和|或)[ \t]*|,[ \t]*)$",
    re.IGNORECASE,
)

_PROTECTED_LOGIC_RE = re.compile(
    r"(?:"
    r"\b(?:not|only|if|unless|except|all|any|may|might|must|"
    r"never|always|exactly|correct|incorrect|false|true)\b|"
    r"\bat[ \t]+(?:most|least)\b|"
    r"\bno[ \t]+more[ \t]+than\b|"
    r"\bno[ \t]+less[ \t]+than\b|"
    r"不成立|成立|正确|错误|仅当|当且仅当|至多|至少|"
    r"除非|如果|必须|可能|总是|从不"
    r")",
    re.IGNORECASE,
)

_ROLE_ANCHOR_RULES = (
    {
        "kind": "question_rejection",
        "pattern": re.compile(
            r"\?\s*(?:No|Wrong|Incorrect)\b[,.]?",
            re.IGNORECASE,
        ),
        "forced_problem": True,
    },
    {
        "kind": "question_reconsider",
        "pattern": re.compile(
            r"\?\s*Actually,",
            re.IGNORECASE,
        ),
        "forced_problem": True,
    },
    {
        "kind": "question_check",
        "pattern": re.compile(
            r"\?\s*Let's check:",
            re.IGNORECASE,
        ),
        "forced_problem": True,
    },
    {
        "kind": "wait_transition",
        "pattern": re.compile(
            r"\bWait,(?=[ \t]+"
            r"(?:in|the|this|that|if|but|we|maybe|perhaps|try)\b)",
            re.IGNORECASE,
        ),
        "forced_problem": True,
    },
    {
        "kind": "actually_no",
        "pattern": re.compile(
            r"\bActually,[ \t]+no\b",
            re.IGNORECASE,
        ),
        "forced_problem": True,
    },
    {
        "kind": "actually_transition",
        "pattern": re.compile(
            r"\bActually,",
            re.IGNORECASE,
        ),
        "forced_problem": False,
    },
    {
        "kind": "standalone_no",
        "pattern": re.compile(
            r"(?<!\w)No\.(?=[ \t]|$)",
            re.IGNORECASE | re.MULTILINE,
        ),
        "forced_problem": False,
    },
    {
        "kind": "process_history_label",
        "pattern": _ROLE_PROCESS_TRACE_RE,
        "forced_problem": False,
    },
    {
        "kind": "look_at_evidence",
        "pattern": re.compile(
            r"\bLet's look at the evidence\b:?",
            re.IGNORECASE,
        ),
        "forced_problem": True,
    },
    {
        "kind": "re_evaluate",
        "pattern": re.compile(
            r"\bLet's re-evaluate\b",
            re.IGNORECASE,
        ),
        "forced_problem": True,
    },
    {
        "kind": "look_at_answer",
        "pattern": re.compile(
            r"\bLet's look at the answer\b",
            re.IGNORECASE,
        ),
        "forced_problem": True,
    },
    {
        "kind": "answer_label",
        "pattern": re.compile(
            r"\bAnswer(?=[ \t]+(?:[-+]?(?:\d|\.\d)|\$))",
            re.IGNORECASE,
        ),
        "forced_problem": True,
    },
    {
        "kind": "trust_pattern",
        "pattern": re.compile(
            r"\bLet's trust the pattern\b:?",
            re.IGNORECASE,
        ),
        "forced_problem": True,
    },
    {
        "kind": "question_rejection_zh",
        "pattern": re.compile(
            r"？\s*(?:不对|错误)[，。]?",
        ),
        "forced_problem": True,
    },
    {
        "kind": "wait_transition_zh",
        "pattern": re.compile(
            r"\b等等[，：]",
        ),
        "forced_problem": True,
    },
    {
        "kind": "try_alternative_zh",
        "pattern": re.compile(
            r"\b再试试[：，]",
        ),
        "forced_problem": True,
    },
    {
        "kind": "external_role_or_tool",
        "pattern": re.compile(
            r"(?:"
            r"\bForward\b|"
            r"\bPythonTool\b|"
            r"\b(?:validator|grader|evaluator|tool agent)\b|"
            r"\b(?:reference answer|ground truth|hidden state|"
            r"system prompt)\b|"
            r"\binternal (?:tool|workflow|record|state|component)\b|"
            r"\b(?:subsequent|later|downstream)\b"
            r"[^\r\n]{0,32}"
            r"\b(?:validation|verdict|feedback|revision)\b|"
            r"验证器|纠错代理|评测器|参考答案|隐藏状态|"
            r"系统提示词|内部(?:工具|流程|记录|状态|组件)|"
            r"反馈记录|修订记录|"
            r"后续[^\r\n]{0,16}(?:验证|判定|反馈|修订)"
            r")",
            re.IGNORECASE,
        ),
        "forced_problem": False,
    },
)

_PROTECTED_MATH_OR_NUMBER_RE = re.compile(
    r"(?:"
    r"\\begin\{(?P<math_env>[A-Za-z*]+)\}.*?"
    r"\\end\{(?P=math_env)\}|"
    r"(?<!\\)\$\$.*?(?<!\\)\$\$|"
    r"\\\[.*?\\\]|"
    r"\\\(.*?\\\)|"
    r"(?<!\\)\$(?:\\.|[^$\n])*?(?<!\\)\$|"
    r"(?<!\d)[-+]?(?:\d+\.\d+|\d+|\.\d+)(?!\d)"
    r")",
    re.DOTALL,
)

_ROLE_TRANSPORT_TOKEN_RE = re.compile(
    r"⟦ROLE_(?:BACKSLASH|LINE_FEED)_\d+⟧"
)

_SLOW_UPDATE_START = "<!-- SLOW_UPDATE_START -->"
_SLOW_UPDATE_END = "<!-- SLOW_UPDATE_END -->"

_SUBSKILL_METADATA_KEYS = (
    "schema_version",
    "subskill_id",
    "source_edit_ids",
    "evidence_ids",
    "evidence_count",
    "support_count",
    "scope",
    "source_type",
)

# The English text is the single semantic source of truth.
# Each variant is a fixed, manually reviewed, parallel
# rendering. The versions differ only in wording.
_FORMAT_RETRY_MASTER_EN = (
    "## Output Format Requirements\n\n"
    "The task and the complete input are exactly the same as "
    "stated above.\n\n"
    "Output only one valid bare JSON object.\n"
    "Do not use a Markdown code block.\n"
    "Do not output any explanation, prefix, or suffix.\n"
    "Use only the specified fields and do not add other fields.\n"
    "All field names and strings must use valid JSON syntax "
    "with double quotation marks.\n"
    "If any transport placeholder is provided above, including "
    "ROLE_IMMUTABLE, follow its encoding rules exactly; use "
    "standard JSON backslash escaping only when no corresponding "
    "placeholder is provided.\n"
    "Every literal backslash inside a JSON string must be "
    "escaped as two backslashes; for example, represent the "
    "LaTeX text \\sin as \"\\\\sin\".\n\n"
    "The required top-level structure is:\n"
    "{expected_schema}"
)

_FORMAT_RETRY_VARIANTS_EN = (
    (
        "## Equivalent English Rendering\n\n"
        "The task and the complete input are exactly the same as "
        "stated above.\n\n"
        "Output only one valid bare JSON object.\n"
        "Do not use a Markdown code block.\n"
        "Do not output any explanation, prefix, or suffix.\n"
        "Use only the specified fields and do not add other fields.\n"
        "All field names and strings must use valid JSON syntax "
        "with double quotation marks.\n"
        "If any transport placeholder is provided above, including "
        "ROLE_IMMUTABLE, follow its encoding rules exactly; use "
        "standard JSON backslash escaping only when no corresponding "
        "placeholder is provided.\n"
        "Every literal backslash inside a JSON string must be "
        "escaped as two backslashes; for example, represent the "
        "LaTeX text \\sin as \"\\\\sin\".\n\n"
        "The required top-level structure is:\n"
        "{expected_schema}"
    ),
    (
        "## Equivalent English Rendering\n\n"
        "The task and the full input remain exactly the same as "
        "provided above.\n\n"
        "Return only one valid plain JSON object.\n"
        "Do not place it inside a Markdown code block.\n"
        "Include no explanation, prefix, or suffix.\n"
        "Include only the required fields and add no extra fields.\n"
        "Write every field name and string using valid "
        "double-quoted JSON syntax.\n"
        "When the prompt supplies any transport placeholder, including "
        "ROLE_IMMUTABLE, follow its encoding rules exactly; apply "
        "normal JSON backslash escaping only if no matching placeholder "
        "was supplied.\n"
        "Escape every literal backslash in a JSON string by "
        "doubling it; for example, write the LaTeX text \\sin "
        "as \"\\\\sin\".\n\n"
        "Use the following top-level structure:\n"
        "{expected_schema}"
    ),
    (
        "## Equivalent English Rendering\n\n"
        "The task and all input content are unchanged from what is "
        "given above.\n\n"
        "Provide only a single valid raw JSON object.\n"
        "Do not wrap it in a Markdown code block.\n"
        "Provide no explanation or text before or after the object.\n"
        "Use the prescribed fields only, without additional fields.\n"
        "Represent every field name and string with valid JSON "
        "double-quote syntax.\n"
        "For any transport placeholder given above, including "
        "ROLE_IMMUTABLE, follow its encoding rules exactly; only in "
        "the absence of such a placeholder should standard JSON "
        "backslash escaping be used.\n"
        "Double each literal backslash when it appears inside a "
        "JSON string; for example, encode the LaTeX text \\sin "
        "as \"\\\\sin\".\n\n"
        "Follow this top-level structure:\n"
        "{expected_schema}"
    ),
)


def _select_role_transport_token(
    token_name: str,
    *values: Any,
) -> str:
    occupied = "\0".join(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )
        for value in values
    )

    index = 0

    while True:
        token = (
            f"⟦ROLE_{token_name}_{index}⟧"
        )

        if token not in occupied:
            return token

        index += 1


def _select_backslash_transport_token(
    *values: Any,
) -> str:
    return _select_role_transport_token(
        "BACKSLASH",
        *values,
    )


def _select_line_feed_transport_token(
    *values: Any,
) -> str:
    return _select_role_transport_token(
        "LINE_FEED",
        *values,
    )


def _encode_backslash_transport(
    value: Any,
    *,
    token: str,
) -> Any:
    if isinstance(value, str):
        return value.replace(
            "\\",
            token,
        )

    if isinstance(value, list):
        return [
            _encode_backslash_transport(
                item,
                token=token,
            )
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: _encode_backslash_transport(
                item,
                token=token,
            )
            for key, item in value.items()
        }

    return value


def _decode_backslash_transport(
    value: Any,
    *,
    token: str,
) -> Any:
    if isinstance(value, str):
        return value.replace(
            token,
            "\\",
        )

    if isinstance(value, list):
        return [
            _decode_backslash_transport(
                item,
                token=token,
            )
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: _decode_backslash_transport(
                item,
                token=token,
            )
            for key, item in value.items()
        }

    return value


def _encode_line_feed_transport(
    value: Any,
    *,
    token: str,
) -> Any:
    if isinstance(value, str):
        return value.replace(
            "\n",
            token,
        )

    if isinstance(value, list):
        return [
            _encode_line_feed_transport(
                item,
                token=token,
            )
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: _encode_line_feed_transport(
                item,
                token=token,
            )
            for key, item in value.items()
        }

    return value


def _decode_line_feed_transport(
    value: Any,
    *,
    token: str,
) -> Any:
    if isinstance(value, str):
        return value.replace(
            token,
            "\n",
        )

    if isinstance(value, list):
        return [
            _decode_line_feed_transport(
                item,
                token=token,
            )
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: _decode_line_feed_transport(
                item,
                token=token,
            )
            for key, item in value.items()
        }

    return value


def _introduces_role_transport_token(
    replacement_text: str,
    *,
    source_text: str,
) -> bool:
    for token in set(
        _ROLE_TRANSPORT_TOKEN_RE.findall(
            replacement_text
        )
    ):
        if (
            replacement_text.count(token)
            > source_text.count(token)
        ):
            return True

    return False


def _is_exact_source_extract(
    replacement_text: str,
    *,
    source_text: str,
) -> bool:
    """Return whether replacement is one exact contiguous source span."""
    return (
        replacement_text == ""
        or replacement_text in source_text
    )


def _contains_explicit_role_contamination(
    text: str,
) -> bool:
    return (
        _EXPLICIT_ROLE_CONTAMINATION_RE.search(
            str(text)
        )
        is not None
    )


def _is_complete_line_span(
    candidate_skill: str,
    quoted_text: str,
) -> bool:
    start = candidate_skill.index(
        quoted_text
    )
    end = start + len(
        quoted_text
    )
    line_start = (
        candidate_skill.rfind(
            "\n",
            0,
            start,
        )
        + 1
    )
    line_end = candidate_skill.find(
        "\n",
        end,
    )

    if line_end < 0:
        line_end = len(
            candidate_skill
        )

    return (
        not candidate_skill[
            line_start:start
        ].strip(" \t")
        and not candidate_skill[
            end:line_end
        ].strip(" \t")
    )


def _protected_math_and_number_tokens(
    text: str,
) -> list[str]:
    return [
        match.group(0)
        for match in (
            _PROTECTED_MATH_OR_NUMBER_RE.finditer(
                text
            )
        )
    ]


def _repair_replacement_control_character_error(
    payload: dict,
) -> str | None:
    repairs = payload.get("repairs")

    if not isinstance(repairs, list):
        return None

    for repair in repairs:
        if not isinstance(repair, dict):
            continue

        replacement_text = repair.get(
            "replacement_text"
        )

        if not isinstance(
            replacement_text,
            str,
        ):
            continue

        if any(
            ord(character) < 0x20
            for character in replacement_text
        ):
            return (
                "Repair replacement_text must not contain "
                "decoded control characters; use the declared "
                "role transport token."
            )

    return None


def _backslash_transport_note(
    token: str,
) -> str:
    return (
        "## 反斜杠传输约定\n\n"
        f"下文数据中的 {token} 表示原文中的一个字面"
        "反斜杠字符。它只是为了保证严格 JSON 传输而使用的"
        "占位符，不是候选 Skill 的内容。理解内容时把它视为"
        "原始反斜杠；在输出 quoted_text、problem 或 "
        "replacement_text 时，如需保留该字符，必须继续原样"
        "使用这个占位符，不要自行还原。程序只会在模型输出"
        "首先通过严格裸 JSON 解析后机械还原该占位符。"
    )


def _line_feed_transport_note(
    token: str,
) -> str:
    return (
        "## 换行传输约定\n\n"
        f"角色问题数据中的 {token} 表示原文中的一个"
        "换行字符。它只是严格 JSON 传输占位符，不是"
        "候选 Skill 的内容。replacement_text 如需换行，"
        "必须原样输出该占位符；不要输出字面控制字符"
        "或 JSON \\n转义。程序只会在模型输出首先通过"
        "严格裸 JSON 解析和控制字符检查后机械还原"
        "该占位符。"
    )


def _load_role_system_prompt(
    prompt_name: str,
    *,
    env: str,
) -> str:
    """Inject the same target-role profile into both system prompts."""
    template = load_prompt(
        prompt_name,
        env=env,
    )
    role_profile = load_prompt(
        "role_consistency_role",
        env=env,
    )

    if template.count(_ROLE_PROFILE_PLACEHOLDER) != 1:
        raise ValueError(
            f"{prompt_name}.md must contain exactly one "
            f"{_ROLE_PROFILE_PLACEHOLDER!r} placeholder."
        )

    return template.replace(
        _ROLE_PROFILE_PLACEHOLDER,
        role_profile.rstrip(),
    )


def _candidate_block(
    candidate_skill: str,
) -> str:
    text = str(candidate_skill)

    if text and not text.endswith("\n"):
        text += "\n"

    return text


def _sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def structured_atomic_skill_error(
    candidate_skill: Any,
) -> str | None:
    """Return a deterministic error for a two-field solver Skill."""
    try:
        parse_atomic_skill_draft(
            candidate_skill,
            path="$.skill",
        )
    except (StructuredSkillError, ValueError) as error:
        return str(error)
    return None


def _canonical_structured_atomic_skill(
    candidate_skill: dict[str, Any],
) -> dict[str, str]:
    """Return one field-ordered copy after deterministic validation."""
    error = structured_atomic_skill_error(
        candidate_skill
    )
    if error is not None:
        raise ValueError(error)

    return parse_atomic_skill_draft(
        candidate_skill,
        path="$.skill",
    ).to_dict()


def _structured_atomic_skill_json(
    candidate_skill: dict[str, Any],
) -> str:
    return json.dumps(
        _canonical_structured_atomic_skill(
            candidate_skill
        ),
        ensure_ascii=False,
        indent=2,
    )


def _structured_atomic_skill_anchor_text(
    candidate_skill: dict[str, Any],
) -> str:
    """Render a non-canonical view for the existing anchor locator."""
    skill = _canonical_structured_atomic_skill(
        candidate_skill
    )
    return (
        f"Title: {skill['title']}\n"
        "Solver guidance: "
        f"{skill['solver_guidance']}"
    )


def _structured_role_anchor_catalog(
    candidate_skill: dict[str, Any],
) -> list[dict[str, Any]]:
    rendered = _structured_atomic_skill_anchor_text(
        candidate_skill
    )
    anchors = _build_role_anchor_catalog(
        rendered
    )
    return _with_full_candidate_review_anchor(
        _structured_atomic_skill_json(candidate_skill),
        anchors,
    )


def _build_role_anchor_catalog(
    candidate_skill: str,
) -> list[dict[str, Any]]:
    candidate = str(candidate_skill)
    matches: list[dict[str, Any]] = []

    for priority, rule in enumerate(
        _ROLE_ANCHOR_RULES
    ):
        for match in rule["pattern"].finditer(
            candidate
        ):
            start, end = match.span()

            if start == end:
                continue

            line_start = (
                candidate.rfind(
                    "\n",
                    0,
                    start,
                )
                + 1
            )
            line_end = candidate.find(
                "\n",
                end,
            )

            if line_end < 0:
                line_end = len(candidate)

            marker = candidate[start:end]
            unit_text = candidate[
                line_start:line_end
            ]

            matches.append({
                "priority": priority,
                "kind": rule["kind"],
                "forced_problem": rule[
                    "forced_problem"
                ],
                "start": start,
                "end": end,
                "line_start": line_start,
                "line_end": line_end,
                "marker": marker,
                "unit_text": unit_text,
                "marker_sha256": _sha256_text(
                    marker
                ),
                "unit_sha256": _sha256_text(
                    unit_text
                ),
            })

    selected: list[dict[str, Any]] = []

    for item in sorted(
        matches,
        key=lambda value: (
            value["start"],
            value["priority"],
            -(
                value["end"]
                - value["start"]
            ),
        ),
    ):
        if any(
            item["start"] < previous["end"]
            and previous["start"] < item["end"]
            for previous in selected
        ):
            continue

        selected.append(item)

    selected.sort(
        key=lambda value: (
            value["start"],
            value["end"],
        )
    )

    for index, item in enumerate(
        selected,
        start=1,
    ):
        item["anchor_id"] = (
            f"A{index:04d}"
        )

    return selected


def _with_full_candidate_review_anchor(
    candidate_skill: str,
    anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append one semantic review item for the complete non-empty Skill."""
    candidate = str(candidate_skill)
    if not candidate.strip():
        return anchors

    review_marker = "[complete candidate Skill]"
    reviewed_anchors = list(anchors)
    reviewed_anchors.append({
        "priority": len(_ROLE_ANCHOR_RULES),
        "kind": "full_candidate_review",
        "forced_problem": False,
        "start": 0,
        "end": len(candidate),
        "line_start": 0,
        "line_end": len(candidate),
        "marker": review_marker,
        "unit_text": (
            "Review the complete candidate Skill shown above, "
            "including every independent ### mathematics "
            "problem-solving subskill."
        ),
        "marker_sha256": _sha256_text(
            review_marker
        ),
        "unit_sha256": _sha256_text(
            candidate
        ),
        "anchor_id": (
            f"A{len(reviewed_anchors) + 1:04d}"
        ),
    })
    return reviewed_anchors


def _public_role_anchor(
    anchor: dict[str, Any],
) -> dict[str, Any]:
    return {
        "anchor_id": anchor["anchor_id"],
        "kind": anchor["kind"],
        "marker": anchor["marker"],
        "context": anchor["unit_text"],
        "forced_problem": anchor[
            "forced_problem"
        ],
    }


def _build_role_check_user_prompt(
    candidate_skill: str,
    anchors: list[dict[str, Any]],
) -> str:
    public_anchors = [
        _public_role_anchor(anchor)
        for anchor in anchors
    ]

    return (
        "## Candidate Skill\n\n"
        "----- BEGIN CANDIDATE SKILL -----\n"
        f"{_candidate_block(candidate_skill)}"
        "----- END CANDIDATE SKILL -----\n\n"
        "## Program-Located Candidate Role Anchors\n\n"
        f"{json.dumps(public_anchors, ensure_ascii=False, indent=2)}"
        "\n\n"
        "Determine for each anchor_id whether it constitutes a "
        "role-consistency issue. Do not copy, rewrite, or supplement "
        "the candidate Skill source text."
    )


def _validate_role_decision_payload(
    payload: dict,
    *,
    anchors: list[dict[str, Any]],
) -> str | None:
    if set(payload) != _ROLE_DECISION_KEYS:
        return (
            "Role checker output must contain exactly "
            "'decisions'."
        )

    decisions = payload.get(
        "decisions"
    )

    if not isinstance(decisions, list):
        return "'decisions' must be a JSON array."

    if len(decisions) != len(anchors):
        return (
            "Role checker output must contain exactly "
            "one decision for every anchor."
        )

    for index, (
        decision,
        anchor,
    ) in enumerate(
        zip(decisions, anchors),
        start=1,
    ):
        if not isinstance(decision, dict):
            return (
                f"Decision {index} must be a JSON object."
            )

        if (
            set(decision)
            != _ROLE_DECISION_ITEM_KEYS
        ):
            return (
                f"Decision {index} must contain exactly "
                "'anchor_id' and 'is_problem'."
            )

        if (
            decision.get("anchor_id")
            != anchor["anchor_id"]
        ):
            return (
                "Decisions must cover every anchor_id "
                "once and in source order."
            )

        if type(
            decision.get("is_problem")
        ) is not bool:
            return (
                f"Decision {index} is_problem must "
                "be a JSON boolean."
            )

        if (
            anchor["forced_problem"]
            and not decision["is_problem"]
        ):
            return (
                f"Decision {index} must classify the "
                "explicit unresolved draft marker as "
                "a role problem."
            )

    return None


def _role_issue_record(
    anchor: dict[str, Any],
) -> dict[str, Any]:
    if anchor["kind"] == "external_role_or_tool":
        problem = (
            "The source explicitly refers to another role, "
            "an unavailable tool, or internal workflow state."
        )
    elif anchor["kind"] == "full_candidate_review":
        problem = (
            "The complete candidate Skill is not directly usable "
            "by a mathematics problem solver or contains role, "
            "tool, or internal-process traces."
        )
    else:
        problem = (
            "The source contains an explicit unresolved "
            "analysis-draft marker."
        )

    return {
        "quoted_text": anchor["marker"],
        "problem": problem,
        "anchor_ids": [
            anchor["anchor_id"],
        ],
        "anchor_kind": anchor["kind"],
        "source_start": anchor["start"],
        "source_end": anchor["end"],
        "source_unit_sha256": anchor[
            "unit_sha256"
        ],
    }


def _role_validation_feedback(
    previous_attempt: dict[str, Any],
) -> str:
    validation_error = previous_attempt.get(
        "validation_error"
    )

    if not validation_error:
        return ""

    immutable_guidance = ""
    if "immutable" in str(
        validation_error
    ).casefold():
        immutable_guidance = (
            "\nFor each unit, choose exactly one of the two equivalent "
            "encodings: start from masked_source and retain every "
            "ROLE_IMMUTABLE placeholder; or restore every placeholder "
            "to its source_text from the mapping. Do not mix, omit, "
            "add, or reorder them; choose no_commit if you cannot "
            "comply. Mathematical expressions that are rejected, "
            "incorrect, or unresolved are also immutable content: "
            "do not delete them as a form of summarization; organize "
            "them into statements usable by the problem solver while "
            "preserving their original cognitive status."
        )

    if "role-contamination marker" in str(
        validation_error
    ):
        immutable_guidance += (
            "\nEvery residual role anchor listed in the error must be "
            "fully resolved within its corresponding unit; leave other "
            "already-compliant text unchanged as much as possible. "
            "Do not promote a rejected candidate into usable advice."
        )

    if (
        "must contain exactly 'unit_id', 'decision', "
        "and 'replacement_text'"
    ) in str(validation_error):
        immutable_guidance += (
            "\nEvery repairs item must include unit_id, decision, and "
            "replacement_text. When decision=no_commit, keep "
            "decision=no_commit and output replacement_text as the "
            "empty string \"\". Do not switch to replace merely to "
            "satisfy the schema."
        )

    previous_output = previous_attempt.get(
        "raw_response"
    )
    previous_output_guidance = ""
    if isinstance(previous_output, str) and previous_output:
        previous_output_guidance = (
            "\nThe previous output follows and is data to be corrected "
            "only. Preserve the parts that already satisfy the "
            "requirements and correct only the mechanical errors "
            "identified above; do not revert to the unrepaired source "
            "text or change other units:\n"
            "----- BEGIN PREVIOUS OUTPUT -----\n"
            f"{previous_output}\n"
            "----- END PREVIOUS OUTPUT -----"
        )

    return (
        "## Mechanical Validation Feedback for the Previous Output\n\n"
        "The previous output is invalid. Correct only the mechanical "
        "protocol errors below; do not change the task or input.\n"
        f"Mechanical error: {validation_error}"
        f"{immutable_guidance}"
        f"{previous_output_guidance}"
    )


def _selected_role_anchors(
    candidate_skill: str,
    issues: list[dict[str, Any]],
    *,
    full_candidate_review_only: bool = False,
) -> tuple[
    list[dict[str, Any]],
    str | None,
]:
    if full_candidate_review_only:
        catalog = (
            _with_full_candidate_review_anchor(
                candidate_skill,
                [],
            )
        )
    else:
        catalog = _build_role_anchor_catalog(
            candidate_skill
        )
    by_id = {
        anchor["anchor_id"]: anchor
        for anchor in catalog
    }
    selected_ids: list[str] = []

    for index, issue in enumerate(
        issues,
        start=1,
    ):
        anchor_ids = issue.get(
            "anchor_ids"
        )

        if (
            not isinstance(anchor_ids, list)
            or not anchor_ids
            or not all(
                isinstance(value, str)
                and value
                for value in anchor_ids
            )
        ):
            return [], (
                f"Issue {index} must contain a non-empty "
                "anchor_ids array."
            )

        selected_ids.extend(
            anchor_ids
        )

    if len(selected_ids) != len(
        set(selected_ids)
    ):
        return [], (
            "Issue anchor_ids must not repeat."
        )

    unknown_ids = [
        anchor_id
        for anchor_id in selected_ids
        if anchor_id not in by_id
    ]

    if unknown_ids:
        return [], (
            "Issue anchor_ids must refer to the current "
            f"candidate catalog: {unknown_ids!r}."
        )

    selected_set = set(
        selected_ids
    )
    selected = [
        anchor
        for anchor in catalog
        if anchor["anchor_id"]
        in selected_set
    ]

    return selected, None


def _role_anchor_in_slow_update(
    candidate_skill: str,
    anchor: dict[str, Any],
) -> bool:
    start = candidate_skill.find(
        _SLOW_UPDATE_START
    )
    end = candidate_skill.find(
        _SLOW_UPDATE_END
    )

    if start < 0 or end < 0:
        return False

    protected_end = (
        end + len(_SLOW_UPDATE_END)
    )
    return (
        start
        <= anchor["start"]
        < protected_end
    )


def _role_rewrite_atom_spans(
    source_text: str,
) -> list[tuple[int, int, str]]:
    candidates: list[
        tuple[int, int, int, str]
    ] = []

    for match in (
        _PROTECTED_MATH_OR_NUMBER_RE.finditer(
            source_text
        )
    ):
        candidates.append((
            match.start(),
            match.end(),
            0,
            match.group(0),
        ))

    for match in re.finditer(r"\\", source_text):
        candidates.append((
            match.start(),
            match.end(),
            1,
            match.group(0),
        ))

    selected: list[tuple[int, int, str]] = []
    for start, end, priority, text in sorted(
        candidates,
        key=lambda value: (
            value[0],
            value[2],
            -(value[1] - value[0]),
        ),
    ):
        del priority
        if any(
            start < previous_end
            and previous_start < end
            for previous_start, previous_end, _text
            in selected
        ):
            continue
        selected.append((start, end, text))

    selected.sort(
        key=lambda value: (
            value[0],
            value[1],
        )
    )
    return selected


def _select_role_immutable_namespace(
    candidate_skill: str,
) -> int:
    namespace = 0
    while (
        f"⟦ROLE_IMMUTABLE_{namespace}_"
        in candidate_skill
    ):
        namespace += 1
    return namespace


def _build_role_repair_units(
    candidate_skill: str,
    anchors: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    grouped: dict[
        tuple[int, int],
        list[dict[str, Any]],
    ] = {}

    for anchor in anchors:
        if _role_anchor_in_slow_update(
            candidate_skill,
            anchor,
        ):
            return [], (
                f"Anchor {anchor['anchor_id']} intersects "
                "protected slow-update content."
            )
        key = (
            int(anchor["line_start"]),
            int(anchor["line_end"]),
        )
        grouped.setdefault(key, []).append(
            anchor
        )

    namespace = _select_role_immutable_namespace(
        candidate_skill
    )
    units: list[dict[str, Any]] = []

    for unit_index, (
        (line_start, line_end),
        unit_anchors,
    ) in enumerate(
        sorted(grouped.items()),
        start=1,
    ):
        source_start = line_start
        source_end = line_end
        while (
            source_start < source_end
            and candidate_skill[source_start]
            in " \t"
        ):
            source_start += 1
        while (
            source_end > source_start
            and candidate_skill[source_end - 1]
            in " \t"
        ):
            source_end -= 1

        source_text = candidate_skill[
            source_start:source_end
        ]
        if not source_text:
            return [], (
                "A role repair unit must contain non-whitespace "
                "source text."
            )

        unit_id = f"U{unit_index:04d}"
        atoms: list[dict[str, str]] = []
        pieces: list[str] = []
        cursor = 0

        for atom_index, (
            atom_start,
            atom_end,
            atom_text,
        ) in enumerate(
            _role_rewrite_atom_spans(
                source_text
            ),
            start=1,
        ):
            token = (
                f"⟦ROLE_IMMUTABLE_{namespace}_"
                f"{unit_id}_{atom_index:04d}⟧"
            )
            pieces.append(
                source_text[cursor:atom_start]
            )
            pieces.append(token)
            atoms.append({
                "token": token,
                "source_text": atom_text,
                "source_ordinal": atom_index,
                "left_context": source_text[
                    max(0, atom_start - 48):atom_start
                ],
                "right_context": source_text[
                    atom_end:min(
                        len(source_text),
                        atom_end + 48,
                    )
                ],
            })
            cursor = atom_end

        pieces.append(source_text[cursor:])
        masked_source = "".join(pieces)
        units.append({
            "unit_id": unit_id,
            "anchor_ids": [
                anchor["anchor_id"]
                for anchor in unit_anchors
            ],
            "anchor_kinds": [
                anchor["kind"]
                for anchor in unit_anchors
            ],
            "source_start": source_start,
            "source_end": source_end,
            "source_text": source_text,
            "source_sha256": _sha256_text(
                source_text
            ),
            "masked_source": masked_source,
            "immutable_atoms": atoms,
        })

    return units, None


def _public_role_repair_unit(
    unit: dict[str, Any],
) -> dict[str, Any]:
    cognitive_status_obligations = []
    for anchor_id, anchor_kind in zip(
        unit["anchor_ids"],
        unit["anchor_kinds"],
    ):
        if (
            anchor_kind
            in _COGNITIVE_REJECTED_ANCHOR_KINDS
        ):
            required_status = "rejected"
        elif (
            anchor_kind
            in _COGNITIVE_UNRESOLVED_ANCHOR_KINDS
        ):
            required_status = "unresolved"
        else:
            continue

        cognitive_status_obligations.append({
            "anchor_id": anchor_id,
            "required_status": required_status,
        })

    return {
        "unit_id": unit["unit_id"],
        "anchor_ids": unit["anchor_ids"],
        "anchor_kinds": unit["anchor_kinds"],
        "cognitive_status_obligations": (
            cognitive_status_obligations
        ),
        "masked_source": unit[
            "masked_source"
        ],
        "immutable_atoms": unit[
            "immutable_atoms"
        ],
    }


def _build_role_repair_user_prompt(
    candidate_skill: str,
    units: list[dict[str, Any]],
) -> str:
    public_units = [
        _public_role_repair_unit(unit)
        for unit in units
    ]
    return (
        "## Complete Candidate Skill (Read-Only Context)\n\n"
        "----- BEGIN CANDIDATE SKILL -----\n"
        f"{_candidate_block(candidate_skill)}"
        "----- END CANDIDATE SKILL -----\n\n"
        "## Structural Units to Repair\n\n"
        f"{json.dumps(public_units, ensure_ascii=False, indent=2)}"
        "\n\n"
        "Using the complete context, minimally reorganize each unit_id "
        "into a local passage suitable for a mathematics problem "
        "solver. Do not replace an old draft marker with another "
        "process label. The rejected or unresolved status of every "
        "proposition in cognitive_status_obligations must be preserved; "
        "the program will verify this mechanically before committing "
        "the repair. Choose no_commit when a safe reorganization is "
        "not possible."
    )


def _restore_role_unit_atoms(
    masked_text: str,
    unit: dict[str, Any],
) -> str:
    restored = masked_text
    for atom in unit["immutable_atoms"]:
        restored = restored.replace(
            atom["token"],
            atom["source_text"],
            1,
        )
    return restored


def _canonicalize_role_rewrite_payload(
    payload: dict,
    *,
    units: list[dict[str, Any]],
) -> tuple[dict, str | None]:
    """Canonicalize one complete immutable-wire encoding per unit."""
    canonical = copy.deepcopy(payload)
    repairs = canonical.get("repairs")

    if (
        set(canonical) != _ROLE_REWRITE_KEYS
        or not isinstance(repairs, list)
        or len(repairs) != len(units)
    ):
        return canonical, None

    for index, (repair, unit) in enumerate(
        zip(repairs, units),
        start=1,
    ):
        if (
            not isinstance(repair, dict)
            or set(repair) != _ROLE_REWRITE_ITEM_KEYS
            or repair.get("unit_id") != unit["unit_id"]
            or repair.get("decision") != "replace"
            or not isinstance(
                repair.get("replacement_text"),
                str,
            )
        ):
            continue

        replacement = repair["replacement_text"]
        expected_tokens = [
            atom["token"]
            for atom in unit["immutable_atoms"]
        ]
        expected_atoms = [
            atom["source_text"]
            for atom in unit["immutable_atoms"]
        ]
        actual_tokens = (
            _ROLE_IMMUTABLE_TOKEN_BROAD_RE.findall(
                replacement
            )
        )

        if actual_tokens:
            if actual_tokens != expected_tokens:
                remaining_tokens = list(
                    actual_tokens
                )
                missing_tokens: list[str] = []
                for token in expected_tokens:
                    if token in remaining_tokens:
                        remaining_tokens.remove(token)
                    else:
                        missing_tokens.append(token)
                source_by_token = {
                    atom["token"]: atom
                    for atom in unit["immutable_atoms"]
                }
                missing_atoms = [
                    copy.deepcopy(
                        source_by_token[token]
                    )
                    for token in missing_tokens
                ]
                return canonical, (
                    f"Repair {index} immutable token sequence mismatch "
                    f"for {unit['unit_id']}: expected_count="
                    f"{len(expected_tokens)}, actual_count="
                    f"{len(actual_tokens)}, missing_atoms="
                    f"{missing_atoms}, extra={remaining_tokens}, "
                    "and source order must be unchanged. Use either "
                    "every immutable token or no immutable tokens."
                )

            without_tokens = replacement
            for token in expected_tokens:
                without_tokens = without_tokens.replace(
                    token,
                    "",
                    1,
                )

            if (
                _ROLE_IMMUTABLE_TOKEN_BROAD_RE.search(
                    without_tokens
                )
                or _role_rewrite_atom_spans(
                    without_tokens
                )
            ):
                return canonical, (
                    f"Repair {index} must not mix immutable tokens "
                    "with raw immutable source text."
                )
            continue

        raw_spans = _role_rewrite_atom_spans(
            replacement
        )
        actual_atoms = [
            atom_text
            for _start, _end, atom_text in raw_spans
        ]
        if actual_atoms != expected_atoms:
            remaining_atoms = list(actual_atoms)
            missing_atoms = []
            for atom in unit["immutable_atoms"]:
                atom_text = atom["source_text"]
                if atom_text in remaining_atoms:
                    remaining_atoms.remove(atom_text)
                else:
                    missing_atoms.append(
                        copy.deepcopy(atom)
                    )
            return canonical, (
                f"Repair {index} immutable raw sequence mismatch "
                f"for {unit['unit_id']}: expected_count="
                f"{len(expected_atoms)}, actual_count="
                f"{len(actual_atoms)}, missing_atoms="
                f"{missing_atoms}, extra_atoms={remaining_atoms}, "
                "and source order must be unchanged."
            )

        canonical_replacement = replacement
        for (
            atom_start,
            atom_end,
            _atom_text,
        ), token in reversed(
            list(zip(raw_spans, expected_tokens))
        ):
            canonical_replacement = (
                canonical_replacement[:atom_start]
                + token
                + canonical_replacement[atom_end:]
            )
        repair["replacement_text"] = (
            canonical_replacement
        )

    return canonical, None


def _role_rewrite_wire_error(
    payload: dict,
    *,
    units: list[dict[str, Any]],
) -> str | None:
    _canonical, error = (
        _canonicalize_role_rewrite_payload(
            payload,
            units=units,
        )
    )
    return error


def _decode_role_rewrite_wire(
    payload: dict,
    *,
    units: list[dict[str, Any]],
) -> dict:
    canonical, error = (
        _canonicalize_role_rewrite_payload(
            payload,
            units=units,
        )
    )
    if error is not None:
        raise ValueError(error)
    return canonical


def _protected_logic_tokens(
    text: str,
) -> list[str]:
    return [
        match.group(0)
        for match in _PROTECTED_LOGIC_RE.finditer(
            text
        )
    ]


def _is_ordered_subsequence(
    required: list[str],
    observed: list[str],
) -> bool:
    observed_index = 0

    for required_token in required:
        while (
            observed_index < len(observed)
            and observed[observed_index]
            != required_token
        ):
            observed_index += 1

        if observed_index >= len(observed):
            return False

        observed_index += 1

    return True


def _cognitive_status_shadow(
    text: str,
) -> tuple[
    str,
    list[tuple[int, int, str]],
]:
    spans = [
        (
            match.start(),
            match.end(),
            match.group(0),
        )
        for match in (
            _PROTECTED_MATH_OR_NUMBER_RE.finditer(
                text
            )
        )
    ]
    characters = list(text)
    for start, end, _token in spans:
        characters[start:end] = " " * (
            end - start
        )
    return "".join(characters), spans


def _cognitive_status_envelope(
    shadow: str,
    start: int,
    end: int,
) -> tuple[int, int]:
    left = 0
    right = len(shadow)

    for match in (
        _COGNITIVE_STATUS_BOUNDARY_RE.finditer(
            shadow
        )
    ):
        if match.end() <= start:
            left = match.end()
            continue
        if match.start() >= end:
            right = match.start()
            break

    return left, right


def _cognitive_coordinate_group(
    text: str,
    spans: list[tuple[int, int, str]],
    *,
    base_index: int,
    envelope_start: int,
    expand: bool,
) -> set[int]:
    group = {base_index}
    if not expand:
        return group
    current_index = base_index

    while current_index > 0:
        previous_index = current_index - 1
        previous = spans[previous_index]
        current = spans[current_index]
        if previous[0] < envelope_start:
            break
        gap = text[previous[1]:current[0]]
        if not _COGNITIVE_COORDINATOR_RE.fullmatch(
            gap
        ):
            break
        group.add(previous_index)
        current_index = previous_index

    return group


def _cognitive_cue_targets(
    text: str,
    spans: list[tuple[int, int, str]],
    *,
    cue_start: int,
    cue_end: int,
    envelope_start: int,
    envelope_end: int,
    prefix: bool,
    expand_coordinators: bool,
    allowed_coordinate_indices: set[int],
) -> set[int]:
    envelope_indices = [
        index
        for index, (start, end, _token)
        in enumerate(spans)
        if (
            envelope_start <= start
            and end <= envelope_end
        )
    ]
    if not envelope_indices:
        return set()

    if prefix:
        following = [
            index
            for index in envelope_indices
            if spans[index][0] >= cue_end
        ]
        if not following:
            return set()
        base_index = following[0]
        gap = text[cue_end:spans[base_index][0]]
        if gap.strip(" \t,:，："):
            return set()
        targets = {base_index}
        if not targets.issubset(
            allowed_coordinate_indices
        ):
            return set()
        return targets

    preceding = [
        index
        for index in envelope_indices
        if spans[index][1] <= cue_start
    ]
    if not preceding:
        return set()
    base_index = preceding[-1]
    gap = text[spans[base_index][1]:cue_start]
    if gap.strip(" \t,:，："):
        return set()
    targets = _cognitive_coordinate_group(
        text,
        spans,
        base_index=base_index,
        envelope_start=envelope_start,
        expand=expand_coordinators,
    )
    if not targets.issubset(
        allowed_coordinate_indices
    ):
        return set()
    return targets


def _cognitive_status_for_claim(
    text: str,
    spans: list[tuple[int, int, str]],
    *,
    claim_index: int,
    envelope_start: int,
    envelope_end: int,
    allowed_coordinate_indices: set[int],
) -> str:
    envelope = text[
        envelope_start:envelope_end
    ]
    double_negations = [
        (
            envelope_start + match.start(),
            envelope_start + match.end(),
        )
        for match in (
            _COGNITIVE_DOUBLE_NEGATION_RE.finditer(
                envelope
            )
        )
    ]
    statuses: set[str] = set()

    for status, pattern, prefix_pattern in (
        (
            "unresolved",
            _COGNITIVE_UNRESOLVED_CUE_RE,
            _COGNITIVE_PREFIX_UNRESOLVED_RE,
        ),
        (
            "rejected",
            _COGNITIVE_REJECTED_CUE_RE,
            _COGNITIVE_PREFIX_REJECTION_RE,
        ),
    ):
        for match in pattern.finditer(envelope):
            cue_start = envelope_start + match.start()
            cue_end = envelope_start + match.end()
            if (
                status == "rejected"
                and any(
                    cue_start < end
                    and start < cue_end
                    for start, end in double_negations
                )
            ):
                continue
            targets = _cognitive_cue_targets(
                text,
                spans,
                cue_start=cue_start,
                cue_end=cue_end,
                envelope_start=envelope_start,
                envelope_end=envelope_end,
                prefix=bool(
                    prefix_pattern.fullmatch(
                        match.group(0).strip()
                    )
                ),
                expand_coordinators=bool(
                    re.match(
                        r"^(?:are|were|do|fail)\b",
                        match.group(0).strip(),
                        re.IGNORECASE,
                    )
                ),
                allowed_coordinate_indices=(
                    allowed_coordinate_indices
                ),
            )
            if claim_index in targets:
                statuses.add(status)

    if len(statuses) > 1:
        return "ambiguous"
    if statuses:
        return next(iter(statuses))
    return "affirmed"


def _wait_transition_requires_fail_closed(
    original_unit: str,
    anchor: dict[str, Any],
) -> bool:
    shadow, _spans = _cognitive_status_shadow(
        original_unit
    )
    boundary = _COGNITIVE_STATUS_BOUNDARY_RE.search(
        shadow,
        int(anchor["end"]),
    )
    tail_end = (
        boundary.start()
        if boundary is not None
        else len(original_unit)
    )
    tail = original_unit[
        int(anchor["end"]):tail_end
    ]
    return bool(
        re.match(
            r"[ \t]*(?:maybe|perhaps|try)\b",
            tail,
            re.IGNORECASE,
        )
        or _COGNITIVE_UNRESOLVED_CUE_RE.search(
            tail
        )
        or _COGNITIVE_REJECTED_CUE_RE.search(
            tail
        )
    )


def _cognitive_status_promotion_error(
    original_unit: str,
    repaired_unit: str,
) -> str | None:
    """Reject status promotion without judging mathematical truth."""
    original_shadow, original_spans = (
        _cognitive_status_shadow(
            original_unit
        )
    )
    repaired_shadow, repaired_spans = (
        _cognitive_status_shadow(
            repaired_unit
        )
    )

    original_tokens = [
        token
        for _start, _end, token in original_spans
    ]
    repaired_tokens = [
        token
        for _start, _end, token in repaired_spans
    ]
    if original_tokens != repaired_tokens:
        return (
            "Cognitive-status guard requires identical mathematical "
            "and numeric occurrences in source order."
        )

    anchors = _build_role_anchor_catalog(
        original_unit
    )
    unsafe_untracked = [
        anchor
        for anchor in anchors
        if (
            anchor["kind"]
            in _COGNITIVE_FAIL_CLOSED_ANCHOR_KINDS
            or (
                anchor["kind"] == "wait_transition"
                and _wait_transition_requires_fail_closed(
                    original_unit,
                    anchor,
                )
            )
        )
    ]
    if unsafe_untracked:
        kinds = [
            anchor["kind"]
            for anchor in unsafe_untracked
        ]
        return (
            "Cognitive-status guard cannot safely infer the claim "
            f"scope for process/source anchors: {kinds}."
        )

    tracked_anchors = [
        anchor
        for anchor in anchors
        if (
            anchor["kind"]
            in _COGNITIVE_REJECTED_ANCHOR_KINDS
            or anchor["kind"]
            in _COGNITIVE_UNRESOLVED_ANCHOR_KINDS
        )
    ]

    obligations: list[dict[str, Any]] = []
    for obligation_index, anchor in enumerate(
        tracked_anchors,
        start=1,
    ):
        claim_start, _claim_end = (
            _cognitive_status_envelope(
                original_shadow,
                anchor["start"],
                anchor["start"],
            )
        )
        claim_indices = [
            index
            for index, (start, end, _token)
            in enumerate(original_spans)
            if (
                claim_start <= start
                and end <= anchor["start"]
            )
        ]
        if not claim_indices:
            return (
                f"Cognitive-status guard cannot uniquely track "
                f"status obligation {obligation_index} "
                f"({anchor['kind']}) because "
                "its claim has no mathematical or numeric occurrence."
            )

        required_status = (
            "rejected"
            if (
                anchor["kind"]
                in _COGNITIVE_REJECTED_ANCHOR_KINDS
            )
            else "unresolved"
        )
        obligations.append({
            "obligation_index": obligation_index,
            "anchor": anchor,
            "claim_index": claim_indices[-1],
            "required_status": required_status,
        })

    allowed_by_status = {
        status: {
            int(obligation["claim_index"])
            for obligation in obligations
            if obligation["required_status"] == status
        }
        for status in (
            "rejected",
            "unresolved",
        )
    }

    for obligation in obligations:
        obligation_index = int(
            obligation["obligation_index"]
        )
        anchor = obligation["anchor"]
        claim_index = int(
            obligation["claim_index"]
        )
        required_status = str(
            obligation["required_status"]
        )
        repaired_claim_span = repaired_spans[
            claim_index
        ]
        first_start = repaired_claim_span[0]
        last_end = repaired_claim_span[1]
        envelope_start, envelope_end = (
            _cognitive_status_envelope(
                repaired_shadow,
                first_start,
                first_start,
            )
        )
        if last_end > envelope_end:
            return (
                "Cognitive-status guard cannot map status obligation "
                f"{obligation_index} "
                "to one repaired status envelope."
            )

        observed_status = _cognitive_status_for_claim(
            repaired_unit,
            repaired_spans,
            claim_index=claim_index,
            envelope_start=envelope_start,
            envelope_end=envelope_end,
            allowed_coordinate_indices=(
                allowed_by_status[required_status]
            ),
        )
        if observed_status != required_status:
            return (
                "Cognitive-status guard rejected status obligation "
                f"{obligation_index} ({anchor['kind']}): "
                f"required={required_status}, "
                f"observed={observed_status}."
            )

    return None


def _validate_role_rewrite_payload(
    payload: dict,
    *,
    units: list[dict[str, Any]],
) -> str | None:
    if set(payload) != _ROLE_REWRITE_KEYS:
        return (
            "Role repair output must contain exactly "
            "'repairs'."
        )
    repairs = payload.get("repairs")
    if not isinstance(repairs, list):
        return "'repairs' must be a JSON array."
    if len(repairs) != len(units):
        return (
            "Role repair output must contain exactly one "
            "repair for every unit."
        )

    for index, (repair, unit) in enumerate(
        zip(repairs, units),
        start=1,
    ):
        if not isinstance(repair, dict):
            return f"Repair {index} must be a JSON object."
        if set(repair) != _ROLE_REWRITE_ITEM_KEYS:
            return (
                f"Repair {index} must contain exactly "
                "'unit_id', 'decision', and 'replacement_text'."
            )
        if repair.get("unit_id") != unit["unit_id"]:
            return (
                "Repairs must cover every unit_id once and "
                "in source order."
            )
        decision = repair.get("decision")
        if decision not in _ROLE_REWRITE_DECISIONS:
            return (
                f"Repair {index} decision must be 'replace' "
                "or 'no_commit'."
            )
        replacement = repair.get(
            "replacement_text"
        )
        if not isinstance(replacement, str):
            return (
                f"Repair {index} replacement_text must be "
                "a string."
            )
        if decision == "no_commit":
            if replacement:
                return (
                    f"Repair {index} no_commit must use an "
                    "empty replacement_text."
                )
            continue
        if (
            not replacement
            or replacement != replacement.strip()
        ):
            return (
                f"Repair {index} replacement_text must be "
                "non-empty and contain no outer whitespace."
            )
        if "\r" in replacement or "\n" in replacement:
            return (
                f"Repair {index} must remain one source line."
            )
        if any(
            ord(character) < 0x20
            or character in {"\u2028", "\u2029"}
            for character in replacement
        ):
            return (
                f"Repair {index} must not contain control "
                "characters or Unicode line separators."
            )
        expected_tokens = [
            atom["token"]
            for atom in unit[
                "immutable_atoms"
            ]
        ]
        actual_tokens = (
            _ROLE_IMMUTABLE_TOKEN_BROAD_RE.findall(
                replacement
            )
        )
        if actual_tokens != expected_tokens:
            return (
                f"Repair {index} must preserve every immutable "
                "token exactly once and in source order."
            )
        restored = _restore_role_unit_atoms(
            replacement,
            unit,
        )
        if restored == unit["source_text"]:
            return (
                f"Repair {index} must use no_commit when the "
                "unit cannot be changed safely."
            )
        if (
            _protected_math_and_number_tokens(restored)
            != _protected_math_and_number_tokens(
                unit["source_text"]
            )
        ):
            return (
                f"Repair {index} changed a mathematical "
                "expression or number."
            )
        if not _is_ordered_subsequence(
            _protected_logic_tokens(
                unit["source_text"]
            ),
            _protected_logic_tokens(restored),
        ):
            return (
                f"Repair {index} removed or reordered an existing "
                "protected logic or epistemic-status expression."
            )
        if (
            _SLOW_UPDATE_START in restored
            or _SLOW_UPDATE_END in restored
            or _ROLE_IMMUTABLE_TOKEN_BROAD_RE.search(
                restored
            )
        ):
            return (
                f"Repair {index} introduced a protected or "
                "unknown transport marker."
            )
        if (
            _markdown_prefix_fingerprint(restored)
            != _markdown_prefix_fingerprint(
                unit["source_text"]
            )
        ):
            return (
                f"Repair {index} changed Markdown structure."
            )
        if _contains_explicit_role_contamination(restored):
            residual_anchors = [
                anchor
                for anchor in _build_role_anchor_catalog(
                    restored
                )
                if (
                    anchor.get("forced_problem")
                    or _contains_explicit_role_contamination(
                        anchor["marker"]
                    )
                )
            ]
            residual_detail = [
                {
                    "kind": anchor["kind"],
                    "marker": anchor["marker"],
                    "offset": anchor["start"],
                }
                for anchor in residual_anchors
            ]
            return (
                f"Repair {index} still contains an explicit "
                "role-contamination marker: "
                f"{residual_detail}."
            )
        if _ROLE_PROCESS_TRACE_RE.search(restored):
            return (
                f"Repair {index} replaced the original draft "
                "with another process-history label."
            )

    return None


def _line_ending_fingerprint(
    text: str,
) -> tuple[str, ...]:
    return tuple(
        re.findall(
            r"\r\n|\r|\n",
            text,
        )
    )


def _markdown_prefix_fingerprint(
    text: str,
) -> tuple[str, ...]:
    return tuple(
        match.group(0)
        for match in re.finditer(
            r"(?m)^[ \t]*(?:"
            r"#{1,6}[ \t]+|"
            r"(?:[-*+]|\d+[.)])[ \t]+"
            r")",
            text,
        )
    )


def _apply_role_unit_rewrites(
    candidate_skill: str,
    units: list[dict[str, Any]],
    repairs_payload: list[dict[str, str]],
) -> tuple[
    dict[str, Any] | None,
    str | None,
]:
    located: list[
        tuple[
            int,
            int,
            str,
            str,
            dict[str, Any],
        ]
    ] = []

    for unit, repair in zip(
        units,
        repairs_payload,
    ):
        if repair["decision"] != "replace":
            return None, (
                f"Unit {unit['unit_id']} requested no_commit."
            )
        replacement = _restore_role_unit_atoms(
            repair["replacement_text"],
            unit,
        )
        start = int(unit["source_start"])
        end = int(unit["source_end"])
        source_text = str(unit["source_text"])

        if (
            candidate_skill[start:end]
            != source_text
            or _sha256_text(source_text)
            != unit["source_sha256"]
        ):
            return None, (
                f"Unit {unit['unit_id']} no longer "
                "matches its frozen source span."
            )
        located.append((
            start,
            end,
            source_text,
            replacement,
            unit,
        ))

    located.sort(
        key=lambda value: value[0]
    )

    for previous, current in zip(
        located,
        located[1:],
    ):
        if current[0] < previous[1]:
            return None, (
            "Role repair unit spans must not overlap."
        )

    repaired_skill = candidate_skill

    for (
        start,
        end,
        _source_text,
        replacement,
        _unit,
    ) in reversed(
        located
    ):
        repaired_skill = (
            repaired_skill[:start]
            + replacement
            + repaired_skill[end:]
        )

    if (
        _protected_math_and_number_tokens(
            repaired_skill
        )
        != _protected_math_and_number_tokens(
            candidate_skill
        )
    ):
        return None, (
            "Role unit repair changed a mathematical "
            "expression or number."
        )

    if not _is_ordered_subsequence(
        _protected_logic_tokens(candidate_skill),
        _protected_logic_tokens(repaired_skill),
    ):
        return None, (
            "Role unit repair removed or reordered an existing "
            "protected logic or epistemic-status expression."
        )

    if (
        _line_ending_fingerprint(
            repaired_skill
        )
        != _line_ending_fingerprint(
            candidate_skill
        )
    ):
        return None, (
            "Role unit repair changed line endings."
        )

    if (
        _markdown_prefix_fingerprint(
            repaired_skill
        )
        != _markdown_prefix_fingerprint(
            candidate_skill
        )
    ):
        return None, (
            "Role unit repair changed Markdown structure."
        )

    if _contains_explicit_role_contamination(
        repaired_skill
    ):
        return None, (
            "Role unit repair left an explicit role-contamination "
            "marker in the candidate Skill."
        )

    if _ROLE_PROCESS_TRACE_RE.search(
        repaired_skill
    ):
        return None, (
            "Role unit repair left a process-history label in "
            "the candidate Skill."
        )

    repairs = [
        {
            "unit_id": unit["unit_id"],
            "anchor_ids": unit["anchor_ids"],
            "quoted_text": source_text,
            "replacement_text": replacement,
            "source_start": start,
            "source_end": end,
        }
        for (
            start,
            end,
            source_text,
            replacement,
            unit,
        ) in located
    ]
    apply_report = [
        {
            "op": "replace_offset",
            "unit_id": repair["unit_id"],
            "anchor_ids": repair["anchor_ids"],
            "target": repair["quoted_text"],
            "content_preview": repair[
                "replacement_text"
            ],
            "status": "applied_replace",
            "source_start": repair["source_start"],
            "source_end": repair["source_end"],
        }
        for repair in repairs
    ]

    return {
        "repaired_skill": repaired_skill,
        "repairs": repairs,
        "apply_report": apply_report,
    }, None


def _build_check_user_prompt(
    candidate_skill: str,
    *,
    transport_token: str | None = None,
) -> str:
    token = (
        transport_token
        or _select_backslash_transport_token(
            candidate_skill
        )
    )
    transported_candidate = (
        _encode_backslash_transport(
            candidate_skill,
            token=token,
        )
    )

    return (
        f"{_backslash_transport_note(token)}\n\n"
        "## 候选 Skill\n\n"
        "----- BEGIN CANDIDATE SKILL -----\n"
        f"{_candidate_block(transported_candidate)}"
        "----- END CANDIDATE SKILL -----\n\n"
        "请判断以上候选 Skill 是否符合目标角色，并严格按照"
        "系统提示词规定的 JSON 格式输出结果。"
    )


def _build_repair_user_prompt(
    candidate_skill: str,
    issues: list[dict[str, str]],
    *,
    transport_token: str | None = None,
    line_feed_transport_token: str | None = None,
) -> str:
    token = (
        transport_token
        or _select_backslash_transport_token(
            candidate_skill,
            issues,
        )
    )
    line_feed_token = (
        line_feed_transport_token
        or _select_line_feed_transport_token(
            candidate_skill,
            issues,
        )
    )
    indexed_issues = [
        {
            "issue_index": index,
            "quoted_text": issue[
                "quoted_text"
            ],
            "problem": issue["problem"],
        }
        for index, issue in enumerate(
            issues,
            start=1,
        )
    ]
    transported_candidate = (
        _encode_backslash_transport(
            candidate_skill,
            token=token,
        )
    )
    transported_issues = (
        _encode_line_feed_transport(
            _encode_backslash_transport(
                indexed_issues,
                token=token,
            ),
            token=line_feed_token,
        )
    )

    return (
        f"{_backslash_transport_note(token)}\n\n"
        f"{_line_feed_transport_note(line_feed_token)}\n\n"
        "## 候选 Skill\n\n"
        "----- BEGIN CANDIDATE SKILL -----\n"
        f"{_candidate_block(transported_candidate)}"
        "----- END CANDIDATE SKILL -----\n\n"
        "## 角色一致性问题\n\n"
        f"{json.dumps(transported_issues, ensure_ascii=False, indent=2)}"
        "\n\n"
        "请按照问题列表的原有顺序处理每个问题。每个 "
        "replacement_text 必须是非空的最小角色整理结果；"
        "只清除或转化明确的角色外壳，并逐字、按原顺序保留"
        "所有公式和数字，不得新增公式、数字、条件或结论。"
        "如果无法安全完成，必须原样返回对应 quoted_text，"
        "由程序执行 no-commit。请严格按照系统提示词规定的 "
        "JSON 格式输出 repairs。"
    )


def _append_format_reminder(
    user_prompt: str,
    *,
    attempt_index: int,
    expected_schema: str,
) -> str:
    if not (
        0
        <= attempt_index
        < len(_FORMAT_RETRY_VARIANTS_EN)
    ):
        raise ValueError(
            "attempt_index must select one of the three fixed "
            "format prompt variants, "
            f"got {attempt_index}."
        )

    base_prompt = (
        _FORMAT_RETRY_MASTER_EN.format(
            expected_schema=expected_schema,
        )
    )
    english_prompt = (
        _FORMAT_RETRY_VARIANTS_EN[
            attempt_index
        ].format(
            expected_schema=expected_schema,
        )
    )

    return (
        user_prompt
        + "\n\n"
        + base_prompt
        + "\n\n"
        + english_prompt
    )


def _extract_bare_json_object(
    response: str,
) -> tuple[dict | None, str | None]:
    text = str(response).strip()

    if not (
        text.startswith("{")
        and text.endswith("}")
    ):
        return None, "Response is not one bare JSON object."

    payload = extract_json(text)

    if not isinstance(payload, dict):
        return None, "Response could not be parsed as a JSON object."

    return payload, None


def _issue_location_error(
    candidate_skill: str,
    issues: list[dict[str, str]],
) -> str | None:
    quoted_texts: list[str] = []
    spans: list[tuple[int, int]] = []

    for index, issue in enumerate(issues):
        quoted_text = issue["quoted_text"]

        if quoted_text in quoted_texts:
            return (
                "Each issue must use a different quoted_text; "
                f"duplicate found at issue {index + 1}."
            )

        occurrence_count = candidate_skill.count(
            quoted_text
        )

        if occurrence_count != 1:
            return (
                "Each quoted_text must occur exactly once in the "
                f"candidate Skill; issue {index + 1} occurs "
                f"{occurrence_count} times."
            )

        start = candidate_skill.index(
            quoted_text
        )
        end = start + len(quoted_text)

        quoted_texts.append(
            quoted_text
        )
        spans.append(
            (start, end)
        )

    spans.sort()

    for previous, current in zip(
        spans,
        spans[1:],
    ):
        if current[0] < previous[1]:
            return (
                "Issue quoted_text spans must not overlap."
            )

    return None


def _validate_check_payload(
    payload: dict,
    *,
    candidate_skill: str,
    require_explicit_role_anchors: bool = False,
) -> str | None:
    if set(payload) != _CHECK_KEYS:
        return (
            "Checker output must contain exactly "
            "'passed' and 'issues'."
        )

    if type(payload.get("passed")) is not bool:
        return "'passed' must be a JSON boolean."

    issues = payload.get("issues")

    if not isinstance(issues, list):
        return "'issues' must be a JSON array."

    if payload["passed"] and issues:
        return (
            "'issues' must be empty when 'passed' is true."
        )

    if not payload["passed"] and not issues:
        return (
            "'issues' must be non-empty when 'passed' is false."
        )

    for index, issue in enumerate(
        issues
    ):
        if not isinstance(issue, dict):
            return (
                f"Issue {index + 1} must be a JSON object."
            )

        if set(issue) != _ISSUE_KEYS:
            return (
                f"Issue {index + 1} must contain exactly "
                "'quoted_text' and 'problem'."
            )

        if not isinstance(
            issue.get("quoted_text"),
            str,
        ) or not issue["quoted_text"].strip():
            return (
                f"Issue {index + 1} quoted_text must be "
                "a non-empty string."
            )

        if not isinstance(
            issue.get("problem"),
            str,
        ) or not issue["problem"].strip():
            return (
                f"Issue {index + 1} problem must be "
                "a non-empty string."
            )

    if issues:
        location_error = (
            _issue_location_error(
                candidate_skill,
                issues,
            )
        )

        if location_error is not None:
            return location_error

    if require_explicit_role_anchors:
        candidate_has_anchor = (
            _contains_explicit_role_contamination(
                candidate_skill
            )
        )
        actionable_issues = [
            (index, issue)
            for index, issue in enumerate(
                issues
            )
            if _contains_explicit_role_contamination(
                issue["quoted_text"]
            )
        ]

        if (
            payload["passed"]
            and candidate_has_anchor
        ):
            return (
                "A candidate containing an explicit "
                "role-contamination marker cannot pass "
                "without an issue."
            )

        if (
            not payload["passed"]
            and candidate_has_anchor
            and not actionable_issues
        ):
            return (
                "At least one issue must locate the explicit "
                "role-contamination marker."
            )

        for index, issue in actionable_issues:
            quoted_text = issue[
                "quoted_text"
            ]

            if (
                quoted_text
                != quoted_text.strip(
                    " \t\r\n"
                )
            ):
                return (
                    f"Issue {index + 1} must leave outer "
                    "whitespace outside quoted_text."
                )

            if not _is_complete_line_span(
                candidate_skill,
                quoted_text,
            ):
                return (
                    f"Issue {index + 1} quoted_text must "
                    "cover the complete non-whitespace "
                    "content of one or more source lines."
                )

    return None


def _partition_explicitly_allowed_role_issues(
    issues: list[dict[str, str]],
    *,
    candidate_skill: str,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
]:
    """Keep only issues anchored by explicit role contamination."""
    retained: list[dict[str, str]] = []
    ignored: list[dict[str, str]] = []

    for issue in issues:
        normalised_quote = " ".join(
            issue["quoted_text"].split()
        ).casefold()

        if (
            normalised_quote
            in _EXPLICITLY_ALLOWED_ROLE_ISSUE_QUOTES
            or not _contains_explicit_role_contamination(
                issue["quoted_text"]
            )
        ):
            ignored.append(
                copy.deepcopy(issue)
            )
        else:
            retained.append(
                copy.deepcopy(issue)
            )

    return retained, ignored


def _validate_repair_payload(
    payload: dict,
    *,
    candidate_skill: str,
    issues: list[dict[str, str]],
    allow_indexed: bool = False,
    require_source_extract: bool = False,
    require_nonempty: bool = False,
    require_protected_math_identity: bool = False,
    require_trimmed_replacement: bool = False,
    allow_unchanged: bool = False,
) -> str | None:
    if set(payload) != _REPAIR_KEYS:
        return (
            "Repair output must contain exactly 'repairs'."
        )

    repairs = payload.get("repairs")

    if not isinstance(repairs, list):
        return "'repairs' must be a JSON array."

    if len(repairs) != len(issues):
        return (
            "Repair output must contain exactly one repair "
            "for every issue."
        )

    item_format: str | None = None

    for index, repair in enumerate(
        repairs
    ):
        if not isinstance(repair, dict):
            return (
                f"Repair {index + 1} must be a JSON object."
            )

        repair_keys = set(repair)

        if (
            allow_indexed
            and repair_keys == _REPAIR_ITEM_KEYS
        ):
            current_format = "indexed"
        elif (
            repair_keys
            == _LEGACY_REPAIR_ITEM_KEYS
        ):
            current_format = "legacy"
        else:
            return (
                f"Repair {index + 1} must contain either "
                "'issue_index' and 'replacement_text', or "
                "'quoted_text' and 'replacement_text'."
            )

        if item_format is None:
            item_format = current_format
        elif current_format != item_format:
            return (
                "All repair items must use the same format."
            )

        expected_quote = issues[index][
            "quoted_text"
        ]

        if current_format == "indexed":
            issue_index = repair.get(
                "issue_index"
            )

            if type(issue_index) is not int:
                return (
                    f"Repair {index + 1} issue_index must "
                    "be a JSON integer."
                )

            if issue_index != index + 1:
                return (
                    "Repairs must preserve the one-based issue "
                    "order exactly."
                )

        else:
            quoted_text = repair.get(
                "quoted_text"
            )

            if not isinstance(
                quoted_text,
                str,
            ) or not quoted_text:
                return (
                    f"Repair {index + 1} quoted_text must be "
                    "a non-empty string."
                )

            if quoted_text != expected_quote:
                return (
                    "Repairs must preserve every issue quoted_text "
                    "exactly and remain in the same order."
                )

        replacement_text = repair.get(
            "replacement_text"
        )

        if not isinstance(
            replacement_text,
            str,
        ):
            return (
                f"Repair {index + 1} replacement_text must "
                "be a string."
            )

        if (
            require_nonempty
            and not replacement_text.strip()
        ):
            return (
                f"Repair {index + 1} replacement_text must "
                "be a non-empty string."
            )

        if (
            require_trimmed_replacement
            and replacement_text
            != replacement_text.strip()
        ):
            return (
                f"Repair {index + 1} replacement_text must "
                "not contain outer whitespace or line breaks."
            )

        if (
            require_protected_math_identity
            and _protected_math_and_number_tokens(
                replacement_text
            )
            != _protected_math_and_number_tokens(
                expected_quote
            )
        ):
            return (
                f"Repair {index + 1} must preserve every "
                "mathematical expression and number exactly, "
                "in the original order, without additions."
            )

        if _introduces_role_transport_token(
            replacement_text,
            source_text=expected_quote,
        ):
            return (
                f"Repair {index + 1} must not introduce "
                "a role transport token."
            )

        if (
            require_source_extract
            and not _is_exact_source_extract(
                replacement_text,
                source_text=expected_quote,
            )
        ):
            return (
                f"Repair {index + 1} replacement_text must be "
                "empty or one exact contiguous substring of "
                "quoted_text; characters must not be added, "
                "changed, reordered, or joined across a deletion."
            )

        if (
            replacement_text == expected_quote
            and not allow_unchanged
        ):
            return (
                f"Repair {index + 1} replacement_text must "
                "differ from quoted_text."
            )

        if (
            _SLOW_UPDATE_START in replacement_text
            or _SLOW_UPDATE_END in replacement_text
        ):
            return (
                f"Repair {index + 1} must not create or copy "
                "slow-update markers."
            )

    return _issue_location_error(
        candidate_skill,
        issues,
    )


def _normalise_repair_items(
    payload: dict,
    *,
    issues: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "quoted_text": issues[index][
                "quoted_text"
            ],
            "replacement_text": repair[
                "replacement_text"
            ],
        }
        for index, repair in enumerate(
            payload["repairs"]
        )
    ]


def _call_json_with_format_retry(
    *,
    system_prompt: str,
    user_prompt: str,
    expected_schema: str,
    validator: Callable[[dict], str | None],
    stage: str,
    max_completion_tokens: int,
    format_max_attempts: int,
    payload_decoder: (
        Callable[[dict], dict] | None
    ) = None,
    pre_decode_validator: (
        Callable[[dict], str | None] | None
    ) = None,
    validation_feedback_builder: (
        Callable[
            [dict[str, Any]],
            str,
        ]
        | None
    ) = None,
) -> tuple[dict | None, list[dict[str, Any]]]:
    if not 1 <= format_max_attempts <= 3:
        raise ValueError(
            "format_max_attempts must be between 1 and 3, "
            f"got {format_max_attempts}."
        )

    attempts: list[dict[str, Any]] = []

    for attempt_index in range(
        format_max_attempts
    ):
        attempt_record: dict[str, Any] = {
            "attempt": attempt_index + 1,
            "format_prompt_variant": (
                attempt_index + 1
            ),
        }
        attempt_user_prompt = user_prompt

        if (
            validation_feedback_builder
            is not None
            and attempts
        ):
            feedback = (
                validation_feedback_builder(
                    attempts[-1]
                )
            )

            if feedback:
                attempt_user_prompt = (
                    attempt_user_prompt
                    + "\n\n"
                    + feedback
                )

        try:
            response, usage = chat_optimizer(
                system=system_prompt,
                user=_append_format_reminder(
                    attempt_user_prompt,
                    attempt_index=attempt_index,
                    expected_schema=expected_schema,
                ),
                max_completion_tokens=max_completion_tokens,
                retries=4,
                stage=stage,
            )

        except Exception as error:  # noqa: BLE001
            attempt_record["error"] = (
                f"{type(error).__name__}: {error}"
            )
            attempts.append(
                attempt_record
            )

            if (
                attempt_index + 1
                < format_max_attempts
            ):
                print(
                    f"    [{stage}] model call failed; "
                    "retrying with an equivalent prompt variant"
                )

            continue

        text = str(response).strip()

        attempt_record["raw_response"] = text
        attempt_record["usage"] = (
            usage
            if isinstance(usage, dict)
            else {}
        )

        payload, parse_error = (
            _extract_bare_json_object(
                text
            )
        )

        validation_error = parse_error
        validated_payload = payload

        if (
            validation_error is None
            and validated_payload is not None
        ):
            if pre_decode_validator is not None:
                validation_error = (
                    pre_decode_validator(
                        validated_payload
                    )
                )

            if (
                validation_error is None
                and payload_decoder is not None
            ):
                validated_payload = (
                    payload_decoder(
                        validated_payload
                    )
                )

            if validation_error is None:
                validation_error = validator(
                    validated_payload
                )

        if validation_error is None:
            attempts.append(
                attempt_record
            )
            return validated_payload, attempts

        attempt_record["validation_error"] = (
            validation_error
        )
        attempts.append(
            attempt_record
        )

        if (
            attempt_index + 1
            < format_max_attempts
        ):
            print(
                f"    [{stage}] invalid output; "
                "retrying with an equivalent prompt variant"
            )

    return None, attempts


def check_role_consistency(
    candidate_skill: str,
    *,
    env: str = "forward_math",
    max_completion_tokens: int = 0,
    format_max_attempts: int = 3,
    full_candidate_review_only: bool = False,
) -> dict[str, Any]:
    """Check located anchors or semantically review the complete Skill."""
    anchors = (
        _with_full_candidate_review_anchor(
            candidate_skill,
            [],
        )
        if full_candidate_review_only
        else _build_role_anchor_catalog(
            candidate_skill
        )
    )

    if not anchors:
        return {
            "passed": True,
            "valid_response": True,
            "issues": [],
            "ignored_issues": [],
            "attempts": [],
        }

    payload, attempts = (
        _call_json_with_format_retry(
            system_prompt=_load_role_system_prompt(
                "role_consistency_check",
                env=env,
            ),
            user_prompt=_build_role_check_user_prompt(
                candidate_skill,
                anchors,
            ),
            expected_schema=(
                '{\n'
                '  "decisions": [\n'
                '    {\n'
                '      "anchor_id": "A0001",\n'
                '      "is_problem": true\n'
                '    }\n'
                '  ]\n'
                '}'
            ),
            validator=lambda value: (
                _validate_role_decision_payload(
                    value,
                    anchors=anchors,
                )
            ),
            stage="role_consistency_check",
            max_completion_tokens=max_completion_tokens,
            format_max_attempts=format_max_attempts,
            validation_feedback_builder=(
                _role_validation_feedback
            ),
        )
    )

    if payload is None:
        return {
            "passed": False,
            "valid_response": False,
            "issues": [],
            "ignored_issues": [],
            "attempts": attempts,
        }

    issues: list[dict[str, Any]] = []
    ignored_issues: list[
        dict[str, Any]
    ] = []

    for anchor, decision in zip(
        anchors,
        payload["decisions"],
    ):
        record = _role_issue_record(
            anchor
        )

        if decision["is_problem"]:
            issues.append(record)
        else:
            ignored_issues.append(
                record
            )

    return {
        "passed": not issues,
        "valid_response": True,
        "issues": issues,
        "ignored_issues": ignored_issues,
        "attempts": attempts,
    }


def check_final_candidate_role_consistency(
    candidate_skill: str,
    *,
    env: str = "forward_math",
    max_completion_tokens: int = 0,
    format_max_attempts: int = 3,
) -> dict[str, Any]:
    """Semantically review the complete post-Apply candidate Skill."""
    return check_role_consistency(
        candidate_skill,
        env=env,
        max_completion_tokens=max_completion_tokens,
        format_max_attempts=format_max_attempts,
        full_candidate_review_only=True,
    )


def check_structured_atomic_skill_role_consistency(
    candidate_skill: dict[str, Any],
    *,
    env: str = "forward_math",
    max_completion_tokens: int = 0,
    format_max_attempts: int = 3,
) -> dict[str, Any]:
    """Check one canonical two-field mathematics-solver Skill."""
    structure_error = structured_atomic_skill_error(
        candidate_skill
    )
    if structure_error is not None:
        return {
            "passed": False,
            "valid_response": True,
            "issues": [
                {
                    "quoted_text": "",
                    "problem": structure_error,
                    "anchor_ids": [],
                    "anchor_kind": (
                        "structured_atomic_skill_schema"
                    ),
                    "source_start": 0,
                    "source_end": 0,
                    "source_unit_sha256": "",
                }
            ],
            "ignored_issues": [],
            "attempts": [],
            "structure_error": structure_error,
        }

    canonical_skill = _canonical_structured_atomic_skill(
        candidate_skill
    )
    candidate_text = _structured_atomic_skill_json(
        canonical_skill
    )
    anchors = _structured_role_anchor_catalog(
        canonical_skill
    )
    payload, attempts = _call_json_with_format_retry(
        system_prompt=_load_role_system_prompt(
            "role_consistency_check",
            env=env,
        ),
        user_prompt=_build_role_check_user_prompt(
            candidate_text,
            anchors,
        ),
        expected_schema=(
            '{\n'
            '  "decisions": [\n'
            '    {\n'
            '      "anchor_id": "A0001",\n'
            '      "is_problem": true\n'
            '    }\n'
            '  ]\n'
            '}'
        ),
        validator=lambda value: (
            _validate_role_decision_payload(
                value,
                anchors=anchors,
            )
        ),
        stage="role_consistency_check",
        max_completion_tokens=max_completion_tokens,
        format_max_attempts=format_max_attempts,
        validation_feedback_builder=(
            _role_validation_feedback
        ),
    )

    if payload is None:
        return {
            "passed": False,
            "valid_response": False,
            "issues": [],
            "ignored_issues": [],
            "attempts": attempts,
        }

    issues: list[dict[str, Any]] = []
    ignored_issues: list[dict[str, Any]] = []
    for anchor, decision in zip(
        anchors,
        payload["decisions"],
    ):
        record = _role_issue_record(anchor)
        if decision["is_problem"]:
            issues.append(record)
        else:
            ignored_issues.append(record)

    return {
        "passed": not issues,
        "valid_response": True,
        "issues": issues,
        "ignored_issues": ignored_issues,
        "attempts": attempts,
    }


def _build_role_postcheck_user_prompt(
    original_skill: str,
    repaired_skill: str,
    repairs: list[dict[str, Any]],
) -> str:
    changed_units = [
        {
            "unit_id": repair.get("unit_id"),
            "anchor_ids": repair.get(
                "anchor_ids",
                [],
            ),
            "original_unit": repair.get(
                "quoted_text"
            ),
            "repaired_unit": repair.get(
                "replacement_text"
            ),
        }
        for repair in repairs
    ]
    return (
        "## Complete Skill Before Repair (Read-Only Data)\n\n"
        "----- BEGIN ORIGINAL SKILL -----\n"
        f"{_candidate_block(original_skill)}"
        "----- END ORIGINAL SKILL -----\n\n"
        "## Complete Skill After Repair (Read-Only Data)\n\n"
        "----- BEGIN REPAIRED SKILL -----\n"
        f"{_candidate_block(repaired_skill)}"
        "----- END REPAIRED SKILL -----\n\n"
        "## Changed Units for Individual Review\n\n"
        f"{json.dumps(changed_units, ensure_ascii=False, indent=2)}"
        "\n\nIndependently compare the source and repaired text without "
        "modifying either one, and output the review result strictly "
        "according to the JSON protocol in the system prompt."
    )


def _validate_role_postcheck_payload(
    payload: dict,
    *,
    repairs: list[dict[str, Any]],
) -> str | None:
    if set(payload) != _ROLE_POSTCHECK_KEYS:
        return (
            "Role postcheck output must contain exactly "
            "'unit_reviews', 'candidate_role_clean', and "
            "'candidate_violations'."
        )
    if type(payload.get("candidate_role_clean")) is not bool:
        return "'candidate_role_clean' must be a JSON boolean."
    candidate_violations = payload.get(
        "candidate_violations"
    )
    if (
        not isinstance(candidate_violations, list)
        or any(
            not isinstance(item, str)
            or not item.strip()
            for item in candidate_violations
        )
    ):
        return (
            "'candidate_violations' must be an array of "
            "non-empty strings."
        )
    if (
        payload["candidate_role_clean"]
        == bool(candidate_violations)
    ):
        return (
            "candidate_violations must be empty exactly when "
            "candidate_role_clean is true."
        )

    reviews = payload.get("unit_reviews")
    if not isinstance(reviews, list):
        return "'unit_reviews' must be a JSON array."
    if len(reviews) != len(repairs):
        return (
            "Role postcheck must contain exactly one review "
            "for every changed unit."
        )

    for index, (review, repair) in enumerate(
        zip(reviews, repairs),
        start=1,
    ):
        if not isinstance(review, dict):
            return f"Unit review {index} must be a JSON object."
        if set(review) != _ROLE_POSTCHECK_ITEM_KEYS:
            return (
                f"Unit review {index} contains missing or extra fields."
            )
        if review.get("unit_id") != repair.get("unit_id"):
            return (
                "Unit reviews must cover every unit_id once and "
                "in source order."
            )
        for key in _ROLE_POSTCHECK_BOOLEAN_KEYS:
            if type(review.get(key)) is not bool:
                return (
                    f"Unit review {index} field {key!r} must be "
                    "a JSON boolean."
                )
        violations = review.get("violations")
        if (
            not isinstance(violations, list)
            or any(
                not isinstance(item, str)
                or not item.strip()
                for item in violations
            )
        ):
            return (
                f"Unit review {index} violations must be an array "
                "of non-empty strings."
            )
        review_passed = all(
            review[key]
            for key in _ROLE_POSTCHECK_BOOLEAN_KEYS
        )
        if review_passed == bool(violations):
            return (
                f"Unit review {index} violations must be empty "
                "exactly when every review boolean is true."
            )

    return None


def _postcheck_repaired_role_consistency(
    original_skill: str,
    repaired_skill: str,
    repairs: list[dict[str, Any]],
    *,
    env: str,
    max_completion_tokens: int,
    format_max_attempts: int,
) -> dict[str, Any]:
    """Independently review role cleanliness and rewrite fidelity."""
    hard_issues: list[dict[str, Any]] = []
    for repair in repairs:
        status_error = (
            _cognitive_status_promotion_error(
                str(repair.get("quoted_text") or ""),
                str(repair.get("replacement_text") or ""),
            )
        )
        if status_error is None:
            continue

        hard_issues.append({
            "quoted_text": str(
                repair.get("replacement_text") or ""
            ),
            "problem": status_error,
            "anchor_ids": list(
                repair.get("anchor_ids") or []
            ),
            "anchor_kind": (
                "deterministic_cognitive_status_guard"
            ),
            "source_start": int(
                repair.get("source_start") or 0
            ),
            "source_end": int(
                repair.get("source_end") or 0
            ),
            "source_unit_sha256": _sha256_text(
                str(repair.get("quoted_text") or "")
            ),
        })

    if hard_issues:
        return {
            "passed": False,
            "valid_response": True,
            "issues": hard_issues,
            "ignored_issues": [],
            "attempts": [],
            "full_candidate_review": True,
            "review": None,
        }

    payload, attempts = _call_json_with_format_retry(
        system_prompt=_load_role_system_prompt(
            "role_consistency_postcheck",
            env=env,
        ),
        user_prompt=_build_role_postcheck_user_prompt(
            original_skill,
            repaired_skill,
            repairs,
        ),
        expected_schema=(
            '{\n'
            '  "unit_reviews": [\n'
            '    {\n'
            '      "unit_id": "U0001",\n'
            '      "role_clean": true,\n'
            '      "settled_claims_preserved": true,\n'
            '      "polarity_preserved": true,\n'
            '      "conditions_preserved": true,\n'
            '      "no_status_promotion": true,\n'
            '      "no_new_math": true,\n'
            '      "markdown_coherent": true,\n'
            '      "violations": []\n'
            '    }\n'
            '  ],\n'
            '  "candidate_role_clean": true,\n'
            '  "candidate_violations": []\n'
            '}'
        ),
        validator=lambda value: (
            _validate_role_postcheck_payload(
                value,
                repairs=repairs,
            )
        ),
        stage="role_consistency_check",
        max_completion_tokens=max_completion_tokens,
        format_max_attempts=format_max_attempts,
        validation_feedback_builder=(
            _role_validation_feedback
        ),
    )

    if payload is None:
        return {
            "passed": False,
            "valid_response": False,
            "issues": [],
            "ignored_issues": [],
            "attempts": attempts,
            "full_candidate_review": True,
            "review": None,
        }

    unit_reviews_passed = all(
        all(
            review[key]
            for key in _ROLE_POSTCHECK_BOOLEAN_KEYS
        )
        for review in payload["unit_reviews"]
    )
    passed = (
        payload["candidate_role_clean"]
        and unit_reviews_passed
    )
    violations = list(
        payload["candidate_violations"]
    )
    for review in payload["unit_reviews"]:
        violations.extend(review["violations"])

    issues = []
    if not passed:
        issues.append({
            "quoted_text": repaired_skill,
            "problem": (
                "The independent postcheck rejected the rewrite: "
                + "; ".join(violations)
            ),
            "anchor_ids": ["R0001"],
            "anchor_kind": "full_candidate_review",
            "source_start": 0,
            "source_end": len(repaired_skill),
            "source_unit_sha256": _sha256_text(
                repaired_skill
            ),
        })

    return {
        "passed": passed,
        "valid_response": True,
        "issues": issues,
        "ignored_issues": [],
        "attempts": attempts,
        "full_candidate_review": True,
        "review": payload,
    }


def _repair_attempts_exhausted_only_safety_checks(
    attempts: list[dict[str, Any]],
) -> bool:
    safety_markers = (
        "immutable",
        "mathematical expression or number",
        "protected logic or epistemic-status",
        "changed markdown structure",
        "role-contamination marker",
        "process-history label",
        "must use no_commit",
        "must remain one source line",
        "control characters or unicode line separators",
    )
    if not attempts:
        return False

    for attempt in attempts:
        if attempt.get("error"):
            return False
        validation_error = str(
            attempt.get("validation_error")
            or ""
        ).casefold()
        if not validation_error:
            return False
        if not any(
            marker in validation_error
            for marker in safety_markers
        ):
            return False

    return True


def repair_role_consistency(
    candidate_skill: str,
    issues: list[dict[str, Any]],
    *,
    env: str = "forward_math",
    max_completion_tokens: int = 0,
    format_max_attempts: int = 3,
    full_candidate_review_only: bool = False,
) -> dict[str, Any]:
    """Rewrite complete role-contaminated source units with strict guards."""
    anchors, anchor_error = (
        _selected_role_anchors(
            candidate_skill,
            issues,
            full_candidate_review_only=(
                full_candidate_review_only
            ),
        )
    )

    if anchor_error is not None:
        return {
            "valid_response": False,
            "repairable": False,
            "repairs": [],
            "repaired_skill": candidate_skill,
            "apply_report": [],
            "attempts": [],
            "error": anchor_error,
        }

    if not anchors:
        return {
            "valid_response": False,
            "repairable": False,
            "repairs": [],
            "repaired_skill": candidate_skill,
            "apply_report": [],
            "attempts": [],
            "error": (
                "At least one current role anchor is required."
            ),
        }

    units, unit_error = _build_role_repair_units(
        candidate_skill,
        anchors,
    )
    if unit_error is not None:
        return {
            "valid_response": True,
            "repairable": False,
            "repairs": [],
            "repaired_skill": candidate_skill,
            "apply_report": [],
            "attempts": [],
            "error": unit_error,
        }

    payload, attempts = (
        _call_json_with_format_retry(
            system_prompt=_load_role_system_prompt(
                "role_consistency_repair",
                env=env,
            ),
            user_prompt=_build_role_repair_user_prompt(
                candidate_skill,
                units,
            ),
            expected_schema=(
                '{\n'
                '  "repairs": [\n'
                '    {\n'
                '      "unit_id": "U0001",\n'
                '      "decision": "replace",\n'
                '      "replacement_text": '
                '"<masked solver-facing unit>"\n'
                '    }\n'
                '  ]\n'
                '}'
            ),
            validator=lambda value: (
                _validate_role_rewrite_payload(
                    value,
                    units=units,
                )
            ),
            pre_decode_validator=lambda value: (
                _role_rewrite_wire_error(
                    value,
                    units=units,
                )
            ),
            payload_decoder=lambda value: (
                _decode_role_rewrite_wire(
                    value,
                    units=units,
                )
            ),
            stage="role_consistency_repair",
            max_completion_tokens=max_completion_tokens,
            format_max_attempts=format_max_attempts,
            validation_feedback_builder=(
                _role_validation_feedback
            ),
        )
    )

    if payload is None:
        if _repair_attempts_exhausted_only_safety_checks(
            attempts
        ):
            return {
                "valid_response": True,
                "repairable": False,
                "repairs": [
                    {
                        "unit_id": unit["unit_id"],
                        "anchor_ids": unit[
                            "anchor_ids"
                        ],
                        "decision": "no_commit",
                        "quoted_text": unit[
                            "source_text"
                        ],
                        "replacement_text": None,
                        "source_start": unit[
                            "source_start"
                        ],
                        "source_end": unit[
                            "source_end"
                        ],
                    }
                    for unit in units
                ],
                "repaired_skill": candidate_skill,
                "apply_report": [],
                "attempts": attempts,
                "error": (
                    "safety_validation_exhausted_no_commit"
                ),
            }
        return {
            "valid_response": False,
            "repairable": False,
            "repairs": [],
            "repaired_skill": candidate_skill,
            "apply_report": [],
            "attempts": attempts,
        }

    if any(
        repair["decision"] == "no_commit"
        for repair in payload["repairs"]
    ):
        return {
            "valid_response": True,
            "repairable": False,
            "repairs": [
                {
                    "unit_id": unit["unit_id"],
                    "anchor_ids": unit[
                        "anchor_ids"
                    ],
                    "decision": repair[
                        "decision"
                    ],
                    "quoted_text": unit[
                        "source_text"
                    ],
                    "replacement_text": None,
                    "source_start": unit[
                        "source_start"
                    ],
                    "source_end": unit[
                        "source_end"
                    ],
                }
                for unit, repair in zip(
                    units,
                    payload["repairs"],
                )
            ],
            "repaired_skill": candidate_skill,
            "apply_report": [],
            "attempts": attempts,
        }

    applied, apply_error = (
        _apply_role_unit_rewrites(
            candidate_skill,
            units,
            payload["repairs"],
        )
    )

    if applied is None:
        return {
            "valid_response": False,
            "repairable": False,
            "repairs": [],
            "repaired_skill": candidate_skill,
            "apply_report": [],
            "attempts": attempts,
            "error": apply_error,
        }

    return {
        "valid_response": True,
        "repairable": True,
        "repairs": applied["repairs"],
        "repaired_skill": applied[
            "repaired_skill"
        ],
        "apply_report": applied[
            "apply_report"
        ],
        "attempts": attempts,
    }


def _build_structured_role_repair_user_prompt(
    candidate_skill: dict[str, Any],
    issues: list[dict[str, Any]],
) -> str:
    public_issues = [
        {
            "anchor_ids": list(
                issue.get("anchor_ids") or []
            ),
            "anchor_kind": str(
                issue.get("anchor_kind") or ""
            ),
            "quoted_text": str(
                issue.get("quoted_text") or ""
            ),
            "problem": str(
                issue.get("problem") or ""
            ),
        }
        for issue in issues
    ]
    return (
        "## Repair Mode\n\n"
        "structured_atomic_skill\n\n"
        "## Complete Atomic Skill Before Repair (Read-Only Data)\n\n"
        "----- BEGIN ATOMIC SKILL -----\n"
        f"{_structured_atomic_skill_json(candidate_skill)}\n"
        "----- END ATOMIC SKILL -----\n\n"
        "## Role Issues to Repair\n\n"
        f"{json.dumps(public_issues, ensure_ascii=False, indent=2)}"
        "\n\nRepair this one atomic Skill as one complete two-field "
        "object. Preserve every already role-consistent field value "
        "as literally as possible. Follow the structured_atomic_skill "
        "output protocol in the system prompt."
    )


def _validate_structured_role_repair_payload(
    payload: dict,
    *,
    original_skill: dict[str, Any],
) -> str | None:
    if set(payload) != _STRUCTURED_ROLE_REPAIR_KEYS:
        return (
            "Structured role repair output must contain exactly "
            "'decision' and 'replacement_skill'."
        )

    decision = payload.get("decision")
    if decision not in _STRUCTURED_ROLE_REPAIR_DECISIONS:
        return "'decision' must be 'replace' or 'no_commit'."

    replacement = payload.get("replacement_skill")
    if decision == "no_commit":
        if replacement is not None:
            return (
                "replacement_skill must be null when decision is "
                "'no_commit'."
            )
        return None

    structure_error = structured_atomic_skill_error(
        replacement
    )
    if structure_error is not None:
        return (
            "replacement_skill is invalid: "
            + structure_error
        )

    canonical_original = _canonical_structured_atomic_skill(
        original_skill
    )
    canonical_replacement = _canonical_structured_atomic_skill(
        replacement
    )
    if canonical_replacement == canonical_original:
        return (
            "A 'replace' decision must change the atomic Skill; use "
            "'no_commit' when no safe change is available."
        )

    original_text = _structured_atomic_skill_json(
        canonical_original
    )
    replacement_text = _structured_atomic_skill_json(
        canonical_replacement
    )
    if (
        _protected_math_and_number_tokens(
            replacement_text
        )
        != _protected_math_and_number_tokens(
            original_text
        )
    ):
        return (
            "Structured role repair changed a mathematical "
            "expression or number."
        )

    if not _is_ordered_subsequence(
        _protected_logic_tokens(original_text),
        _protected_logic_tokens(replacement_text),
    ):
        return (
            "Structured role repair removed or reordered an existing "
            "protected logic or epistemic-status expression."
        )

    status_error = _cognitive_status_promotion_error(
        original_text,
        replacement_text,
    )
    if status_error is not None:
        return status_error

    if _contains_explicit_role_contamination(
        replacement_text
    ):
        return (
            "Structured role repair left an explicit "
            "role-contamination marker."
        )

    if _ROLE_PROCESS_TRACE_RE.search(replacement_text):
        return (
            "Structured role repair left a process-history label."
        )

    return None


def repair_structured_atomic_skill_role_consistency(
    candidate_skill: dict[str, Any],
    issues: list[dict[str, Any]],
    *,
    env: str = "forward_math",
    max_completion_tokens: int = 0,
    format_max_attempts: int = 3,
) -> dict[str, Any]:
    """Repair one complete structured atomic Skill or abstain."""
    structure_error = structured_atomic_skill_error(
        candidate_skill
    )
    if structure_error is not None:
        return {
            "valid_response": True,
            "repairable": False,
            "replacement_skill": copy.deepcopy(
                candidate_skill
            ),
            "attempts": [],
            "error": structure_error,
        }

    original_skill = _canonical_structured_atomic_skill(
        candidate_skill
    )
    payload, attempts = _call_json_with_format_retry(
        system_prompt=_load_role_system_prompt(
            "role_consistency_repair",
            env=env,
        ),
        user_prompt=_build_structured_role_repair_user_prompt(
            original_skill,
            issues,
        ),
        expected_schema=(
            '{\n'
            '  "decision": "replace",\n'
            '  "replacement_skill": {\n'
            '    "title": "...",\n'
            '    "solver_guidance": "..."\n'
            '  }\n'
            '}'
        ),
        validator=lambda value: (
            _validate_structured_role_repair_payload(
                value,
                original_skill=original_skill,
            )
        ),
        stage="role_consistency_repair",
        max_completion_tokens=max_completion_tokens,
        format_max_attempts=format_max_attempts,
        validation_feedback_builder=(
            _role_validation_feedback
        ),
    )

    if payload is None:
        return {
            "valid_response": False,
            "repairable": False,
            "replacement_skill": copy.deepcopy(original_skill),
            "attempts": attempts,
        }

    if payload["decision"] == "no_commit":
        return {
            "valid_response": True,
            "repairable": False,
            "replacement_skill": copy.deepcopy(original_skill),
            "attempts": attempts,
            "decision": "no_commit",
        }

    replacement_skill = _canonical_structured_atomic_skill(
        payload["replacement_skill"]
    )
    return {
        "valid_response": True,
        "repairable": True,
        "replacement_skill": replacement_skill,
        "attempts": attempts,
        "decision": "replace",
    }


def enforce_structured_atomic_skill_role_consistency(
    candidate_skill: dict[str, Any],
    *,
    env: str = "forward_math",
    max_completion_tokens: int = 0,
    format_max_attempts: int = 3,
) -> dict[str, Any]:
    """Check, repair, recheck, and postcheck one solver Skill."""
    structure_error = structured_atomic_skill_error(
        candidate_skill
    )
    original_skill = (
        copy.deepcopy(candidate_skill)
        if structure_error is not None
        else _canonical_structured_atomic_skill(
            candidate_skill
        )
    )
    initial_check = check_structured_atomic_skill_role_consistency(
        original_skill,
        env=env,
        max_completion_tokens=max_completion_tokens,
        format_max_attempts=format_max_attempts,
    )
    result: dict[str, Any] = {
        "passed": False,
        "status": "",
        "candidate_skill": copy.deepcopy(original_skill),
        "changed": False,
        "initial_check": initial_check,
        "repair_rounds": [],
        "final_check": None,
    }

    if not initial_check["valid_response"]:
        result["status"] = "check_format_failure"
        return result

    if initial_check["passed"]:
        result.update({
            "passed": True,
            "status": "pass_without_repair",
        })
        return result

    if initial_check.get("structure_error"):
        result.update({
            "status": "unrepairable_no_commit",
            "final_check": initial_check,
        })
        return result

    repair_result = repair_structured_atomic_skill_role_consistency(
        original_skill,
        initial_check["issues"],
        env=env,
        max_completion_tokens=max_completion_tokens,
        format_max_attempts=format_max_attempts,
    )
    repair_record = {
        key: copy.deepcopy(value)
        for key, value in repair_result.items()
        if key != "replacement_skill"
    }

    if not repair_result["valid_response"]:
        result.update({
            "status": "repair_format_failure",
            "repair_rounds": [{
                "round": 1,
                "repair": repair_record,
                "check": None,
            }],
            "final_check": initial_check,
        })
        return result

    if not repair_result["repairable"]:
        result.update({
            "status": "unrepairable_no_commit",
            "repair_rounds": [{
                "round": 1,
                "repair": repair_record,
                "check": None,
            }],
            "final_check": initial_check,
        })
        return result

    repaired_skill = repair_result["replacement_skill"]
    scanner_recheck = check_structured_atomic_skill_role_consistency(
        repaired_skill,
        env=env,
        max_completion_tokens=max_completion_tokens,
        format_max_attempts=format_max_attempts,
    )
    recheck = scanner_recheck
    original_text = _structured_atomic_skill_json(original_skill)
    repaired_text = _structured_atomic_skill_json(repaired_skill)
    repair_evidence = [{
        "unit_id": "S0001",
        "anchor_ids": [
            anchor_id
            for issue in initial_check["issues"]
            for anchor_id in (issue.get("anchor_ids") or [])
        ],
        "quoted_text": original_text,
        "replacement_text": repaired_text,
        "source_start": 0,
        "source_end": len(original_text),
    }]
    if (
        scanner_recheck["valid_response"]
        and scanner_recheck["passed"]
    ):
        recheck = _postcheck_repaired_role_consistency(
            original_text,
            repaired_text,
            repair_evidence,
            env=env,
            max_completion_tokens=max_completion_tokens,
            format_max_attempts=format_max_attempts,
        )

    round_record = {
        "round": 1,
        "repair": repair_record,
        "check": recheck,
    }
    if not recheck["valid_response"]:
        result.update({
            "status": "recheck_format_failure",
            "repair_rounds": [round_record],
            "final_check": recheck,
        })
        return result

    if not recheck["passed"]:
        result.update({
            "status": "fail_after_repair",
            "repair_rounds": [round_record],
            "final_check": recheck,
        })
        return result

    result.update({
        "passed": True,
        "status": "pass_after_repair",
        "candidate_skill": copy.deepcopy(repaired_skill),
        "changed": repaired_skill != original_skill,
        "repair_rounds": [round_record],
        "final_check": recheck,
    })
    return result


def enforce_role_consistency(
    candidate_skill: str,
    *,
    env: str = "forward_math",
    max_completion_tokens: int = 0,
    format_max_attempts: int = 3,
    repair_max_rounds: int = 1,
    full_candidate_review_only: bool = False,
) -> dict[str, Any]:
    """Check, optionally repair once, and recheck."""
    if repair_max_rounds != 1:
        raise ValueError(
            "repair_max_rounds must be exactly 1, "
            f"got {repair_max_rounds}."
        )

    original_skill = candidate_skill
    working_skill = candidate_skill

    initial_check = check_role_consistency(
        working_skill,
        env=env,
        max_completion_tokens=max_completion_tokens,
        format_max_attempts=format_max_attempts,
        full_candidate_review_only=(
            full_candidate_review_only
        ),
    )

    result: dict[str, Any] = {
        "passed": False,
        "status": "",
        "candidate_skill": working_skill,
        "changed": False,
        "initial_check": initial_check,
        "repair_rounds": [],
        "final_check": None,
    }

    if not initial_check["valid_response"]:
        result["status"] = (
            "check_format_failure"
        )
        return result

    if initial_check["passed"]:
        result.update({
            "passed": True,
            "status": "pass_without_repair",
        })
        return result

    latest_check = initial_check
    round_records: list[dict[str, Any]] = []

    for round_index in range(
        repair_max_rounds
    ):
        repair_result = repair_role_consistency(
            working_skill,
            latest_check["issues"],
            env=env,
            max_completion_tokens=max_completion_tokens,
            format_max_attempts=format_max_attempts,
            full_candidate_review_only=(
                full_candidate_review_only
            ),
        )

        repair_record = {
            key: value
            for key, value
            in repair_result.items()
            if key != "repaired_skill"
        }

        if not repair_result["valid_response"]:
            round_records.append({
                "round": round_index + 1,
                "repair": repair_record,
                "check": None,
            })
            result.update({
                "status": "repair_format_failure",
                "candidate_skill": original_skill,
                "changed": False,
                "repair_rounds": round_records,
                "final_check": latest_check,
            })
            return result

        if not repair_result[
            "repairable"
        ]:
            round_records.append({
                "round": round_index + 1,
                "repair": repair_record,
                "check": None,
            })
            result.update({
                "status": "unrepairable_no_commit",
                "candidate_skill": original_skill,
                "changed": False,
                "repair_rounds": round_records,
                "final_check": latest_check,
            })
            return result

        working_skill = repair_result[
            "repaired_skill"
        ]

        scanner_recheck = check_role_consistency(
            working_skill,
            env=env,
            max_completion_tokens=max_completion_tokens,
            format_max_attempts=format_max_attempts,
            full_candidate_review_only=(
                full_candidate_review_only
            ),
        )

        recheck = scanner_recheck
        if (
            scanner_recheck["valid_response"]
            and scanner_recheck["passed"]
        ):
            recheck = (
                _postcheck_repaired_role_consistency(
                    original_skill,
                    working_skill,
                    repair_result["repairs"],
                    env=env,
                    max_completion_tokens=max_completion_tokens,
                    format_max_attempts=format_max_attempts,
                )
            )

        round_records.append({
            "round": round_index + 1,
            "repair": repair_record,
            "check": recheck,
        })

        if not recheck["valid_response"]:
            result.update({
                "status": "recheck_format_failure",
                "candidate_skill": original_skill,
                "changed": False,
                "repair_rounds": round_records,
                "final_check": recheck,
            })
            return result

        if recheck["passed"]:
            result.update({
                "passed": True,
                "status": "pass_after_repair",
                "candidate_skill": working_skill,
                "changed": (
                    working_skill != original_skill
                ),
                "repair_rounds": round_records,
                "final_check": recheck,
            })
            return result

        latest_check = recheck
        if recheck.get(
            "full_candidate_review"
        ):
            break

    result.update({
        "status": "fail_after_repair",
        "candidate_skill": original_skill,
        "changed": False,
        "repair_rounds": round_records,
        "final_check": latest_check,
    })

    return result


def enforce_subskill_role_consistency(
    candidate_skill: str,
    *,
    env: str = "forward_math",
    max_completion_tokens: int = 0,
    format_max_attempts: int = 3,
) -> dict[str, Any]:
    """Check one solver subskill locally, then review the complete subskill."""
    original_skill = candidate_skill

    local_result = enforce_role_consistency(
        candidate_skill,
        env=env,
        max_completion_tokens=max_completion_tokens,
        format_max_attempts=format_max_attempts,
        repair_max_rounds=1,
    )
    local_record = {
        key: copy.deepcopy(value)
        for key, value in local_result.items()
        if key != "candidate_skill"
    }

    if not local_result.get("passed"):
        result = copy.deepcopy(local_result)
        result["local_role_consistency"] = (
            local_record
        )
        result[
            "full_subskill_role_consistency"
        ] = None
        return result

    locally_screened_skill = local_result[
        "candidate_skill"
    ]
    full_result = enforce_role_consistency(
        locally_screened_skill,
        env=env,
        max_completion_tokens=max_completion_tokens,
        format_max_attempts=format_max_attempts,
        repair_max_rounds=1,
        full_candidate_review_only=True,
    )
    full_record = {
        key: copy.deepcopy(value)
        for key, value in full_result.items()
        if key != "candidate_skill"
    }

    result = copy.deepcopy(full_result)
    result["local_role_consistency"] = (
        local_record
    )
    result[
        "full_subskill_role_consistency"
    ] = full_record

    if not full_result.get("passed"):
        return result

    final_skill = full_result[
        "candidate_skill"
    ]
    changed = final_skill != original_skill
    result.update({
        "passed": True,
        "status": (
            "pass_after_repair"
            if changed
            else "pass_without_repair"
        ),
        "candidate_skill": final_skill,
        "changed": changed,
    })
    return result


def enforce_ranked_patch_consistency(
    ranked_patch: dict,
    *,
    item_enforcer: Callable[
        [str, dict[str, Any]],
        dict[str, Any],
    ],
    result_field: str,
    status_prefix: str,
    gate_label: str,
    drop_failed_items: bool = True,
) -> dict[str, Any]:
    """Apply one semantic gate to every non-delete ranked atomic edit.

    ``item_enforcer`` receives the edit content and a deep copy of the
    complete edit metadata. It must return the same result contract used
    by ``enforce_role_consistency``.

    Semantic failure after repair drops only that subskill when
    ``drop_failed_items`` is true. Otherwise it fails closed and returns
    the original ranked patch without committing a partial result.
    """

    if not callable(
        item_enforcer
    ):
        raise TypeError(
            "item_enforcer must be callable."
        )

    if (
        not isinstance(result_field, str)
        or not result_field.strip()
    ):
        raise ValueError(
            "result_field must be a non-empty string."
        )

    if (
        not isinstance(status_prefix, str)
        or not status_prefix.strip()
    ):
        raise ValueError(
            "status_prefix must be a non-empty string."
        )

    if (
        not isinstance(gate_label, str)
        or not gate_label.strip()
    ):
        raise ValueError(
            "gate_label must be a non-empty string."
        )

    original_patch = copy.deepcopy(
        ranked_patch
    )

    invalid_result_status = (
        f"invalid_{result_field}_result"
    )
    exception_status = (
        f"{status_prefix}_exception"
    )
    technical_failure_status = (
        f"{status_prefix}_technical_failure"
    )

    def _empty_summary(
        input_count: int,
    ) -> dict[str, int | bool]:
        return {
            "input_count": input_count,
            "processed_count": 0,
            "output_count": input_count,
            "passed_without_repair_count": 0,
            "repaired_count": 0,
            "dropped_count": 0,
            "delete_passthrough_count": 0,
            "technical_failure_count": 0,
            "committed": False,
        }

    if not isinstance(
        ranked_patch,
        dict,
    ):
        return {
            "passed": False,
            "status": "invalid_ranked_patch",
            "changed": False,
            "filtered_patch": original_patch,
            "summary": _empty_summary(0),
            "results": [],
            "error": "ranked_patch must be a dictionary.",
        }

    edits = ranked_patch.get(
        "edits"
    )

    if not isinstance(edits, list):
        return {
            "passed": False,
            "status": "invalid_ranked_patch",
            "changed": False,
            "filtered_patch": original_patch,
            "summary": _empty_summary(0),
            "results": [],
            "error": "ranked_patch.edits must be a list.",
        }

    summary = _empty_summary(
        len(edits)
    )
    results: list[dict[str, Any]] = []
    filtered_edits: list[dict] = []

    def _technical_failure(
        *,
        status: str,
        error: str,
    ) -> dict[str, Any]:
        # A technical failure returns the original patch, so mutations
        # accumulated before the failure were only tentative.
        summary["repaired_count"] = 0
        summary["dropped_count"] = 0
        summary["processed_count"] = len(
            results
        )
        summary["technical_failure_count"] = 1
        summary["output_count"] = len(
            edits
        )
        summary["committed"] = False
        return {
            "passed": False,
            "status": status,
            "changed": False,
            "filtered_patch": original_patch,
            "summary": summary,
            "results": results,
            "error": error,
        }

    for index, edit in enumerate(
        edits
    ):
        if not isinstance(edit, dict):
            results.append({
                "index": index,
                "op": "",
                "action": "technical_failure",
                "status": "invalid_ranked_edit",
            })
            return _technical_failure(
                status="invalid_ranked_edit",
                error=(
                    f"ranked_patch.edits[{index}] must be a dictionary."
                ),
            )

        result_metadata = {
            key: copy.deepcopy(edit[key])
            for key in _SUBSKILL_METADATA_KEYS
            if key in edit
        }

        op = edit.get(
            "op"
        )

        if op == "delete":
            filtered_edits.append(
                copy.deepcopy(edit)
            )
            summary[
                "delete_passthrough_count"
            ] += 1
            results.append({
                "index": index,
                "op": op,
                "action": "delete_passthrough",
                "status": "delete_passthrough",
                "changed": False,
                **result_metadata,
            })
            continue

        if op not in {
            "append",
            "insert_after",
            "replace",
        }:
            results.append({
                "index": index,
                "op": str(op or ""),
                "action": "technical_failure",
                "status": "invalid_ranked_edit",
                **result_metadata,
            })
            return _technical_failure(
                status="invalid_ranked_edit",
                error=(
                    f"ranked_patch.edits[{index}].op is invalid: {op!r}."
                ),
            )

        content = edit.get(
            "content"
        )

        if not isinstance(content, str) or not content.strip():
            results.append({
                "index": index,
                "op": op,
                "action": "technical_failure",
                "status": "invalid_ranked_edit",
                **result_metadata,
            })
            return _technical_failure(
                status="invalid_ranked_edit",
                error=(
                    f"ranked_patch.edits[{index}].content must be "
                    "a non-empty string."
                ),
            )

        try:
            gate_result = item_enforcer(
                content,
                copy.deepcopy(edit),
            )
        except Exception as error:  # noqa: BLE001
            results.append({
                "index": index,
                "op": op,
                "action": "technical_failure",
                "status": exception_status,
                "changed": False,
                "error": f"{type(error).__name__}: {error}",
                **result_metadata,
            })
            return _technical_failure(
                status=exception_status,
                error=(
                    f"{gate_label} processing raised for "
                    f"ranked_patch.edits[{index}]: "
                    f"{type(error).__name__}: {error}"
                ),
            )

        if not isinstance(gate_result, dict):
            results.append({
                "index": index,
                "op": op,
                "action": "technical_failure",
                "status": invalid_result_status,
                "changed": False,
                **result_metadata,
            })
            return _technical_failure(
                status=invalid_result_status,
                error=(
                    f"{gate_label} item_enforcer did not return a "
                    "dictionary "
                    f"for ranked_patch.edits[{index}]."
                ),
            )

        gate_record = {
            key: value
            for key, value in gate_result.items()
            if key != "candidate_skill"
        }
        status = str(
            gate_result.get(
                "status",
                "",
            )
        )
        passed = gate_result.get(
            "passed"
        )

        record: dict[str, Any] = {
            "index": index,
            "op": op,
            "action": "",
            "status": status,
            "changed": False,
            result_field: gate_record,
            **result_metadata,
        }

        if type(passed) is not bool or not status:
            record["action"] = (
                "technical_failure"
            )
            record["status"] = (
                invalid_result_status
            )
            results.append(
                record
            )
            return _technical_failure(
                status=invalid_result_status,
                error=(
                    f"{gate_label} item_enforcer returned an invalid "
                    "result "
                    f"for ranked_patch.edits[{index}]."
                ),
            )

        if passed:
            if status not in {
                "pass_without_repair",
                "pass_after_repair",
            }:
                record["action"] = (
                    "technical_failure"
                )
                record["status"] = (
                    invalid_result_status
                )
                results.append(
                    record
                )
                return _technical_failure(
                    status=invalid_result_status,
                    error=(
                        f"A passing {gate_label} result has an "
                        f"unexpected status {status!r}."
                    ),
                )

            repaired_content = gate_result.get(
                "candidate_skill"
            )

            if not isinstance(
                repaired_content,
                str,
            ):
                record["action"] = (
                    "technical_failure"
                )
                record["status"] = (
                    invalid_result_status
                )
                results.append(
                    record
                )
                return _technical_failure(
                    status=invalid_result_status,
                    error=(
                        f"A passing {gate_label} result must contain "
                        "a string candidate_skill."
                    ),
                )

            if not repaired_content.strip():
                record["action"] = (
                    "no_commit"
                )
                record["changed"] = False
                results.append(
                    record
                )
                return _technical_failure(
                    status=(
                        f"{status_prefix}_no_commit"
                    ),
                    error=(
                        f"{gate_label} produced empty content "
                        f"for ranked_patch.edits[{index}]; no "
                        "screening changes were committed."
                    ),
                )

            filtered_edit = copy.deepcopy(
                edit
            )
            filtered_edit["content"] = (
                repaired_content
            )
            filtered_edits.append(
                filtered_edit
            )

            if repaired_content == content:
                record["action"] = (
                    "passed_without_repair"
                )
                summary[
                    "passed_without_repair_count"
                ] += 1
            else:
                record["action"] = (
                    "repaired"
                )
                record["changed"] = True
                summary[
                    "repaired_count"
                ] += 1

            results.append(
                record
            )
            continue

        if (
            status in {
                "fail_after_repair",
                "unrepairable_no_commit",
            }
            and drop_failed_items
        ):
            record["action"] = "dropped"
            record["changed"] = True
            summary[
                "dropped_count"
            ] += 1
            results.append(
                record
            )
            continue

        if (
            not drop_failed_items
            and status in {
                "fail_after_repair",
                "unrepairable_no_commit",
            }
        ):
            record["action"] = "no_commit"
            record["changed"] = False
            results.append(
                record
            )
            return _technical_failure(
                status=(
                    f"{status_prefix}_no_commit"
                ),
                error=(
                    f"{gate_label} could not safely repair "
                    f"ranked_patch.edits[{index}]; no screening "
                    "changes were committed."
                ),
            )

        record["action"] = (
            "technical_failure"
        )
        results.append(
            record
        )
        return _technical_failure(
            status=technical_failure_status,
            error=(
                f"{gate_label} processing failed closed for "
                f"ranked_patch.edits[{index}] with status {status!r}."
            ),
        )

    filtered_patch = copy.deepcopy(
        ranked_patch
    )
    filtered_patch["edits"] = (
        filtered_edits
    )
    changed = filtered_patch != ranked_patch
    summary["processed_count"] = len(
        results
    )
    summary["output_count"] = len(
        filtered_edits
    )
    summary["committed"] = True

    return {
        "passed": True,
        "status": (
            "pass_with_changes"
            if changed
            else "pass_without_changes"
        ),
        "changed": changed,
        "filtered_patch": filtered_patch,
        "summary": summary,
        "results": results,
    }


def enforce_ranked_patch_role_consistency(
    ranked_patch: dict,
    *,
    env: str = "forward_math",
    max_completion_tokens: int = 0,
    format_max_attempts: int = 3,
    full_candidate_review: bool = False,
) -> dict[str, Any]:
    """Filter ranked edits through one role-consistency repair round.

    Every non-delete edit is treated as one candidate subskill.  Only its
    ``content`` field may be replaced; all other edit metadata and the
    original edit order are preserved. A subskill that cannot be safely
    repaired is dropped without rejecting the other subskills. Delete edits
    pass through unchanged because they do not add student-facing Skill
    content.

    Model/format/application failures fail closed and return the original
    patch rather than a partially filtered patch.  The caller must check
    ``passed`` before using ``filtered_patch``.
    """

    def _enforce_item(
        content: str,
        _edit: dict[str, Any],
    ) -> dict[str, Any]:
        if full_candidate_review:
            return enforce_subskill_role_consistency(
                content,
                env=env,
                max_completion_tokens=(
                    max_completion_tokens
                ),
                format_max_attempts=(
                    format_max_attempts
                ),
            )

        return enforce_role_consistency(
            content,
            env=env,
            max_completion_tokens=max_completion_tokens,
            format_max_attempts=format_max_attempts,
            repair_max_rounds=1,
        )

    return enforce_ranked_patch_consistency(
        ranked_patch,
        item_enforcer=_enforce_item,
        result_field="role_consistency",
        status_prefix=(
            "subskill_role_consistency"
        ),
        gate_label="Role-consistency",
        drop_failed_items=True,
    )
