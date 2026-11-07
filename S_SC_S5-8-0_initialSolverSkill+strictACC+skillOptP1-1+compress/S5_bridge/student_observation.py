from __future__ import annotations

from typing import Any

from S5_bridge.forward_evidence import ForwardRolloutEvidence


SCHEMA_VERSION = "student_observation_v1"


def export_student_observation(
    evidence: ForwardRolloutEvidence,
) -> dict[str, Any]:
    """只读导出已有证据，供 SPQ15 等外部观察插件使用。

    兼容性约定：
    1. 非必要不得删除、重命名已有字段，或改变字段类型与含义。
    2. 扩展优先增加可选字段；不兼容修改须另开版本并保留 v1。
    3. passed 的 None 表示未完成验证，不等于 False（验证失败）。

    行为约定：
    1. 不修改输入、不写文件、不调用模型、不重新验证/纠错/训练。
    2. 不改变 S5 原入口、默认参数、提示词或实验流程。
    3. initial_response 仅为 S5 保存的原答，不得替换为修正回答。
       上传原始文本由调用方另存；S5 原 runner 可能去掉首尾空白。
    4. 整体异常可能来自后续轮次，不可冒充首次验证异常。
    """
    first = evidence.steps[0] if evidence.steps else None
    rollout_reason = evidence.verification_unavailable_reason or ""
    first_reason = (
        (rollout_reason or "missing_first_verification")
        if first is None
        else rollout_reason if len(evidence.steps) == 1 else ""
    )
    if first is not None and type(first.r_f) is not bool:
        raise TypeError("First verification verdict must be bool.")
    passed = None if first_reason else first.r_f
    status = "unavailable" if passed is None else "passed" if passed else "failed"
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": evidence.sample_id,
        "problem": evidence.q,
        "initial_response": evidence.y,
        "initial_answer": evidence.a,
        "skill_version": evidence.skill_version,
        "first_verification": {
            "status": status,
            "passed": passed,
            "verification_flow": first.V_f if first else "",
            "execution_record": first.R_f if first else "",
            "feedback": first.F_f if first else None,
            "unavailable_reason": first_reason,
        },
        "rollout_verification_unavailable_reason": rollout_reason,
    }
