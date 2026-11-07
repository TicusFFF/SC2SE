"""Offline S5 reporting only: never generate answers or select/update skills.

ACC means standard-answer accuracy. Forward pass rate is a separate metric.
Keep input/output skill versions distinct; a round evaluates its input library
before producing its output library. Chat reports should read this final report.
用户约定：“最终输出表格”使用 R1→R2→R3→R4 横向汇总，每个数据集独立一表。
保留逐轮明细；只展示已有记录，不为填表重新评估或选择技能。
"""
from __future__ import annotations

import json
from pathlib import Path


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def _skills(value):
    if not isinstance(value, dict) or not isinstance(value.get("skills"), list):
        return None
    rows = value["skills"]
    if any(not isinstance(x, dict) or not all(k in x for k in
           ("skill_id", "title", "solver_guidance")) for x in rows):
        return None
    if len({x["skill_id"] for x in rows}) != len(rows):
        return None
    return rows


def _identity(value):
    rows = _skills(value)
    if rows is None:
        return None
    # Match exact ordered solver input, including evidence; do not match by count.
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _acc(m, key):
    if m.get("status") == "pending_real_acc":
        return "待补判分"
    if m.get("real_acc_status") != "ok" or m.get("status", "complete") != "complete":
        return "未评估"
    n, correct = m.get("real_n"), m.get(key)
    if not isinstance(n, int) or n <= 0 or not isinstance(correct, int) or not 0 <= correct <= n:
        return "未评估"
    return f"{correct / n:.1%}（{correct}/{n}）"


def _gain(m):
    if m.get("real_acc_status") != "ok" or _acc(m, "real_initial_correct") == "未评估" or _acc(m, "real_final_correct") == "未评估":
        return "—"
    delta = m["real_final_correct"] - m["real_initial_correct"]
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
    return f"{arrow}{delta:+d}题（{delta / m['real_n'] * 100:+.1f}个百分点）"


def _pct(x):
    return f"{x:.1%}" if isinstance(x, (int, float)) else "未记录"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _round_overview(root, summary, method, metric_name, history, round_rows,
                    best, matches, final_meta, final_m):
    """One dataset per horizontal table; reuse exact-match ACC observations."""
    cfg = summary.get("config", {})
    epochs = max([int(cfg.get("num_epochs") or 0), 4] + [row["epoch"] for row in round_rows])
    # A non-four-round configuration must keep its actual planned round count.
    if cfg.get("num_epochs"):
        epochs = max(int(cfg["num_epochs"]), max((row["epoch"] for row in round_rows), default=0))
    label = "→".join(f"R{i}" for i in range(1, epochs + 1))

    def sequence(key):
        values = []
        for epoch in range(1, epochs + 1):
            rows = [row for row in round_rows if row["epoch"] == epoch]
            if not rows:
                values.append("—")
            elif len(rows) == 1:
                values.append(rows[0][key])
            else:
                # Multiple updates in one epoch are not independent full-set ACCs.
                values.append("[" + "；".join(f"步骤{r['step']}:{r[key]}" for r in rows) + "]")
        return "→".join(values)

    best_step = summary.get("best_step")
    best_record = next((r for r in history if r.get("step") == best_step), None)
    best_round = (f"R{best_record.get('epoch', best_step)}（步骤{best_step}）" if best_record
                  else "R0（空库或初始库，见快照）" if best_step == 0 else "未记录")
    rows = _skills(best)
    best_count = f"{len(rows)} 条" if rows is not None else "未记录"
    best_acc = []
    for _, m, source, _ in matches:
        first, last = _acc(m, "real_initial_correct"), _acc(m, "real_final_correct")
        if first == last == "未评估":
            continue
        if source.startswith("final_skill_evaluation/"):
            origin = "最终评估"
        else:
            source_step = next((r for r in history if f"step_{r['step']:04d}/" in source), None)
            origin = f"R{source_step.get('epoch', source_step['step'])}" if source_step else source
            if "/batch_" in source:
                origin += "/" + next(p for p in source.split("/") if p.startswith("batch_"))
        best_acc.append(f"{first}／{last}（来自{origin}）")
    mode = {"loose": "宽松", "strict_with_rescue": "严格（整批验证＋失败抢救）"}.get(
        summary.get("skill_selection_mode"), summary.get("skill_selection_mode", "未记录"))
    total = (summary.get("token_summary") or {}).get("_total") or {}
    token = total.get("total_tokens") if isinstance(total, dict) else None
    token_text = f"{token:,}" if isinstance(token, int) and token >= 0 else "未记录"
    seconds = summary.get("total_wall_time_s")
    if isinstance(seconds, (int, float)) and seconds >= 0:
        minutes = int(seconds // 60)
        duration = f"训练器计时 {minutes // 60}小时{minutes % 60}分"
    else:
        duration = "耗时未记录"
    recorded_epochs = len({r["epoch"] for r in round_rows})
    progress = f"已记录{recorded_epochs}/{epochs}轮、{len(history)}步骤；"
    progress += ("最终答题已完成，真实 ACC 待补判分"
                 if final_meta.get("status") == "pending_real_acc" else
                 "最终评估完成" if final_meta.get("status") == "complete" else "最终评估未完成或未记录")
    headers = ["数据集／方法", f"{label} 首次回答 ACC（%）", f"{label} 纠错后 ACC（%）",
               f"{label} 更新后{metric_name}（%）", f"{label} 更新后技能数（条）",
               "最佳技能轮次／规模", "最佳技能对应首次／纠错后 ACC", "最终技能首次／纠错后 ACC",
               "最佳技能 Token／已记录调用 Token", "完成耗时／当前进度"]
    cells = [f"{cfg.get('dataset_key', root.name)}／{method} {mode}",
             sequence("first"), sequence("last"), sequence("score"), sequence("count"),
             f"{best_round}／{best_count}", "\n".join(best_acc) or "未评估（无匹配记录）",
             f"{_acc(final_m, 'real_initial_correct')}／{_acc(final_m, 'real_final_correct')}",
             f"未记录／{token_text}", f"{duration}；{progress}"]
    return ["\n【最终输出表格：R1→R4 横向汇总】",
            "| " + " | ".join(_cell(x) for x in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            "| " + " | ".join(_cell(x) for x in cells) + " |",
            "说明：各轮 ACC 对应轮初答题所用技能；更新后技能数和选轮指标对应轮末技能。",
            f"最佳轮仅按{metric_name}选择；最佳技能 ACC 引用标注来源的已有答题记录，不是额外重测。",
            "—＝该项未记录，不代表零；多步骤/多批次分别标注，不混算为一次完整数据集 ACC。",
            "技能 Token 未单独记录，不以字符数估算；调用 Token 仅覆盖现有追踪器，不宣称全量。",
            "耗时采用训练器 total_wall_time_s，不等于入口启动至退出总耗时。"]


def _label(skill):
    return f"[{skill['skill_id']}] {str(skill['title']).replace(chr(10), ' ')}"


def _changes(before, after, step_dir, record):
    old, new = _skills(before), _skills(after)
    if old is None or new is None:
        return ["技能变化：未记录（缺少有效的前后快照）"]
    old = {x["skill_id"]: x for x in old}
    new = {x["skill_id"]: x for x in new}
    removed, added = set(old) - set(new), set(new) - set(old)
    modified = []
    edits = (_read(step_dir / "ranked_edits.json", {}) or {}).get("edits", [])
    rescue = record.get("rescue")
    if isinstance(rescue, dict):
        applied = [x for x in rescue.get("trials", [])
                   if x.get("action") in ("accept", "accept_new_best")]
    elif (record.get("action") in ("accept", "accept_new_best", "force_accept")
          and _identity(_read(step_dir / "candidate_skill.json")) == _identity(after)):
        applied = [x for x in _read(step_dir / "edit_apply_report.json", [])
                   if str(x.get("status", "")).startswith("applied")]
    else:
        applied = []
    for row in applied:
        idx = row.get("source_edit")
        if row.get("op") != "replace" or not isinstance(idx, int) or not 1 <= idx <= len(edits):
            continue
        edit = edits[idx - 1]
        target, drafts = edit.get("target_skill_id"), edit.get("skills", [])
        if target not in removed or len(drafts) != 1:
            continue
        match = [sid for sid in added if all(new[sid].get(k) == drafts[0].get(k)
                                            for k in ("title", "solver_guidance"))]
        if len(match) == 1:
            sid = match[0]
            modified.append(f"↻ {_label(old[target])} → {_label(new[sid])}")
            removed.remove(target)
            added.remove(sid)
    lines = [f"技能数量：{len(old)} → {len(new)} 条；净变化 {len(new)-len(old):+d} 条",
             f"快照变更：新增 +{len(added)} 条；删除 −{len(removed)} 条；确认改写 ↻{len(modified)} 条"]
    lines += [f"+ {_label(x)}" for sid, x in new.items() if sid in added]
    lines += [f"− {_label(x)}" for sid, x in old.items() if sid in removed]
    lines += modified
    if added or removed:
        lines.append("注：未能由实际应用记录确认的改写，以移出/移入列示；不推断语义合并。")
    if set(old) == set(new) and list(old) != list(new):
        lines.append("顺序调整：技能成员不变，仅顺序改变。")
    if any(old[sid] != new[sid] for sid in set(old) & set(new)):
        lines.append("已有技能记录存在字段更新（如证据）；不计作新增技能。")
    return lines


def _role_lines(step_dir):
    lines = []
    for path in sorted(step_dir.rglob("*aggregate_role_screening.json")):
        data = _read(path, {})
        if not isinstance(data, dict):
            continue
        if "input_atomic_skill_count" not in data and isinstance(data.get("results"), list):
            batches = data["results"]
            combined = {"status": f"分批汇总，技术错误{len(data.get('errors', []))}条"}
            for key in ("input_atomic_skill_count", "output_atomic_skill_count", "dropped_count",
                        "repaired_count", "reused_role_approval_count"):
                values = [row.get(key) for row in batches if isinstance(row, dict)]
                combined[key] = sum(values) if len(values) == len(batches) and all(isinstance(x, int) for x in values) else "未记录"
            data = combined
        lines.append(f"角色检查 [{path.relative_to(step_dir)}]：状态={data.get('status', '未记录')}；"
                     f"输入={data.get('input_atomic_skill_count', '未记录')}条次；"
                     f"输出={data.get('output_atomic_skill_count', '未记录')}条次；"
                     f"淘汰={data.get('dropped_count', '未记录')}；"
                     f"修复={data.get('repaired_count', '未记录')}；"
                     f"复用批准={data.get('reused_role_approval_count', '未记录')}")
    return lines or ["角色检查：未记录；不据此认定全部通过。"]


def render_final_report(out_root, summary=None):
    """Render existing artifacts; safe to run again offline on completed runs."""
    root = Path(out_root)
    summary = summary if summary is not None else _read(root / "summary.json", {})
    cfg = summary.get("config", {})
    history = _read(root / "history.json", [])
    if not isinstance(history, list):
        history = []
    history = sorted(history, key=lambda x: x.get("step", 0))
    best_step = summary.get("best_step")
    best = _read(root / "best_skill.json")
    metric = summary.get("selection_metric", "forward_pass_rate")
    metric_name = "正向通过率" if metric == "forward_pass_rate" else "ACC" if metric == "real_acc" else f"选择指标({metric})"
    method = next((part for part in str(cfg.get("out_root", root)).replace("\\", "/").split("/")
                   if part.startswith("S5-")), "S5")
    lines = ["================ S5 最终实验结果 ================",
             f"方法：{method} | 模式：{summary.get('skill_selection_mode', '未记录')}",
             f"数据集：{cfg.get('dataset_key', root.name)} | 模型：{cfg.get('target_model', '未记录')}",
             f"已记录训练步骤：{len(history)}；计划轮数：{cfg.get('num_epochs', '未记录')}（多批次时一轮可含多个步骤）",
             "ACC＝标准答案判题准确率；正向通过率≠ACC。",
             f"★ 最优轮按{metric_name}选择；汇总只读已有结果，不额外调用模型。",
             "符号：+新增；−删除；↻确认改写；↑/↓/=指标变化；—未评估或不适用。",
             "\n【每轮过程】",
             f"阶段 | 答题使用技能 | 学生首次回答 ACC | 纠错后 ACC | 纠错净变化 | 更新后技能/相邻步骤净增 | 更新后{metric_name}"]
    observations, details, round_rows = [], [], []
    previous = _read(root / "skills/skill_v0000.json")
    previous_step = 0
    previous_score = summary.get("baseline_selection_hard")
    for rec in history:
        step = rec["step"]
        sd = root / "steps" / f"step_{step:04d}"
        inp = _read(sd / "rollout_input_skill.json")
        source_note = "轮初快照"
        if inp is None and not cfg.get("use_slow_update", False) and not cfg.get("use_meta_skill", False):
            inp = previous
            source_note = "相邻快照推定（旧记录；未启用 slow/meta 更新）"
        after = _read(root / "skills" / f"skill_v{step:04d}.json")
        ins, outs = _skills(inp), _skills(after)
        input_label = f"L{previous_step}：{len(ins)}条" if ins is not None else "未记录"
        # Multiple accumulation batches must remain separate; never double-count metrics.
        metric_paths = sorted(sd.glob("batch_*/rollout/forward_metrics.json"))
        if not metric_paths:
            metric_paths = [sd / "rollout/forward_metrics.json"]
        outlabel = f"{'★ ' if best_step == step else ''}L{step}：{len(outs)}条" if outs is not None else "未记录"
        if ins is not None and outs is not None:
            outlabel += f"（净增{len(outs)-len(ins):+d}）"
        score = rec.get("current_score")
        score_text = _pct(score)
        if isinstance(score, (float, int)) and isinstance(previous_score, (float, int)):
            delta = score - previous_score
            score_text += f" {'↑' if delta>0 else '↓' if delta<0 else '='}{delta*100:+.1f}个百分点"
        acc_cells = {"real_initial_correct": [], "real_final_correct": []}
        for mp in metric_paths:
            m = _read(mp, {})
            for key in acc_cells:
                value = _acc(m, key)
                value = "—" if value == "未评估" else value.split("%", 1)[0]
                if len(metric_paths) > 1:
                    value = f"{mp.parent.parent.name}:{value}"
                acc_cells[key].append(value)
            source = mp.relative_to(root).as_posix()
            stage = f"第{rec.get('epoch', step)}轮/步骤{step}"
            if len(metric_paths) > 1:
                stage += f"/{mp.parent.parent.name}"
            lines.append(f"{stage} | {input_label} | {_acc(m, 'real_initial_correct')} | "
                         f"{_acc(m, 'real_final_correct')} | {_gain(m)} | {outlabel} | {score_text}")
            matched_input = inp
            if ins is not None and m.get("skill_count") != len(ins):
                matched_input = None
                source_note += "；指标技能数与快照不匹配，不用于最佳轮归属"
            observations.append((matched_input, m, source, source_note))
            details.append(f"[{stage}] 记录来源：{source}；技能来源：{source_note}")
            details.append(f"纠错尝试={m.get('repair_attempted', '未记录')}题；验证器接受修正={m.get('repair_accepted', '未记录')}题（不代表真实答对）；"
                           f"验证器异常={m['trajectory_counts'].get('verifier_error', 0) if isinstance(m.get('trajectory_counts'), dict) else '未记录'}题")
            if m.get("real_acc_status") == "ok":
                for key, label in (("real_improved_ids", "错→对"), ("real_regressed_ids", "对→错")):
                    ids = m.get(key)
                    details.append(f"{label}：{len(ids)}题 {ids}" if isinstance(ids, list) else f"{label}：未记录")
        round_rows.append({"epoch": rec.get("epoch", step), "step": step,
                           "first": "；".join(acc_cells["real_initial_correct"]),
                           "last": "；".join(acc_cells["real_final_correct"]),
                           "score": f"{score * 100:.1f}" if isinstance(score, (int, float)) else "—",
                           "count": str(len(outs)) if outs is not None else "—"})
        details += [f"\n【第{rec.get('epoch', step)}轮/步骤{step}技能变更】",
                    f"更新结果：{rec.get('action', '未记录')}；接收来源：{rec.get('acceptance_source', '未记录')}"]
        details += _changes(inp, after, sd, rec) + _role_lines(sd)
        previous, previous_step, previous_score = after, step, score
    final_meta = _read(root / "final_skill_evaluation.json", {})
    final_m = _read(root / "final_skill_evaluation/forward_metrics.json", {})
    if final_meta.get("status") == "pending_real_acc":
        final_m = {**final_m, "status": "pending_real_acc"}
    elif final_meta.get("status") != "complete":
        final_m = {}
    final_version = final_meta.get("evaluated_skill_version")
    final_library = _read(root / "skills" / f"skill_v{final_version:04d}.json") if isinstance(final_version, int) else None
    if final_meta.get("status") == "complete":
        lines.append(f"最终评估 | L{final_version}：{final_m.get('skill_count', '未记录')}条 | "
                     f"{_acc(final_m, 'real_initial_correct')} | {_acc(final_m, 'real_final_correct')} | {_gain(final_m)} | 不更新 | —")
    elif final_meta.get("status") == "pending_real_acc":
        lines.append(f"最终评估 | L{final_version}：答题已完成 | 待补判分 | 待补判分 | — | 不更新 | —")
        lines.append("判分技术异常（不计作答错，回答已保留）：" + str(final_meta.get("real_acc_error", "未记录")))
    else:
        lines.append("最终评估 | 未完成或未评估 | 未评估 | 未评估 | — | 不更新 | —")
    lines += ["\n【最优轮与最终轮 ACC】", "对象 | 学生首次回答 ACC | 纠错后 ACC | 数据来源"]
    baseline = [x for x in observations if (x[1].get("initial_response_source_counts") or {}).get("shared_initial_response_bootstrap", 0) == x[1].get("n") and x[1].get("n", 0) > 0]
    matches = [x for x in observations if _identity(best) is not None and _identity(x[0]) == _identity(best)
               and x[1].get("real_acc_status") == "ok"]
    if _identity(best) is not None and _identity(best) == _identity(final_library) and final_meta.get("status") == "complete" and final_m.get("real_acc_status") == "ok":
        matches = [(final_library, final_m, "final_skill_evaluation/forward_metrics.json", "最终评估")]
    overview_at = lines.index("\n【每轮过程】")
    lines[overview_at:overview_at] = _round_overview(
        root, summary, method, metric_name, history, round_rows, best, matches, final_meta, final_m)
    for label, candidates in (("公共初始回答", baseline), (f"★ 最优轮 L{best_step}（{summary.get('best_atomic_skill_count', '未记录')}条）", matches)):
        if not candidates:
            lines.append(f"{label} | 未评估 | 未评估 | 无匹配记录，不拿其他轮次替代")
        for _, m, source, note in candidates:
            lines.append(f"{label} | {_acc(m, 'real_initial_correct')} | {_acc(m, 'real_final_correct')} | {source}；{note}")
    lines.append(f"最终评估 L{final_version} | {_acc(final_m, 'real_initial_correct')} | {_acc(final_m, 'real_final_correct')} | final_skill_evaluation/forward_metrics.json")
    lines.append("最优轮可能引用后续训练记录，不是统一重测；有多批匹配记录时逐批列出，不挑 ACC 最高的一批。")
    lines += details
    lines += ["\n【运行情况】", f"训练器统计耗时：{summary.get('total_wall_time_s', '未记录')}秒（不等于入口启动至退出总耗时）"]
    for rec in history:
        timing = rec.get("timing", {})
        lines.append(f"步骤{rec['step']}耗时：{rec.get('wall_time_s', '未记录')}秒；" + "；".join(
            f"{label}={timing.get(key, '未记录')}秒" for key,label in (
                ('rollout_s','答题/纠错'),('reflect_s','技能分析'),('aggregate_s','合并'),
                ('select_s','排序'),('update_s','更新'),('evaluate_s','选轮验证'))))
    lines += ["阶段耗时按已记录字段打印；最终评估单独耗时未记录时不估算。",
              f"Token：{summary.get('token_summary', {}).get('_total', '未记录')}（仅现有追踪器统计范围，不宣称覆盖全部调用）",
              "异常恢复次数：未记录；各阶段验证器异常见上表，不将格式警告条数等同于失败次数。",
              f"结果目录：{root}", f"最佳技能：{root / 'best_skill.json'}",
              f"最终轮技能：{root / 'skills' / ('skill_v%04d.json' % previous_step)}",
              "================================================"]
    return "\n".join(lines) + "\n"


def print_final_report(out_root, summary=None):
    text = render_final_report(out_root, summary)
    (Path(out_root) / "final_report.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="只读已有 S5 结果并打印，不调用模型")
    parser.add_argument("out_root")
    args = parser.parse_args()
    print(render_final_report(args.out_root), end="")
