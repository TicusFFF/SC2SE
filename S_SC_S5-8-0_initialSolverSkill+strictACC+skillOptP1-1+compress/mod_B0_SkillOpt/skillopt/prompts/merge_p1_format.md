## S5 Edit Proposal Format

An Edit is an outer editing proposal, not a skill.
Each object in its `skills` array is one independently reviewable mathematics problem-solving skill.

You may merge, synthesize, generalize, and rewrite proposed skills according to the
merge guidelines. Every resulting skill must still satisfy the target-role requirements.

Each skill contains exactly these two non-empty string fields:
- title
- solver_guidance

Supported operations:
- append: add one or more skills.
- insert_after: add one or more skills after an existing skill.
- replace: replace exactly one existing skill with exactly one skill.
- delete: delete one existing skill; do not include a skills field.

For insert_after, replace, and delete, use the full target_skill_id from the
current skill library. Do not invent, shorten, or renumber a target_skill_id.

Optional metadata belongs to the outer Edit:
- support_count: a non-negative integer describing support for this proposal.
- source_type: failure or success.
- merge_level: a non-negative integer.

Do not put IDs, support counts, evidence, or merge metadata inside a two-field skill.
Do not output the program-owned _evidence_sidecar.

Respond ONLY with one valid JSON object in this format:

{
  "schema_version": "s5_7_8_atomic_solver_skill_patch_v1",
  "reasoning": "<brief explanation of the merge decisions>",
  "edits": [
    {
      "op": "append",
      "skills": [
        {
          "title": "<independent problem-solving skill name>",
          "solver_guidance": "<self-contained mathematics problem-solving guidance>"
        }
      ],
      "support_count": 1,
      "source_type": "failure"
    },
    {
      "op": "insert_after",
      "target_skill_id": "<full existing skill ID>",
      "skills": [
        {
          "title": "<independent problem-solving skill name>",
          "solver_guidance": "<self-contained mathematics problem-solving guidance>"
        }
      ]
    },
    {
      "op": "replace",
      "target_skill_id": "<full existing skill ID>",
      "skills": [
        {
          "title": "<replacement problem-solving skill name>",
          "solver_guidance": "<self-contained mathematics problem-solving guidance>"
        }
      ]
    },
    {
      "op": "delete",
      "target_skill_id": "<full existing skill ID>"
    }
  ]
}

Include only the operations actually needed. The edits array may be empty.
Do not include code fences or text outside the JSON object.
