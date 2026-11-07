## Target Role

The sole user of the candidate Skill is a mathematics problem solver.

This role is responsible for:

- performing mathematical analysis, derivation, and calculation for the current problem;
- using reusable mathematical knowledge, problem-solving strategies, and self-checking methods;
- producing clear and complete mathematical solutions.

Each independent Skill is one atomic JSON object with exactly two non-empty string fields:

- `title`: a concise name for the independent mathematics problem-solving Skill;
- `solver_guidance`: complete, self-contained guidance that a mathematics problem solver can apply directly.

The two fields together are the complete atomic Skill. Do not add Markdown headings, wrapper prose, IDs, evidence metadata, support counts, or extra fields. Program-owned identifiers and provenance are not part of the mathematics problem-solving Skill.

The candidate Skill should be directly usable by a mathematics problem solver during problem solving. It must not:

- depend on subsequent Forward verification or its judgments, feedback, or revision records;
- require the mathematics problem solver to act as a Forward verifier, correction agent, evaluator, or tool agent;
- contain invocation procedures or execution records for internal tools such as PythonTool;
- contain information about internal system workflows, hidden states, reference answers, ground truth, or other roles;
- contain unfinished self-reasoning, mutually contradictory candidate conclusions, analysis drafts, attempt history, or revision history.

The analysis used to generate a candidate Skill may refer to Forward verification, feedback, and revision records, but the final Skill must transform that information into guidance that a mathematics problem solver can use independently, without mentioning or depending on such subsequent information.

Ordinary mathematical checking and self-checking are allowed, including checking calculations, signs, domains, boundary conditions, special cases, and whether one's own derivation is valid, or cross-checking with another mathematical method.

General expressions such as "numerical checking," "numerical verification," "computational verification," direct enumeration, symbolic manipulation, or "consider numerical verification" do not by themselves indicate dependence on PythonTool or another internal tool. Treat them as tool dependence only when the candidate Skill explicitly requires invoking an internal tool, executing an internal-tool workflow, or accessing information or execution results from an internal tool or system component.
