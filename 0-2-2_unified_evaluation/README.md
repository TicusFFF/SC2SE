# Answer evaluation

SC2SE scores two responses per question using the same V5 answer extractor and V2 equivalence judge. The extractor reads the question and the response. The judge compares the extracted answer with the reference.

The experiment runner prepares both inputs and scores them automatically. A demo run writes the following files inside its output directory.

| Response from the final skill library | Saved responses | Scorer input | Score report |
|---|---|---|---|
| Initial answer using the frozen skills | `demo/responses_initial.json` | `evaluation_input_initial.json` | `evaluation_initial.json` |
| Answer after correction | `demo/responses_corrected.json` | `evaluation_input_corrected.json` | `evaluation_corrected.json` |

To check the saved inputs locally, run the commands below from the repository root. Replace `runs/demo-run` with the output directory printed by the runner.

```console
python 0-2-2_unified_evaluation/evaluate.py --input_json runs/demo-run/evaluation_input_initial.json
python 0-2-2_unified_evaluation/evaluate.py --input_json runs/demo-run/evaluation_input_corrected.json
```

For a manual evaluation, the example below rebuilds the input from the corrected answers and runs the scorer. Use `responses_initial.json` and change the output and cache names from `corrected` to `initial` to evaluate the skill-guided initial answers. Repeat `--dataset KEY DATASET_JSON RESPONSES_JSON` to combine datasets at the same stage. The preparer checks each question's ID and text against the selected dataset. Set `DEEPSEEK_API_KEY` for the official DeepSeek endpoint. Extraction and judging use `deepseek-flash` with thinking disabled and temperature zero when a model call is needed.

```console
python 0-2-2_unified_evaluation/prepare_inputs.py --dataset demo 0-0_datasets/hard_test_raw/demo/test_top2_filt2.json runs/demo-run/demo/responses_corrected.json --output runs/demo-run/recheck_input_corrected.json
python 0-2-2_unified_evaluation/evaluate.py --input_json runs/demo-run/recheck_input_corrected.json --output_json runs/demo-run/recheck_corrected.json --cache_dir runs/demo-run/recheck_cache_corrected --run
```

`acc` is the percentage of correct responses among all expected IDs. An empty answer counts as incorrect. If any score is pending, the report gives its ID and leaves `acc` unset. A resumed run can reuse its checkpoint with `--resume` and a new result path. Use `--retry_pending` with `--resume` to retry recorded transient failures.

To run the offline checks with Python's standard library:

```console
python 0-2-2_unified_evaluation/test_offline.py
```
