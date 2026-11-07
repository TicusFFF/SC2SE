from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List

from S5_bridge.data_sanitizer import (
    load_json_or_jsonl,
    split_same_pool,
)
from skillopt.datasets.base import (
    BaseDataLoader,
    BatchSpec,
)
from skillopt.envs.base import (
    EnvAdapter,
)
from skillopt.gradient.reflect import (
    run_minibatch_reflect,
)
from skillopt.optimizer.structured_skills import write_json_atomic

from .rollout import (
    run_forward_math_batch,
    uses_shared_initial_response_phase,
)


class ForwardMathDataLoader(
    BaseDataLoader,
):
    def __init__(
        self,
        dataset_path: str = "",
        dataset_key: str = "",
        seed: int = 42,
        limit: int = 0,
    ) -> None:
        self.dataset_path = dataset_path
        self.dataset_key = dataset_key
        self.seed = seed
        self.limit = limit

        self.splits: Dict[
            str,
            List[Dict[str, Any]],
        ] = {}

    def setup(
        self,
        cfg: dict,
    ) -> None:
        dataset_path = (
            self.dataset_path
            or cfg.get("dataset_path")
            or cfg.get("data_path")
        )

        if not dataset_path:
            raise ValueError(
                "forward_math requires dataset_path / data_path in cfg."
            )

        dataset_key = (
            self.dataset_key
            or str(
                cfg.get(
                    "dataset_key",
                    "",
                )
            )
        ).strip()

        self.dataset_path = str(
            dataset_path
        )
        self.dataset_key = dataset_key

        items = load_json_or_jsonl(
            dataset_path,
            dataset_key=dataset_key or None,
        )

        if self.limit:
            items = items[
                : self.limit
            ]

        if not items:
            raise ValueError(
                f"forward_math dataset is empty: {dataset_path}"
            )

        self.splits = split_same_pool(
            items
        )

    def get_split(
        self,
        split: str,
    ) -> List[Dict[str, Any]]:
        return list(
            self.splits[split]
        )

    def get_split_items(
        self,
        split: str,
    ) -> List[Dict[str, Any]]:
        if split in self.splits:
            return self.get_split(split)

        if split in {
            "selection",
            "val",
            "valid",
        }:
            return self.get_split(
                "valid_seen"
            )

        if split == "test":
            return self.get_split(
                "test"
            )

        return self.get_split(
            "valid_seen"
        )

    def get_train_size(self) -> int:
        return len(
            self.splits.get(
                "train",
                [],
            )
        )

    def plan_train_epoch(
        self,
        *,
        epoch: int,
        steps_per_epoch: int,
        accumulation: int,
        batch_size: int,
        seed: int,
        **kwargs,
    ) -> list[BatchSpec]:
        """
        Shuffle once per epoch, then slice batches sequentially.

        This prevents each batch from independently sampling the whole
        dataset and avoids unnecessary duplicate/omitted samples under the
        default accumulation=1 setting.
        """
        del kwargs

        items = self.get_split(
            "train"
        )

        random.Random(
            seed + epoch * 1000
        ).shuffle(items)

        total_batches = (
            steps_per_epoch
            * accumulation
        )

        batches: list[BatchSpec] = []
        cursor = 0

        for batch_index in range(
            total_batches
        ):
            batch_seed = (
                seed
                + epoch * 1000
                + batch_index
                + 1
            )

            batch_items = items[
                cursor : cursor + batch_size
            ]

            cursor += len(batch_items)

            if not batch_items and items:
                batch_items = list(items)

                random.Random(
                    batch_seed
                ).shuffle(batch_items)

                batch_items = batch_items[
                    :batch_size
                ]

            batches.append(
                BatchSpec(
                    phase="train",
                    split="train",
                    seed=batch_seed,
                    batch_size=len(batch_items),
                    payload=batch_items,
                )
            )

        return batches

    def build_train_batch(
        self,
        batch_size: int,
        seed: int,
        **kwargs,
    ) -> BatchSpec:
        del kwargs

        items = self.get_split(
            "train"
        )

        random.Random(seed).shuffle(
            items
        )

        if batch_size > 0:
            items = items[:batch_size]

        return BatchSpec(
            phase="train",
            split="train",
            seed=seed,
            batch_size=len(items),
            payload=items,
        )

    def build_eval_batch(
        self,
        env_num: int,
        split: str,
        seed: int,
        **kwargs,
    ) -> BatchSpec:
        del kwargs

        items = self.get_split_items(
            split
        )

        if (
            env_num > 0
            and env_num < len(items)
        ):
            items = items[:env_num]

        return BatchSpec(
            phase="eval",
            split=split,
            seed=seed,
            batch_size=len(items),
            payload=items,
        )

    def get_task_types(
        self,
    ) -> List[str]:
        return [
            "forward_math",
        ]


class ForwardMathAdapter(
    EnvAdapter,
):
    name = "forward_math"

    def __init__(
        self,
        dataset_path: str = "",
        data_path: str = "",
        rollout_system_path: str = "",
        forward_runner=None,
        forward_runner_factory=None,
        rollout_workers: int = 1,
        analyst_workers: int = 16,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        dataset_key: str = "",
        metrics_model: str = "",
        defer_report_scoring: bool = False,
        shared_initial_response_map: dict | None = None,
        bootstrap_shared_initial_response: bool = True,
        **kwargs,
    ) -> None:
        del kwargs

        self.forward_runner = forward_runner
        self.forward_runner_factory = (
            forward_runner_factory
        )
        self.rollout_system_path = rollout_system_path

        try:
            self.rollout_workers = max(
                1,
                int(
                    rollout_workers
                    or 1
                ),
            )

        except (TypeError, ValueError) as error:
            raise ValueError(
                "forward_math rollout_workers must be a positive integer."
            ) from error

        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.shared_initial_response_map = (
            shared_initial_response_map
        )
        self.bootstrap_shared_initial_response = bool(
            bootstrap_shared_initial_response
        )
        self.defer_report_scoring = bool(defer_report_scoring)
        self.metrics_model = str(
            metrics_model
            or ""
        )

        self.dataloader = ForwardMathDataLoader(
            dataset_path=(
                dataset_path
                or data_path
            ),
            dataset_key=dataset_key,
            seed=seed,
            limit=limit,
        )

    def setup(
        self,
        cfg: dict,
    ) -> None:
        super().setup(cfg)

        self.dataloader.setup(
            cfg
        )

        raw_bootstrap_shared_initial = cfg.get(
            "bootstrap_shared_initial_response",
            self.bootstrap_shared_initial_response,
        )
        self.bootstrap_shared_initial_response = (
            raw_bootstrap_shared_initial
            if isinstance(raw_bootstrap_shared_initial, bool)
            else str(raw_bootstrap_shared_initial or "").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        if self.forward_runner is None:
            raise ValueError(
                "ForwardMathAdapter requires a ForwardSkillRunner. "
                "Start training through S5_bridge/train_S5.py."
            )

        if (
            self.rollout_workers > 1
            and self.forward_runner_factory is None
        ):
            raise ValueError(
                "forward_math rollout_workers > 1 requires "
                "forward_runner_factory. Do not share one "
                "ForwardSkillRunner across worker threads."
            )

        rollout_system_path = (
            self.rollout_system_path
            or cfg.get(
                "rollout_system_path"
            )
            or (
                Path(__file__).parent
                / "prompts"
                / "rollout_system.md"
            )
        )

        rollout_system_path = Path(
            rollout_system_path
        )

        if not rollout_system_path.is_absolute():
            b0_root = (
                Path(__file__)
                .resolve()
                .parents[3]
            )

            rollout_system_path = (
                b0_root
                / rollout_system_path
            )

        if not rollout_system_path.is_file():
            raise FileNotFoundError(
                "Cannot find forward_math rollout prompt: "
                f"{rollout_system_path}"
            )

        self.rollout_system_path = (
            rollout_system_path
        )

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(
        self,
        batch: BatchSpec,
        **kwargs,
    ):
        del kwargs

        return list(
            batch.payload
            or []
        )

    def build_train_env(
        self,
        batch_size: int,
        seed: int,
        **kwargs,
    ):
        batch = (
            self.dataloader
            .build_train_batch(
                batch_size=batch_size,
                seed=seed,
                **kwargs,
            )
        )

        return self.build_env_from_batch(
            batch
        )

    def build_eval_env(
        self,
        env_num: int,
        split: str,
        seed: int,
        **kwargs,
    ):
        batch = (
            self.dataloader
            .build_eval_batch(
                env_num=env_num,
                split=split,
                seed=seed,
                **kwargs,
            )
        )

        return self.build_env_from_batch(
            batch
        )

    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict]:
        adaptive_selection = kwargs.pop(
            "adaptive_selection",
            None,
        )
        student_response_reuse_results = kwargs.pop(
            "student_response_reuse_results",
            None,
        )
        strict_validation_metric = kwargs.pop(
            "strict_validation_metric",
            None,
        )

        del kwargs

        shared_initial_response_map = (
            self.shared_initial_response_map
            if uses_shared_initial_response_phase(
                out_dir,
                bootstrap_shared_initial_response=(
                    self.bootstrap_shared_initial_response
                ),
            )
            else None
        )

        return run_forward_math_batch(
            items=list(env_manager),
            skill_content=skill_content,
            out_dir=out_dir,
            runner=self.forward_runner,
            rollout_system_path=(
                self.rollout_system_path
            ),
            shared_initial_response_map=(
                shared_initial_response_map
            ),
            bootstrap_shared_initial_response=(
                self.bootstrap_shared_initial_response
            ),
            runner_factory=(
                self.forward_runner_factory
            ),
            rollout_workers=(
                self.rollout_workers
            ),
            judge_data_path=(
                "" if self.defer_report_scoring else self.dataloader.dataset_path
            ),
            judge_dataset_key=(
                self.dataloader.dataset_key
            ),
            metrics_model=(
                self.metrics_model
            ),
            judge_workers=(
                self.analyst_workers
            ),
            adaptive_selection=(
                adaptive_selection
            ),
            student_response_reuse_results=(
                student_response_reuse_results
            ),
            strict_validation_metric=(
                strict_validation_metric
            ),
        )

    def reflect(
        self,
        results: list[dict],
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict | None]:
        usable_results = [
            result
            for result in results
            if result.get(
                "trajectory_type"
            ) != "verifier_error"
        ]

        if not usable_results:
            raise RuntimeError(
                "forward_math reflection received no usable trajectories. "
                "Check Forward rollout failures before SkillOpt reflection."
            )

        patches = run_minibatch_reflect(
            results=usable_results,
            skill_content=skill_content,
            prediction_dir=kwargs.get(
                "prediction_dir",
                os.path.join(
                    out_dir,
                    "predictions",
                ),
            ),
            patches_dir=kwargs.get(
                "patches_dir",
                os.path.join(
                    out_dir,
                    "patches",
                ),
            ),
            workers=self.analyst_workers,
            failure_only=self.failure_only,
            minibatch_size=self.minibatch_size,
            edit_budget=self.edit_budget,
            random_seed=kwargs.get(
                "random_seed"
            ),
            error_system=(
                self.get_error_minibatch_prompt()
            ),
            success_system=(
                self.get_success_minibatch_prompt()
            ),
            step_buffer_context=kwargs.get(
                "step_buffer_context",
                "",
            ),
            meta_skill_context=kwargs.get(
                "meta_skill_context",
                "",
            ),
            update_mode=(
                getattr(
                    self,
                    "_cfg",
                    {},
                ).get(
                    "skill_update_mode",
                    "patch",
                )
            ),
            max_completion_tokens=kwargs.get("max_completion_tokens", 0),
            max_analyst_rounds=int(
                getattr(
                    self,
                    "_cfg",
                    {},
                ).get(
                    "max_analyst_rounds",
                    3,
                ) or 3
            ),
            use_preaggregate_compound_screening=kwargs.get(
                "use_preaggregate_compound_screening",
                False,
            ),
            compound_screening_env=kwargs.get(
                "compound_screening_env",
                "forward_math",
            ),
            compound_screening_format_max_attempts=kwargs.get(
                "compound_screening_format_max_attempts",
                3,
            ),
            compound_screening_max_completion_tokens=kwargs.get(
                "compound_screening_max_completion_tokens",
                0,
            ),
        )

        if not patches:
            n_failures = sum(
                1
                for result in usable_results
                if not result.get("hard")
            )

            n_successes = sum(
                1
                for result in usable_results
                if result.get("hard")
            )

            message = (
                "forward_math reflection produced no patches. "
                f"Input trajectories: failures={n_failures}, "
                f"successes={n_successes}. "
                "This usually means the SkillOpt optimizer model did not run, "
                "failed, or returned invalid JSON. Check optimizer backend, "
                "endpoint, deployment name, API key, and analyst traceback."
            )
            write_json_atomic(
                os.path.join(out_dir, "reflection_skipped.json"),
                {
                    "status": "skip_no_reflection_output",
                    "failure_trajectories": n_failures,
                    "success_trajectories": n_successes,
                    "reason": message,
                },
            )
            print(f"    [skip] {message} Keeping the current skill unchanged.")

        return patches

    def get_task_types(
        self,
    ) -> list[str]:
        return (
            self.dataloader
            .get_task_types()
        )
