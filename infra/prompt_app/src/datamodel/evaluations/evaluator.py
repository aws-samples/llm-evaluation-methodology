# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Abstracted wrapper class for configuring supported evaluation algorithms in fmeval
"""
# Python Built-Ins:
import json
from typing import Generator, List, Optional

# External Dependencies:
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval import get_eval_algorithm
from fmeval.eval_algorithms import EvalOutput, EvalScore
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface
from fmeval.model_runners.model_runner import ModelRunner

# Local Dependencies:
from .base import EvalAlgorithm
from .self_critique import QAAccuracyByLLM, QAAccuracyByLLMConfig
from ..model import get_model_runner, MODELS


class Evaluator(EvalAlgorithmInterface):
    """An fmeval EvalAlgorithm-like object with extra abstractions

    - You can create it direct from the target eval algorithm name
    - We stub out the unnecessary `prompt_template` argument in `.evaluate()`
    - Use the `.iter_results()` helper to iterate through detailed results, so long as you still
      know your data_config.dataset_name.
    """

    fmeval_algo: EvalAlgorithmInterface

    def __init__(self, algo_id: EvalAlgorithm):
        # Check custom algorithms first:
        if algo_id == EvalAlgorithm.QA_ACCURACY_BY_LLM:
            self.fmeval_algo = QAAccuracyByLLM(
                # TODO: This evaluator should be pre-configured with evaluator model ID/etc
                QAAccuracyByLLMConfig(
                    eval_model_runners=[get_model_runner(MODELS[0][0], MODELS[0][1][0])],
                )
            )
        else:
            # Otherwise should be a default fmeval algorithm:
            self.fmeval_algo = get_eval_algorithm(algo_id)

    def evaluate(
        self,
        model: Optional[ModelRunner] = None,
        dataset_config: Optional[DataConfig] = None,
        save: bool = True,
        num_records: int = 100,
    ) -> List[EvalOutput]:
        # TODO: Properly reflect the num_records limit in the UI
        return self.fmeval_algo.evaluate(
            model=model,
            dataset_config=dataset_config,
            prompt_template="$model_input",  # This app handles templating outside fmeval
            save=save,
            num_records=num_records,
        )

    def evaluate_sample(
        self,
        model_input: Optional[str] = None,
        target_output: Optional[str] = None,
        model_output: Optional[str] = None,
    ) -> List[EvalScore]:
        return self.eval_algo.evaluate_sample(
            model_input=model_input,
            target_output=target_output,
            model_output=model_output,
        )

    def iter_results(self, dataset_name: str) -> Generator[dict, None, None]:
        with open(
            f"/tmp/eval_results/{self.fmeval_algo.eval_name}_{dataset_name}.jsonl", "r"
        ) as foutput:
            for line in foutput:
                yield json.loads(line) if line else None
