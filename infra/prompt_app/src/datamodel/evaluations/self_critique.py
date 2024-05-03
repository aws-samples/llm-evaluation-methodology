# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Implementing LLM self/peer critique as an evaluation method for AWS FMEval

(As of v1.0.0) the fmeval library avoids any evaluation algorithms using *LLMs themselves* to
critique and score an LLM-generated answer. This stands in contrast to other tools such as
[LlamaIndex](https://docs.llamaindex.ai/en/stable/api_reference/evaluation/correctness/) where the
approach is supported.

These custom utilities implement LLM-based critique within an FMEval context.
"""
# Python Built-Ins:
from dataclasses import dataclass
import json
from string import Template
from typing import Any, Dict, List, Optional

# External Dependencies:
from fmeval.constants import DatasetColumns, MEAN
from fmeval.data_loaders.data_config import DataConfig
from fmeval.data_loaders.util import get_dataset
from fmeval.eval_algorithms import EvalAlgorithm, EvalOutput, EvalScore
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmConfig, EvalAlgorithmInterface
from fmeval.eval_algorithms.util import evaluate_dataset, get_dataset_configs, validate_dataset
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.transforms.transform import Transform
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.transforms.util import validate_call
from fmeval.util import get_eval_results_path

# Local Dependencies:
from .base import EvalAlgorithm
from ..model.bedrock import claude_text_to_structured_messages

# Prompt template to ask an LLM to critique correctness of answer vs ground truth:
EVAL_TPL = Template(
    """Human:
An AI model was asked a question for which the reference correct answer(s) were:

<ref-answers>
${target}
</ref-answers>

The model's answer was:

<model-answer>
${output}
</model-answer>

Did the model answer correctly in agreement with the provided reference(s)? Answer only Y for yes
or N for no, and do not include any other information or reasoning.

Assistant:
"""
)

OUTPUT_KEY = "llm_judged_accuracy"


class QAAccuracyByLLMScores(Transform):
    """Scorer inspired by fmeval.eval_algorithms.qa_accuracy.QAAccuracyScores"""

    def __init__(
        self,
        eval_model_runners: List[ModelRunner],
        target_output_key: str = DatasetColumns.TARGET_OUTPUT.value.name,
        model_output_key: str = DatasetColumns.MODEL_OUTPUT.value.name,
        target_output_delimiter: Optional[str] = "<OR>",
    ):
        output_keys = [OUTPUT_KEY]
        super().__init__(
            eval_model_runners,
            target_output_key,
            model_output_key,
            target_output_delimiter,
        )
        self.register_input_output_keys(
            input_keys=[target_output_key, model_output_key],
            output_keys=output_keys,
        )
        self.target_output_key = target_output_key
        self.model_output_key = model_output_key
        self.output_keys = output_keys
        self.target_output_delimiter = target_output_delimiter

        self.eval_model_runners = eval_model_runners
        if not (self.eval_model_runners and len(self.eval_model_runners)):
            raise EvalAlgorithmClientError(
                "You must provide at least one ModelRunner for LLM-based QA Accuracy evaluation"
            )

    @staticmethod
    def _get_score(model_runner: ModelRunner, model_output: str, targets: List[str]) -> float:
        prompt = EVAL_TPL.substitute(
            target="\n".join([f"<ref-answer>{t}</ref-answer>" for t in targets]),
            output=model_output,
        )
        # TODO: This behaviour should be evaluator-model-specific
        prompt = json.dumps(claude_text_to_structured_messages(prompt))
        eval_resp, logprobs = model_runner.predict(prompt)
        eval_resp = eval_resp.strip().upper()
        if not len(eval_resp):
            return 0.5  # Swallow unexpected evaluation result & return 'not sure'
        elif eval_resp[0] == "Y":
            return 1.0
        elif eval_resp[0] == "N":
            return 0.0
        else:
            return 0.5  # Swallow unexpected evaluation result & return 'not sure'

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        model_output = record[self.model_output_key]
        target_outputs = record[self.target_output_key].split(self.target_output_delimiter)

        # Return average score across all evaluator models
        record[OUTPUT_KEY] = sum(
            self._get_score(model_runner=runner, model_output=model_output, targets=target_outputs)
            for runner in self.eval_model_runners
        ) / len(self.eval_model_runners)
        # sum(score_fn_override(runner, model_output, possible_targets) for runner in model_runners) / len(model_runners)
        return record


@dataclass(frozen=True)
class QAAccuracyByLLMConfig(EvalAlgorithmConfig):
    """Configuration for the QA Accuracy Evaluation

    :param eval_model_runners: The QAAccuracyByLLM evaluator uses one or more LLMs to judge whether the answer
        answer generated by the model under test is in agreement with the reference answers. Therefore you need to
        provide one or more ModelRunner instances to use for the evaluation.
    :param target_output_delimiter: Target Output can have multiple answers. We expect customer to combine all the
        possible answers into a single string and use the delimiter to separate them. For instance,
        if the answers are ["UK", "England"] and the delimiter="<OR>", then the target_output should be "UK<OR>England".
    """

    eval_model_runners: List[ModelRunner]
    target_output_delimiter: Optional[str] = "<OR>"

    def __post_init__(self):
        if not len(self.eval_model_runners):
            raise EvalAlgorithmClientError(
                "You must provide at least one ModelRunner for LLM-based QA Accuracy evaluation"
            )
        if self.target_output_delimiter == "":
            raise EvalAlgorithmClientError(
                "Empty target_output_delimiter is provided. Please either provide a non-empty string, or set it to None."
            )


class QAAccuracyByLLM(EvalAlgorithmInterface):
    """This evaluation measures question answering (QA) performance via critique from LLM(s)

    The code is closely aligned to fmeval's vanilla `QAAccuracy` eval algorithm, since the actual
    logic is implemented in the `QAAccuracyByLLMScores` transformer. We had to re-implement (rather
    than re-using the QAAccuracy) because of the way constants like the list of score names are
    referenced in the upstream.

    This evaluator outputs one metric only: The mean of the judged 0-0.5-1 response quality judged
    by the panel of (potentially multiple) evaluator model runners.
    """

    eval_name = EvalAlgorithm.QA_ACCURACY_BY_LLM.value

    def __init__(self, eval_algorithm_config: QAAccuracyByLLMConfig):
        super().__init__(eval_algorithm_config)
        self._eval_algorithm_config = eval_algorithm_config
        self.transform = QAAccuracyByLLMScores(
            eval_model_runners=eval_algorithm_config.eval_model_runners,
            target_output_delimiter=eval_algorithm_config.target_output_delimiter,
        )

    def evaluate_sample(self, target_output: str, model_output: str) -> List[EvalScore]:
        """Compute QA accuracy metrics for a single sample.

        :param target_output: The expected/desired model output.
        :param model_output: The actual model output.
        :returns: A list of EvalScore objects, one for each of the QA accuracy metrics.
        """
        target_output_key = self.transform.target_output_key
        model_output_key = self.transform.model_output_key
        sample = {target_output_key: target_output, model_output_key: model_output}
        pipeline = TransformPipeline([self.transform])
        result = pipeline.execute_record(sample)
        return [
            EvalScore(name=score_name, value=result[score_name])
            for score_name in self.transform.output_keys
        ]

    def evaluate(
        self,
        model: Optional[ModelRunner] = None,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        num_records: int = 100,
        save: bool = False,
    ) -> List[EvalOutput]:
        dataset_configs = get_dataset_configs(dataset_config, self.eval_name)
        eval_outputs = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [DatasetColumns.TARGET_OUTPUT.value.name])
            eval_output = evaluate_dataset(
                dataset=dataset,
                pipeline=TransformPipeline([self.transform]),
                dataset_name=dataset_config.dataset_name,
                eval_name=self.eval_name,
                metric_names=self.transform.output_keys,
                eval_results_path=get_eval_results_path(),
                model=model,
                prompt_template=prompt_template,
                agg_method=MEAN,
                save=save,
            )
            eval_outputs.append(eval_output)
        return eval_outputs
