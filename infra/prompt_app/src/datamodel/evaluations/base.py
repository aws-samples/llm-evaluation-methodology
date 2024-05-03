# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Base data list of supported algorithms referenced by other submodules in this package
"""
# Python Built-Ins:
from enum import Enum


# TODO: Use upstream fmeval EvalAlgorithm instead once cmp issue fixed
# https://github.com/aws/fmeval/issues/186
class EvalAlgorithm(str, Enum):
    """The evaluation types supported by Amazon Foundation Model Evaluations.

    The evaluation types are used to determine the evaluation metrics for the
    model.
    """

    PROMPT_STEREOTYPING = "prompt_stereotyping"
    FACTUAL_KNOWLEDGE = "factual_knowledge"
    TOXICITY = "toxicity"
    QA_TOXICITY = "qa_toxicity"
    SUMMARIZATION_TOXICITY = "summarization_toxicity"
    GENERAL_SEMANTIC_ROBUSTNESS = "general_semantic_robustness"
    ACCURACY = "accuracy"
    QA_ACCURACY = "qa_accuracy"
    QA_ACCURACY_BY_LLM = "qa_accuracy_by_llm"
    QA_ACCURACY_SEMANTIC_ROBUSTNESS = "qa_accuracy_semantic_robustness"
    SUMMARIZATION_ACCURACY = "summarization_accuracy"
    SUMMARIZATION_ACCURACY_SEMANTIC_ROBUSTNESS = "summarization_accuracy_semantic_robustness"
    CLASSIFICATION_ACCURACY = "classification_accuracy"
    CLASSIFICATION_ACCURACY_SEMANTIC_ROBUSTNESS = "classification_accuracy_semantic_robustness"

    def __str__(self):
        """
        Returns a prettified name
        """
        return self.name.replace("_", " ")


# TODO: Implement configurations for eval algorithms: by_llm needs to configure evaluator LLM(s)
EVALUATIONS = tuple(
    [
        EvalAlgorithm.QA_ACCURACY,
        EvalAlgorithm.QA_ACCURACY_BY_LLM,
        EvalAlgorithm.FACTUAL_KNOWLEDGE,
    ]
)
