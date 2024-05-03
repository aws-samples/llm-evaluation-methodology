# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Common classes for configuring fmeval model runners in the prompt engineering app"""
# Python Built-Ins:
from dataclasses import dataclass
from enum import Enum
import json
from typing import Dict, List, Union

# External Dependencies:
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.composers import Composer


class ModelType(Enum):
    """Supported types of model deployment"""

    BEDROCK: str = "bedrock"
    OPENAI: str = "openai"
    SAGEMAKER: str = "sagemaker"


@dataclass
class BaseModelConfig:
    """Base Model Configuration that all subtypes should extend

    Model Configurations store parameters associated with model deployment and invocation and *NOT*
    any hyperparameters (e.g. temperature, top P, etc) that should be put in an InferenceConfig
    instead.
    """

    model_id: str
    model_type: ModelType


@dataclass
class BaseInferenceConfig:
    """Base Inference Configuration that all subtypes should extend

    Inference Configurations store parameters associated with model inference and *NOT*
    any model deployment parameters (e.g. model_id, container_cpu_units, container_mem_mib, etc)
    that should be put in a ModelConfig instead.

    There are some extremely common hyperparameters like `temperature` that we could promote to
    here, but for now we've left it empty to avoid enforcing *all* model types must accept them

    TODO: Should we just treat inference config as an open dict & not attempt to type it?
    """

    config_id: str


class StructuredContentComposer(Composer):
    """Like fmeval's JsonContentComposer, but supports structured JSON input data not just strings.

    Because the JsonContentComposer initially takes `json.dumps(data)` before filling the template,
    it's not possible to fill with arbitrary JSON even though the syntax '{"data": $prompt}' makes
    it look like it might be. This alternative composer supports that use-case.
    """

    PLACEHOLDER = "prompt"

    def __init__(self, template: str):
        super().__init__(template=template, placeholder=self.PLACEHOLDER)

    def compose(self, data: str) -> Union[str, List, Dict]:
        try:
            return json.loads(self._get_filled_in_template(data))
        except Exception as e:
            raise EvalAlgorithmClientError(
                f"Unable to load a JSON object with template '{self.vanilla_template.template}' using data {data} ",
                e,
            )
