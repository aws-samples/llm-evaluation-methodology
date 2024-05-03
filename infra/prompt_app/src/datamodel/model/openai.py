# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""OpenAI-specific ModelRunner classes for use with fmeval"""
# Python Built-Ins:
from dataclasses import dataclass
import json
from logging import getLogger
from typing import Optional, Tuple

# External Dependencies:
import boto3
from fmeval.model_runners.model_runner import ModelRunner
import requests

# Local Dependencies:
from .base import BaseInferenceConfig, BaseModelConfig, ModelType

logger = getLogger(__name__)
secman = boto3.client("secretsmanager")


@dataclass
class ChatGPTModelConfig(BaseModelConfig):
    """Deployment configuration for OpenAI ChatGPT (excluding inference parameters)"""

    api_key_secret: str = ""  # Can't actually make this mandatory because of dataclass inheritance
    model_id: str = "gpt-3.5-turbo"
    model_type: ModelType = ModelType.OPENAI
    url: str = "https://api.openai.com/v1/chat/completions"

    def __post_init__(self):
        if not self.api_key_secret:
            raise TypeError("__init__ missing 1 required argument: 'api_key_secret'")


@dataclass
class ChatGPTInferenceConfig(BaseInferenceConfig):
    """Inference configuration for OpenAI ChatGPT"""

    max_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0
    presence_penalty: float = 0


class ChatGPTModelRunner(ModelRunner):
    """An fmeval Model Runner class for invoking LLMs hosted on OpenAI

    Specify an AWS Secrets Manager secret in the config, which contains your OpenAI API key.
    """

    model_config: ChatGPTModelConfig
    inference_config: ChatGPTInferenceConfig

    def __init__(self, model_config: ChatGPTModelConfig, inf_config: ChatGPTInferenceConfig):
        self.model_config = model_config
        self.inference_config = inf_config
        try:
            self._api_key = secman.get_secret_value(SecretId=model_config.api_key_secret)[
                "SecretString"
            ]
        except Exception as e:
            logger.exception(
                "Failed to retrieve API Key from AWS Secrets Manager secret "
                f"'{model_config.api_key_secret}'. Check the secret exists and this app has "
                "AWS IAM permissions to read it."
            )
            raise e

    def predict(self, prompt: str) -> Tuple[Optional[str], None]:
        payload = json.dumps(
            {
                "model": self.model_config.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.inference_config.temperature,
                "top_p": self.inference_config.top_p,
                "n": 1,
                "stream": False,
                "presence_penalty": self.inference_config.presence_penalty,
                "frequency_penalty": self.inference_config.frequency_penalty,
            }
        )

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        response = requests.request("POST", self.model_config.url, headers=headers, data=payload)
        return json.loads(response.text)["choices"][0]["message"]["content"], None
