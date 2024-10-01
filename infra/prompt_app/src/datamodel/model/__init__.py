# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities for describing model deployments and inference configurations
"""
# Python Built-Ins:
import json
from logging import getLogger
from typing import Tuple
from uuid import uuid4

# External Dependencies:
from fmeval.model_runners.bedrock_model_runner import BedrockModelRunner
from fmeval.model_runners.model_runner import ModelRunner

# Local Dependencies:
from .base import BaseInferenceConfig, BaseModelConfig, ModelType
from .bedrock import (
    BedrockClaudeV1V2InferenceConfig,
    BedrockClaudeV3InferenceConfig,
    BedrockCohereInferenceConfig,
    BedrockLlamaInferenceConfig,
    BedrockModelConfig,
    BedrockTitanInferenceConfig,
    get_fmeval_bedrock_runner_settings,
    StructuredBedrockModelRunner,
)
from .openai import ChatGPTInferenceConfig, ChatGPTModelConfig, ChatGPTModelRunner

logger = getLogger("model")

# TODO: Replace hard-coded model configurations with dynamic data!
# The list of model deployments, inference configurations, and mappings between them, would ideally
# be managed as data somewhere - which is why all the *Config classes should be serializable
dflt_claude_ifconfig = BedrockClaudeV1V2InferenceConfig(config_id=uuid4().hex)
dflt_claude3_ifconfig = BedrockClaudeV3InferenceConfig(config_id=uuid4().hex)
dflt_cohere_ifconfig = BedrockCohereInferenceConfig(config_id=uuid4().hex)
dflt_llama_ifconfig = BedrockLlamaInferenceConfig(config_id=uuid4().hex)
dflt_titan_ifconfig = BedrockTitanInferenceConfig(config_id=uuid4().hex)
dflt_openai_ifconfig = ChatGPTInferenceConfig(config_id=uuid4().hex)
MODELS: Tuple[Tuple[BaseModelConfig, Tuple[BaseInferenceConfig]]] = tuple(
    [
        (
            BedrockModelConfig(model_id="anthropic.claude-3-haiku-20240307-v1:0"),
            (dflt_claude3_ifconfig,),
        ),
        (
            BedrockModelConfig(model_id="anthropic.claude-3-sonnet-20240229-v1:0"),
            (dflt_claude3_ifconfig,),
        ),
        (
            BedrockModelConfig(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"),
            (dflt_claude3_ifconfig,),
        ),
        (
            BedrockModelConfig(model_id="anthropic.claude-3-opus-20240229-v1:0"),
            (dflt_claude3_ifconfig,),
        ),
        (BedrockModelConfig(model_id="anthropic.claude-v2:1"), (dflt_claude_ifconfig,)),
        (BedrockModelConfig(model_id="anthropic.claude-v2"), (dflt_claude_ifconfig,)),
        (BedrockModelConfig(model_id="anthropic.claude-instant-v1"), (dflt_claude_ifconfig,)),
        (BedrockModelConfig(model_id="cohere.command-text-v14"), (dflt_cohere_ifconfig,)),
        (
            BedrockModelConfig(model_id="cohere.command-light-text-v14:7:4k"),
            (dflt_cohere_ifconfig,),
        ),
        (BedrockModelConfig(model_id="ai21.j2-ultra-v1"), tuple()),
        (BedrockModelConfig(model_id="ai21.j2-mid-v1"), tuple()),
        (BedrockModelConfig(model_id="amazon.titan-text-express-v1"), (dflt_titan_ifconfig,)),
        (BedrockModelConfig(model_id="amazon.titan-text-lite-v1"), (dflt_titan_ifconfig,)),
        (BedrockModelConfig(model_id="meta.llama3-70b-instruct-v1:0"), (dflt_llama_ifconfig,)),
        (BedrockModelConfig(model_id="meta.llama3-8b-instruct-v1:0"), (dflt_llama_ifconfig,)),
        # (BedrockModelConfig(model_id="meta.llama2-70b-chat-v1"), (dflt_llama_ifconfig,)),
        # (BedrockModelConfig(model_id="meta.llama2-13b-chat-v1"), (dflt_llama_ifconfig,)),
        (ChatGPTModelConfig(api_key_secret="openai_key"), (dflt_openai_ifconfig,)),
        (
            ChatGPTModelConfig(api_key_secret="openai_key", model_id="gpt-4"),
            (dflt_openai_ifconfig,),
        ),
        (
            ChatGPTModelConfig(api_key_secret="openai_key", model_id="gpt-4o-mini"),
            (dflt_openai_ifconfig,),
        ),
        (
            ChatGPTModelConfig(api_key_secret="openai_key", model_id="gpt-4o"),
            (dflt_openai_ifconfig,),
        ),
    ]
)


def get_model_runner(
    model_config: BaseModelConfig,
    inf_config: BaseInferenceConfig,
) -> ModelRunner:
    """Get an fmeval ModelRunner for a given pair of model config and inference parameter config

    Parameters
    ----------
    model_config :
        Describes the deployment of the model itself (information necessary to call its API)
    inf_config :
        Describes the inference parameters to use (such as temperature, top P, etc)
    """
    if model_config.model_type == ModelType.BEDROCK:
        cfg = get_fmeval_bedrock_runner_settings(model_config=model_config, inf_config=inf_config)
        logger.info("Creating BedrockModelRunner with settings:\n%s", json.dumps(cfg, indent=2))
        if hasattr(inf_config, "messages_api") and inf_config.messages_api:
            return StructuredBedrockModelRunner(**cfg)
        else:
            return BedrockModelRunner(**cfg)
    elif model_config.model_type == ModelType.OPENAI:
        return ChatGPTModelRunner(model_config=model_config, inf_config=inf_config)
    elif model_config.model_type == ModelType.SAGEMAKER:
        raise NotImplementedError("TODO: Implement SageMaker endpoint support!")
    else:
        raise ValueError(f"Unknown model type: {model_config.type}")
