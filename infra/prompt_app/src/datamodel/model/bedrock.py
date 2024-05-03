# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Bedrock-specific classes for wrapping fmeval ModelRunners"""
# Python Built-Ins:
from dataclasses import asdict, dataclass, field
import json
from logging import getLogger
import re
from typing import Any, Dict, List, Optional

# External Dependencies:
from fmeval.constants import MIME_TYPE_JSON
from fmeval.model_runners.bedrock_model_runner import BedrockModelRunner

# Local Dependencies:
from .base import BaseInferenceConfig, BaseModelConfig, ModelType, StructuredContentComposer


logger = getLogger(__name__)


@dataclass
class BedrockModelConfig(BaseModelConfig):
    """Deployment configuration for Amazon Bedrock (excluding inference parameters)"""

    model_type: ModelType = ModelType.BEDROCK


@dataclass
class BedrockClaudeInferenceConfigBase(BaseInferenceConfig):
    """Partial model for inference configuration of Anthropic Claude text gen models on Bedrock

    Includes the subset of properties that are common to Claude v1, v2, and v3
    """

    temperature: float = 0.5
    top_p: Optional[float] = 1
    top_k: Optional[float] = 250


@dataclass
class BedrockClaudeV1V2InferenceConfig(BedrockClaudeInferenceConfigBase):
    """Type model for inference configuration of Anthropic Claude v3 text gen models on Bedrock"""

    anthropic_version: str = "bedrock-2023-05-31"
    max_tokens_to_sample: int = 200
    stop_sequences: Optional[List[str]] = field(default_factory=lambda: ["\n\nHuman:"])


@dataclass
class BedrockClaudeV3InferenceConfig(BedrockClaudeInferenceConfigBase):
    """Type model for inference configuration of Anthropic Claude v3 text gen models on Bedrock"""

    anthropic_version: str = "bedrock-2023-05-31"
    max_tokens: int = 200
    messages_api: bool = True


@dataclass
class BedrockCohereInferenceConfig(BaseInferenceConfig):
    """Type model for inference configuration of Cohere models on Amazon Bedrock"""

    max_tokens: int = 200
    p: float = 1
    k: int = 0
    temperature: float = 0.5


@dataclass
class BedrockLlamaInferenceConfig(BaseInferenceConfig):
    """Type model for inference configuration of Llama models on Amazon Bedrock"""

    max_gen_len: int = 512
    temperature: float = 0.5
    top_p: float = 0.9


@dataclass
class BedrockTitanInferenceConfig(BaseInferenceConfig):
    """Type model for inference configuration of Amazon Titan text gen models on Amazon Bedrock"""

    maxTokenCount: int = 512
    temperature: float = 0
    topP: float = 1


class StructuredBedrockModelRunner(BedrockModelRunner):
    """Wrapper for fmeval `BedrockModelRunner` modifies the Composer to support structured prompts

    Use this class instead of BedrockModelRunner to allow passing structured JSON data through the
    "prompt" instead of just a string (i.e. for the Claude v3 Messages API)
    """

    def __init__(
        self,
        model_id: str,
        content_template: str,
        output: Optional[str] = None,
        log_probability: Optional[str] = None,
        content_type: str = MIME_TYPE_JSON,
        accept_type: str = MIME_TYPE_JSON,
    ):
        super().__init__(
            model_id, content_template, output, log_probability, content_type, accept_type
        )
        logger.info("Overriding composer for BedrockModelRunner with structured messaging API")
        self._composer = StructuredContentComposer(content_template)


def get_fmeval_bedrock_runner_settings(
    model_config: BedrockModelConfig,
    inf_config: BaseInferenceConfig,
) -> Dict[str, Any]:
    """Build an fmeval `BedrockModelRunner` from app ModelConfig and InferenceConfig objects"""

    def inf_config_params_without_id(cfg, skip_other_fields: List[str] = []):
        return {
            k: v for k, v in asdict(cfg).items() if k not in (skip_other_fields + ["config_id"])
        }

    if model_config.model_id.startswith("amazon.titan-t"):
        return {
            "content_template": (
                '{"inputText": $prompt, "textGenerationConfig": '
                + json.dumps(inf_config_params_without_id(inf_config))
                + "}"
            ),
            "model_id": model_config.model_id,
            "output": "results[0].outputText",
        }
    elif model_config.model_id.startswith("anthropic.claude-"):
        tpl_base = json.dumps(
            inf_config_params_without_id(inf_config, skip_other_fields=["messages_api"])
        )
        result = {"model_id": model_config.model_id}
        if hasattr(inf_config, "messages_api") and inf_config.messages_api:
            result["content_template"] = '{"messages": $prompt, ' + tpl_base[1:]
            result["output"] = "content[0].text"
        else:
            result["content_template"] = '{"prompt": $prompt, ' + tpl_base[1:]
            result["output"] = "completion"
        return result
    elif model_config.model_id.startswith("cohere.command"):
        return {
            "content_template": (
                '{"prompt": $prompt, ' + json.dumps(inf_config_params_without_id(inf_config))[1:]
            ),
            "model_id": model_config.model_id,
            "output": "generations[0].text",
        }
    elif model_config.model_id.startswith("meta.llama"):
        return {
            "content_template": (
                '{"prompt": $prompt, ' + json.dumps(inf_config_params_without_id(inf_config))[1:]
            ),
            "model_id": model_config.model_id,
            "output": "generation",
        }
    raise NotImplementedError(f"Model {model_config.model_id} not yet supported")


def claude_text_to_structured_messages(prompt: str) -> Dict:
    """Convert a Claude text prompt with 'Human:', 'Assistant:' etc labels to messages API

    https://docs.anthropic.com/claude/reference/messages_post
    """
    matcher = re.compile(r"^(Human|System|Assistant):", re.MULTILINE)
    role_map = {"Human": "user", "System": "system", "Assistant": "assistant"}
    messages = []
    role = None
    lastIx = 0
    for res in matcher.finditer(prompt):
        prev_text = prompt[lastIx : res.start()].strip()
        if prev_text:
            if not role:
                raise ValueError(
                    f"No 'Human'/'Assistant'/'System' role defined for text: {prev_text}"
                )
            messages.append(
                {"role": role_map[role], "content": [{"type": "text", "text": prev_text}]}
            )
        role = res.group(1)
        lastIx = res.end()
    prev_text = prompt[lastIx:].strip()
    if prev_text:
        if not role:
            raise ValueError(f"No 'Human'/'Assistant'/'System' role defined for text: {prev_text}")
        messages.append({"role": role_map[role], "content": [{"type": "text", "text": prev_text}]})
    return messages
