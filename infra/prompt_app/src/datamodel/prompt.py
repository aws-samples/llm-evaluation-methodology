# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities for storing/configuring prompt templates 
"""
# Python Built-Ins:
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Configuration-serializable class for a prompt template"""

    template: str

    def fulfil(self, datum: dict) -> str:
        # TODO: Consider using Template(self.template).safe_substitute(datum) instead?
        return self.template.format(**datum)
