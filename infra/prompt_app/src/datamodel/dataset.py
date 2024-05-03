# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities for storing LLM evaluation datasets and using them with fmeval
"""
# Python Built-Ins:
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from io import BytesIO
import json
from tempfile import NamedTemporaryFile
from typing import Any, Generator, List, Union

# External Dependencies:
from fmeval.constants import MIME_TYPE_JSONLINES
from fmeval.data_loaders.data_config import DataConfig
import pandas as pd

# Local Dependencies:
from .model.bedrock import claude_text_to_structured_messages
from .prompt import PromptTemplate


@dataclass
class Dataset:
    """Configuration-serializable class to represent a dataset and implement fmeval helper methods

    Attributes
    ----------
    data :
        Either a direct BytesIO (in the case of streamlit UploadedFile), or a string URI
        TODO implement S3 functionality, pointing to the raw dataset from the user
    id :
        An identifying name for the dataset (required by fmeval, should be filename-friendly)
    fields :
        List of field names/keys present in the dataset
    ref_answer_field :
        Name of the dataset field/key containing the target (reference) answer(s)
    """

    data: Union[BytesIO, str]
    id: str
    fields: List[str]
    ref_answer_field: str

    @staticmethod
    def preview(data: Union[BytesIO, str], n_rows: int = 3) -> pd.DataFrame:
        """Load the first few rows of the dataset into a Pandas DataFrame for preview"""
        with open(data, "r") if isinstance(data, str) else nullcontext(data) as f:
            f.seek(0)
            records = []
            for ix, line in enumerate(f):
                if ix >= n_rows:
                    break
                if not line:  # Skip any empty lines
                    continue
                records.append(json.loads(line))
            f.seek(0)
        return pd.DataFrame(records)

    @contextmanager
    def fmeval_data_config(
        self, prompt_template: PromptTemplate, claude_postproc: bool = False
    ) -> Generator[DataConfig, Any, None]:
        """Yield an fmeval DataConfig for this dataset

        Use this method as a context manager, like:
        >>> with dataset.fmeval_data_config(prompt_template) as data_config:
        ...     eval_algo.evaluate(dataset_config=data_config, ...)

        The reason is that we do prompt template fulfilment *outside* fmeval (to support additional
        fields), so create a temporary file to do it. Using the context manager `with` pattern
        ensures that temporary resources are cleaned up even if your evaluation code errors out.
        """
        with open(self.data, "r") if isinstance(self.data, str) else nullcontext(self.data) as fraw:
            fraw.seek(0)
            # fmeval checks input file extensions, so need appropriate suffix:
            with NamedTemporaryFile(mode="w", suffix=".jsonl") as f:
                for line in fraw:
                    if not line:  # Skip any empty lines
                        continue
                    datum = json.loads(line)
                    prompt = prompt_template.fulfil(datum)
                    if claude_postproc:
                        prompt = json.dumps(claude_text_to_structured_messages(prompt))
                    answers = (
                        "<OR>".join(datum[self.ref_answer_field])
                        if isinstance(datum[self.ref_answer_field], list)
                        else datum[self.ref_answer_field]
                    )
                    f.write(
                        json.dumps(
                            {
                                "prompt": prompt,
                                "answers": answers,
                            }
                        )
                        + "\n"
                    )
                f.seek(0)
                fraw.seek(0)

                yield DataConfig(
                    dataset_name=self.id,
                    dataset_uri=f.name,
                    dataset_mime_type=MIME_TYPE_JSONLINES,
                    model_input_location="prompt",
                    target_output_location="answers",
                )
