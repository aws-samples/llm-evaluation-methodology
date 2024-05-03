# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Main entry point for example prompt template engineering app based on Streamlit
"""
# Python Built-Ins:
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
from textwrap import dedent
from typing import List

# External Dependencies:
from fmeval.eval_algorithms import EvalOutput  # For type hints only! See `datamodel` below
import pandas as pd
import ray
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Local Dependencies:
# It'd be nice to define this folder as a module and have relative `from . import ...`s, but:
# https://github.com/streamlit/streamlit/issues/662
#
# `datamodel` contains classes & helpers intended to draw the boundary of abstraction between
# objects this UI deals with (and might store in configuration databases), and the requirements of
# underlying fmeval library.
from datamodel.dataset import Dataset, PromptTemplate
from datamodel.evaluations.base import EVALUATIONS
from datamodel.evaluations.evaluator import Evaluator
from datamodel.model import BaseModelConfig, BaseInferenceConfig, get_model_runner, MODELS
from datamodel.prompt import PromptTemplate
from datamodel.serialization import json_dumps
from util.auth import CognitoAuth

# Configuration:
COGNITO_SECRET_NAME = os.environ.get("COGNITO_SECRET_NAME")
DISPLAY_MAX_QUESTIONS = 10
DEFAULT_PROMPT_TEMPLATE = """Human:
<context>
{doc}
</context>

<question>
{question}
</question>

Answer the question as if you were a student taking a test.

Assistant:
"""
DEFAULT_REF_ANSWER_FIELD = "answers"
MAX_EVALS_IN_MEMORY = 5


# Streamlit page config needs to be before we create the authenticator for some reason:
st.set_page_config(page_title="Prompt Template Studio", page_icon=":straight_ruler:", layout="wide")
if COGNITO_SECRET_NAME:
    authenticator = CognitoAuth.get_authenticator(COGNITO_SECRET_NAME)
else:
    print(
        "WARNING: Running with no user authentication enabled - publicly accessible! Configure "
        "COGNITO_SECRET_NAME to enable authentication via Amazon Cognito."
    )
    authenticator = None


@dataclass(frozen=True)
class EvaluationRecord:
    """The attributes of an evaluation job that are persisted in Streamlit session state for display

    Attributes
    ----------
    model_config :
        The model configuration the evaluation was run with
    inference_config :
        The inference configuration the evaluation was run with
    summary :
        The `EvalOutput` object(s) returned by the evaluation job
    detail :
        Content of the JSON-Lines file providing sample-level invocation and result details from
        the dataset (kept ready for download from the UI)
    """

    detail: str
    inference_config: BaseInferenceConfig
    model_config: BaseModelConfig
    prompt_template: PromptTemplate
    start_time: datetime
    summary: List[EvalOutput]
    time_taken: timedelta


def main() -> None:
    """Main Streamlit app entry point"""
    # Ensure user is logged in, if cognito auth configured:
    if authenticator and not authenticator.login():
        return st.stop()

    initialize()
    title()
    file_upload()
    if not st.session_state.dataset:
        return
    button_start_eval()
    st.sidebar.divider()
    model_controls()
    st.sidebar.divider()
    eval_algo_controls()
    st.sidebar.divider()
    prompt_template()
    dataset()
    with st.sidebar:
        st.divider()
        if authenticator:
            st.button(f"Log out `{authenticator.get_username()}`", on_click=logout)
        else:
            st.warning(
                "User authentication not configured: This app may be publicly accessible",
                icon="⚠️",
            )

    render_evaluations()


def eval_sel_state(eval_algo: str) -> str:
    """Name of the streamlit state key recording whether a particular model ID is selected"""
    return f"eval_selected_{eval_algo}"


def model_sel_state(model_id: str) -> str:
    """Name of the streamlit state key recording whether a particular eval algo is selected"""
    return f"model_selected_{model_id}"


def initialize() -> None:
    """Initialize the Streamlit Session State."""
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    for mdl, _ in MODELS:
        key = model_sel_state(mdl.model_id)
        if key not in st.session_state:
            st.session_state[key] = False
    for eval in EVALUATIONS:
        key = eval_sel_state(eval.value)
        if key not in st.session_state:
            st.session_state[key] = False
    if "evaluations" not in st.session_state:
        st.session_state.evaluations = []
    if "start_eval" not in st.session_state:
        st.session_state.start_eval = False
    if "container_prompt" not in st.session_state:
        st.session_state.container_prompt = st.empty()
    if "ref_answer_field" not in st.session_state:
        st.session_state.ref_answer_field = DEFAULT_REF_ANSWER_FIELD


def logout() -> None:
    authenticator.logout()
    if "dataset" in st.session_state:
        st.session_state.dataset = None
    if "evaluations" in st.session_state:
        st.session_state.evaluations = []
    if "start_eval" in st.session_state:
        st.session_state.start_eval = False
    if "container_prompt" in st.session_state:
        st.session_state.container_prompt = st.empty()
    if "ref_answer_field" in st.session_state:
        st.session_state.ref_answer_field = DEFAULT_REF_ANSWER_FIELD


def title() -> None:
    """Simply display the title/header on the page."""
    st.markdown(
        "<h3 style='text-align: center'>LLM Evaluation - Prompt Studio</h3>",
        unsafe_allow_html=True,
    )


def file_upload() -> None:
    """Display the File Upload Widget if the Data is not loaded yet."""
    if st.session_state.dataset:
        return
    st.file_uploader(
        label="Upload the dataset in JSONL format.",
        accept_multiple_files=False,
        key="dataset_upload",
        on_change=file_upload_handler,
    )
    st.markdown(
        dedent(
            """
            ### Supported datasets

            Your dataset should be provided in [JSON-Lines format](https://jsonlines.org/), where
            each record can have whatever fields will be used by your prompt template (for example
            'doc', 'question', etc.), plus a specific named field containing the reference /
            'correct' answer(s) for the final prompt (`answers` by default).

            The answers field may be either a JSON list e.g. `["42", "to run marathons"]` or a
            single string wherein `<OR>` will be regarded as a separator - like the following
            example:

            ```json
            {"question": "I believe the meaning of life is", "answers": "42<OR>To run marathons."}
            ```
            """
        ).strip()
    )


def file_upload_handler() -> None:
    """Set the dataset_loaded flag to True after file is uploaded."""
    fileupload: UploadedFile = st.session_state.get("dataset_upload")
    if not fileupload:
        return
    preview_df = Dataset.preview(data=fileupload, n_rows=DISPLAY_MAX_QUESTIONS)
    st.session_state.dataset = Dataset(
        fields=preview_df.columns.to_list(),
        id="squad",
        ref_answer_field=st.session_state.ref_answer_field,
        data=fileupload,
    )
    st.session_state.pd_eval_matrix = preview_df
    clear_evaluations()  # Clear previous evaluation results to avoid confusion


def button_start_eval() -> None:
    """This button when pressed will start the evaluation process."""
    st.sidebar.button(
        label="Start Evaluation",
        key="button_start_eval",
        on_click=button_start_eval_handler,
        type="primary",
    )


def button_start_eval_handler() -> None:
    """What to do when the button is pressed :D"""
    st.session_state.start_eval = True
    start_evaluation()


def model_controls() -> None:
    """Display the controls to select and configure the bedrock models."""
    st.sidebar.subheader("Select a Model ID")
    model_ids = [model_config.model_id for model_config, infcfgs in MODELS if len(infcfgs)]
    st.sidebar.selectbox(label="Model IDs", options=model_ids, key="model_id")
    if "model_id" not in st.session_state:
        st.session_state.model_id = model_ids[0]
    for model_cfg, inf_cfg in MODELS:
        if model_cfg.model_id == st.session_state.model_id:
            st.session_state.model_cfg = model_cfg
            st.session_state.inf_cfg = inf_cfg[0]
    return


def eval_algo_controls() -> None:
    """Provide a list of available evaluation algos to choose from."""
    st.sidebar.subheader("Select an Evaluation Algorithm")
    eval_algos = [eval_algo.value for eval_algo in EVALUATIONS]
    st.sidebar.selectbox(label="Evaluation Algorithms", options=eval_algos, key="eval_algo")
    if "eval_algo" not in st.session_state:
        st.session_state.eval_algo = eval_algos[0]
    for eval in EVALUATIONS:
        if eval.value == st.session_state.eval_algo:
            st.session_state.eval_name = eval
    return


def prompt_template() -> None:
    """The area for composing/editing the actual prompt template under test"""
    st.text_area(
        label="Prompt Template",
        key="textarea_prompt_template",
        height=256,
        value=DEFAULT_PROMPT_TEMPLATE,
    )


def dataset() -> None:
    """Sidebar controls to update the dataset, when one is already loaded"""
    if not st.session_state.dataset:
        return
    st.sidebar.subheader("Dataset")

    def update_dataset_answer_field():
        st.session_state.dataset.ref_answer_field = st.session_state.ref_answer_field

    st.sidebar.text_input(
        "Reference answer field",
        key="ref_answer_field",
        on_change=update_dataset_answer_field,
        value=DEFAULT_REF_ANSWER_FIELD,
    )
    st.sidebar.file_uploader(
        "Switch dataset (JSONL)",
        accept_multiple_files=False,
        key="dataset_upload",
        on_change=file_upload_handler,
    )
    with st.expander(label=":file_cabinet: Dataset", expanded=False):
        with st.container(height=800):
            st.table(st.session_state.pd_eval_matrix)


def start_evaluation() -> None:
    time_start = datetime.now()
    pt = PromptTemplate(template=st.session_state.textarea_prompt_template)
    dataset: Dataset = st.session_state.dataset
    try:
        inf_cfg = st.session_state.inf_cfg
        if hasattr(inf_cfg, "messages_api"):
            is_msg_api = inf_cfg.messages_api
        else:
            is_msg_api = False
        with dataset.fmeval_data_config(pt, claude_postproc=is_msg_api) as data_config:
            model_cfg = st.session_state.model_cfg
            eval_name = st.session_state.eval_name
            model_runner = get_model_runner(model_cfg, inf_cfg)
            eval_algo = Evaluator(eval_name)

            eval_output = eval_algo.evaluate(
                model=model_runner,
                dataset_config=data_config,
            )
            # Unfortunately we can't easily customize/set the output file location:
            # https://github.com/aws/fmeval/issues/165
            eval_path = f"/tmp/eval_results/{eval_name.value}_{data_config.dataset_name}.jsonl"
            with open(eval_path) as fres:
                eval_record = EvaluationRecord(
                    detail=fres.read(),
                    inference_config=inf_cfg,
                    model_config=model_cfg,
                    prompt_template=pt,
                    start_time=time_start,
                    summary=eval_output,
                    time_taken=datetime.now() - time_start,
                )
            st.session_state.evaluations.append(eval_record)
            while len(st.session_state.evaluations) > MAX_EVALS_IN_MEMORY:
                st.session_state.evaluations.pop(0)
        st.session_state.start_eval = False
    except Exception as err:
        st.exception(err)
    finally:
        if ray.is_initialized:
            ray.shutdown()


def render_evaluation(evaluation: EvaluationRecord, ix: int = 0):
    with st.expander(
        label=f"[{evaluation.start_time}] Results for {evaluation.model_config.model_id}",
        expanded=False,
    ):
        with st.container(height=800):
            col_dl_summary, col_dl_detail, col_clear = st.columns(3)
            with col_dl_summary:
                st.download_button(
                    data=json_dumps(
                        {
                            "model": evaluation.model_config,
                            "inference_config": evaluation.inference_config,
                            "prompt_template": evaluation.prompt_template.template,
                            "results": evaluation.summary,
                            "start_time": evaluation.start_time,
                        }
                    ),
                    file_name="eval_summary.json",
                    key=f"button_dl_summary_{ix}",
                    label="Download Summary",
                    mime="application/json",
                )
            with col_dl_detail:
                st.download_button(
                    data=evaluation.detail,
                    file_name="eval_results.jsonl",
                    key=f"button_dl_detail_{ix}",
                    label="Download Per-Datum Results",
                    mime="application/jsonl",
                )
            with col_clear:
                st.button(
                    label=":heavy_multiplication_x: Clear This Evaluation",
                    key=f"button_clear_eval_{ix}",
                    on_click=lambda: st.session_state.evaluations.pop(ix),
                )

            st.markdown("#### Summary")
            for eval_out in evaluation.summary:
                st.table(
                    pd.DataFrame(
                        [{"Value": score.value} for score in eval_out.dataset_scores],
                        index=pd.Series(
                            [score.name for score in eval_out.dataset_scores], name="Metric"
                        ),
                    )
                )
            st.write(f":mantelpiece_clock: Evaluation took {evaluation.time_taken} to run")
            st.text_area(
                disabled=True,
                label="Prompt Template",
                key=f"textarea_past_prompt_{ix}",
                height=256,
                value=evaluation.prompt_template.template,
            )

            st.markdown("#### Detailed Results")
            for ix_sample, line in enumerate(evaluation.detail.split("\n")):
                if ix_sample >= DISPLAY_MAX_QUESTIONS:
                    break
                if not line:
                    continue
                mo = json.loads(line)
                st.divider()
                st.write(f'Expected Answer:\n{mo["target_output"]}')
                st.write(f'Model Output:\n{mo["model_output"]}')
                st.write(f'Scores: {mo["scores"]}')


def render_evaluations():
    for ix, evaluation in enumerate(st.session_state.evaluations):
        render_evaluation(evaluation, ix)
    n_evals = len(st.session_state.evaluations)
    if n_evals == 0:
        return

    col_btn, col_msg = st.columns([0.3, 0.7])
    with col_btn:
        st.button(
            label=":heavy_multiplication_x: Clear Evaluation Job History",
            key="button_clear_evaluations",
            on_click=clear_evaluations,
        )
    if n_evals >= MAX_EVALS_IN_MEMORY:
        col_msg.markdown(
            ":warning: You have **%s** evaluations in history, which is the configured maximum. "
            "Running new evaluations will cause the oldest result to be automatically cleared."
            % (n_evals,)
        )
    elif n_evals > 2:
        col_msg.markdown(
            ":information_source: Keeping many evaluations in history can consume significant "
            "memory, especially for large datasets. Old evaluations will be automatically dropped "
            "after you exceed the configured limit of **%s**." % (MAX_EVALS_IN_MEMORY,)
        )


def clear_evaluations():
    st.session_state.evaluations = []


if __name__ == "__main__":
    main()
