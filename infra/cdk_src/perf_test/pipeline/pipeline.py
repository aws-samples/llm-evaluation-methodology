# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Python Built-Ins:
from typing import Optional, Union

# External Dependencies:
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession, LocalPipelineSession
from sagemaker.workflow.steps import ProcessingStep


def get_pipeline(
    role: str,
    region_name: str,
    image_uri: str,
    input_bucket_name: str,
    output_bucket_name: str,
    input_bucket_prefix: str = "experiments/mistral/",
    pipeline_session: Optional[Union[PipelineSession, LocalPipelineSession]] = None,
):
    """Generate SageMaker Pipeline definition to run FMBench in a processing job"""
    if not pipeline_session:
        pipeline_session = PipelineSession()
    parameters = []

    config_s3uri = ParameterString(
        name="ConfigS3Uri",
        default_value=f"s3://{input_bucket_name}/{input_bucket_prefix}",
    )
    parameters += [config_s3uri]

    instance_type = ParameterString(
        name="ClientInstanceType",
        default_value="ml.c5.xlarge",
    )
    parameters += [instance_type]

    processor_image_uri = ParameterString(
        name="FMBenchImageUri",
        default_value=image_uri,
    )
    parameters += [processor_image_uri]

    output_bucket_param = ParameterString(
        name="OutputS3BucketName",
        default_value=output_bucket_name,
    )
    parameters += [output_bucket_param]

    print("Role", role)
    processor = Processor(
        role=role,
        image_uri=processor_image_uri,
        entrypoint=[
            "conda",
            "run",
            "-n",
            "fmbench_python311",
            "sh",
            "-c",
            "fmbench --config-file /opt/ml/processing/config/config.yaml",
        ],
        instance_type=instance_type,
        instance_count=1,
        volume_size_in_gb=30,
        max_runtime_in_seconds=300 * 3,
        sagemaker_session=pipeline_session,
        env={
            "AWS_DEFAULT_REGION": region_name,
            "WRITE_BUCKET": output_bucket_param,
        },
    )

    step_args = processor.run(
        # add a processing input for an S3 bucket
        inputs=[
            ProcessingInput(
                source=config_s3uri,
                destination="/opt/ml/processing/config",
            ),
        ],
        # Currently the processing job saves its output directly to S3 in a location configured
        # by the fmbench yaml, so there's no output from the job itself
        # TODO: Is it possible/nice to make use of SageMaker job output instead?
        # outputs=[
        #     ProcessingOutput(
        #         output_name="output",
        #         source="/opt/ml/processing/output",
        #     )
        # ],
    )

    step_process = ProcessingStep(name="FMBench", step_args=step_args)

    pipeline = Pipeline(
        name="FMBenchPipeline",
        parameters=parameters,
        steps=[
            step_process,
        ],
        sagemaker_session=pipeline_session,
    )

    return pipeline
