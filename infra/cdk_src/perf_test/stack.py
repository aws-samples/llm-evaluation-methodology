# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK Stack for running LLM latency/performance tests in a SageMaker Pipeline"""
# Python Built-Ins:
import os

# External Dependencies:
from aws_cdk import (
    Aws,
    aws_ecr as ecr,
    aws_ecr_assets as ecr_assets,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_lambda_python_alpha as lambda_python,
    aws_s3 as s3,
    aws_s3_deployment as s3d,
    aws_sagemaker as sagemaker,
    CfnOutput,
    custom_resources as cr,
    CustomResource,
    Duration,
    RemovalPolicy,
    Stack,
)
import cdk_ecr_deployment as ecrd
from constructs import Construct

# Local Dependencies:
from . import functions as F
from .pipeline import assets as A, image as I, pipeline

class LLMPerfTestStack(Stack):
    """CDK Stack for running LLM latency/performance tests in a SageMaker Pipeline"""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        path_functions = os.path.dirname(os.path.abspath(F.__file__))

        # Static config (tokenizers, etc) bucket:
        static_config_bucket = s3.Bucket(
            self,
            "StaticConfigBucket",
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            bucket_name=f"sagemaker-fmbench-read-{Aws.REGION}-{Aws.ACCOUNT_ID}",
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
        )
        # Bucket for writing experiment outputs:
        output_bucket = s3.Bucket(
            self,
            "OutputBucket",
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            bucket_name=f"sagemaker-fmbench-output-{Aws.REGION}-{Aws.ACCOUNT_ID}",
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
        )
        # Bucket for user experiment configurations:
        config_bucket = s3.Bucket(
            self,
            "ConfigBucket",
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            bucket_name=f"sagemaker-fmbench-config-{Aws.REGION}-{Aws.ACCOUNT_ID}",
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Copy FMBench data assets to the static read bucket:
        cr_handler = lambda_python.PythonFunction(
            self,
            "StaticConfigCRFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            entry=path_functions,
            index="handler.py",
            handler="on_event",
            initial_policy=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["s3:*"],
                    resources=[
                        static_config_bucket.bucket_arn,
                        static_config_bucket.arn_for_objects("*"),
                    ],
                ),
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["s3:GetObject"],
                    resources=[
                        s3.Bucket.from_bucket_name(
                            self, "fmbench-source", bucket_name="aws-blogs-artifacts-public"
                        ).arn_for_objects("artifacts/ML-FMBT/*")
                    ],
                ),
            ],
            timeout=Duration.seconds(60 * 5),
            memory_size=1024,
        )
        provider = cr.Provider(self, "StaticConfigCRProvider", on_event_handler=cr_handler)
        CustomResource(
            self,
            "StaticConfigCopy",
            service_token=provider.service_token,
            properties={"BucketName": static_config_bucket.bucket_name},
        )

        # Copy sample experiment configs assets to the config input bucket:
        path_assets = os.path.dirname(os.path.abspath(A.__file__))
        s3d.BucketDeployment(
            self,
            "DeployAssets",
            sources=[s3d.Source.asset(path_assets)],
            destination_bucket=config_bucket,
        )

        # Build the fmbench container:
        repo = ecr.Repository(
            self,
            "ImageRepo",
            repository_name="sm-fmbench",
            removal_policy=RemovalPolicy.DESTROY,
        )
        path_image = os.path.dirname(os.path.abspath(I.__file__))
        image_build = ecr_assets.DockerImageAsset(
            self,
            "ImageAssetBuild",
            directory=path_image,
            # cache_from=[ecr_assets.DockerCacheOption(registry=repo.repository_uri)],
            # cache_to=ecr_assets.DockerCacheOption(registry=repo.repository_uri)
            platform=ecr_assets.Platform.LINUX_AMD64,
        )
        image_uri = f"{repo.repository_uri}:latest"
        ecrd.ECRDeployment(
            self,
            "Image",
            src=ecrd.DockerImageName(image_build.image_uri),
            dest=ecrd.DockerImageName(image_uri),
        )

        # Create a role for the SageMaker pipeline:
        # (Role name/ARN needs to be statically defined, not a CDK token, for SageMaker Pipeline)
        role_name = f"sagemaker-pipeline-execution-{Aws.REGION}-{Aws.ACCOUNT_ID}"
        role_arn = f"arn:aws:iam::{Aws.ACCOUNT_ID}:role/{role_name}"
        role = iam.Role(
            self,
            "SageMakerPipelineRole",
            role_name=role_name,
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonBedrockFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AWSCloudFormationReadOnlyAccess"),
            ],
        )
        static_config_bucket.grant_read(role)
        config_bucket.grant_read_write(role)
        output_bucket.grant_read_write(role)

        # Create the SageMaker pipeline:
        sm_pipeline = pipeline.get_pipeline(
            role=role_arn,
            region_name=Aws.REGION,
            image_uri=image_uri,
            input_bucket_name=config_bucket.bucket_name,
            output_bucket_name=output_bucket.bucket_name,
        )
        sagemaker.CfnPipeline(
            self,
            "SageMakerPipeline",
            pipeline_name="sm-fmbench-pipeline",
            pipeline_definition={
                # https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-sagemaker-pipeline.html
                "PipelineDefinitionBody": sm_pipeline.definition()
            },
            role_arn=role.role_arn,
        )

        CfnOutput(self, "ExperimentConfigBucketName", value=config_bucket.bucket_name)
        CfnOutput(self, "OutputBucketName", value=output_bucket.bucket_name)
