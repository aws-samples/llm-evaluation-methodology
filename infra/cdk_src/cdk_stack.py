# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK stack for AWS workshop on Large Language Model Evaluation"""
# Python Built-Ins:
from typing import Optional

# External Dependencies:
from aws_cdk import Stack, CfnParameter, CfnOutput
from constructs import Construct
from aws_cdk import aws_ec2

# Local Dependencies:
from .prompt_app import PromptEngineeringApp
from .smstudio import WorkshopSageMakerEnvironment


class LLMEValWKshpStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        deploy_prompt_app: bool = True,
        deploy_sagemaker_domain: bool = True,
        sagemaker_code_checkout: Optional[str] = None,
        sagemaker_code_repo: Optional[str] = None,
    ) -> None:
        super().__init__(scope, construct_id)

        if deploy_sagemaker_domain:
            # Shared VPC:
            vpc = aws_ec2.Vpc(self, "Vpc")

            # Deploy SageMaker Studio environment:
            sagemaker_env = WorkshopSageMakerEnvironment(
                self,
                "SageMakerEnvironment",
                vpc=vpc,
                code_checkout=sagemaker_code_checkout,
                code_repo=sagemaker_code_repo,
                instance_type="ml.t3.large",
            )

        if deploy_prompt_app:
            cognito_username_param = CfnParameter(
                self,
                "PromptAppUsername",
                default="workshop",
                type="String",
            )
            cognito_password_param = CfnParameter(
                self,
                "PromptAppPassword",
                default="Time2Evalu8!",
                no_echo=True,
                type="String",
            )
            prompt_app = PromptEngineeringApp(
                self,
                "PromptEngineeringApp",
                cognito_demo_username=cognito_username_param.value_as_string,
                cognito_demo_password=cognito_password_param.value_as_string,
            )
            domain_name_output = CfnOutput(
                self,
                "AppDomainName",
                value=prompt_app.domain_name,
            )
            cognito_username_output = CfnOutput(
                self,
                "AppDemoUsername",
                value=prompt_app.demo_cognito_user.username,
            )
            # TODO: In a production environment you probably won't want to publish your pw:
            cognito_password_output = CfnOutput(
                self,
                "AppDemoPassword",
                value=prompt_app.demo_cognito_user.password,
            )
            data_bucket_output = CfnOutput(
                self,
                "DataBucket",
                value=prompt_app.data_bucket.bucket_name,
            )

        if deploy_prompt_app and deploy_sagemaker_domain:
            prompt_app.data_bucket.grant_read_write(sagemaker_env.execution_role)
