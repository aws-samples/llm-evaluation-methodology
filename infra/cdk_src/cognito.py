# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Python CDK construct for adding an Amazon Cognito User

Inspired by the below repo, which only offered JS/TypeScript distribution at the time of writing:
https://github.com/awesome-cdk/cdk-userpool-user/
"""
# Python Built-Ins:
from typing import Dict, List, Optional, Union

# External Dependencies:
from aws_cdk import SecretValue
from aws_cdk.custom_resources import (
    AwsCustomResource,
    AwsCustomResourcePolicy,
    AwsSdkCall,
    PhysicalResourceId,
)
from aws_cdk.aws_cognito import CfnUserPoolUserToGroupAttachment, IUserPool
from constructs import Construct


class UserPoolUser(Construct):
    """CDK Construct for adding a user to an Amazon Cognito User Pool

    This construct will create a user in the Cognito user pool, and optionally add them to a group.

    TODO: Improve security of how the password is passed around - currently goes through plain text
    """

    _username: str
    _password: SecretValue

    def __init__(
        self,
        scope: Construct,
        id: str,
        *,
        user_pool: IUserPool,
        username: str,
        password: SecretValue,
        attributes: List[Dict[Union["Name", "Value"], str]] = [],
        group_name: Optional[str] = None,
    ):
        super().__init__(scope, id)
        self._username = username
        self._password = password

        # Create the user in the pool:
        create_user_cr = AwsCustomResource(
            self,
            "AwsCustomResource-CreateUser",
            on_create=AwsSdkCall(
                action="adminCreateUser",
                parameters={
                    "UserPoolId": user_pool.user_pool_id,
                    "Username": username,
                    "MessageAction": "SUPPRESS",
                    "TemporaryPassword": password,
                    "UserAttributes": attributes,
                },
                physical_resource_id=PhysicalResourceId.of(
                    f"AwsCustomResource-CreateUser-{username}"
                ),
                service="@aws-sdk/client-cognito-identity-provider",
            ),
            on_delete=AwsSdkCall(
                action="adminDeleteUser",
                parameters={
                    "UserPoolId": user_pool.user_pool_id,
                    "Username": username,
                },
                service="@aws-sdk/client-cognito-identity-provider",
            ),
            policy=AwsCustomResourcePolicy.from_sdk_calls(
                resources=AwsCustomResourcePolicy.ANY_RESOURCE
            ),
            install_latest_aws_sdk=True,
        )

        # Force set the password, to avoid inescapable FORCE_PASSWORD_CHANGE status:
        set_password_cr = AwsCustomResource(
            self,
            "AwsCustomResource-ForcePassword",
            on_create=AwsSdkCall(
                action="adminSetUserPassword",
                parameters={
                    "UserPoolId": user_pool.user_pool_id,
                    "Username": username,
                    "Password": password,
                    "Permanent": True,
                },
                physical_resource_id=PhysicalResourceId.of(
                    f"AwsCustomResource-ForcePassword-{username}"
                ),
                service="@aws-sdk/client-cognito-identity-provider",
            ),
            policy=AwsCustomResourcePolicy.from_sdk_calls(
                resources=AwsCustomResourcePolicy.ANY_RESOURCE
            ),
            install_latest_aws_sdk=True,
        )
        set_password_cr.node.add_dependency(create_user_cr)

        # Also add the user to the group, if one was provided:
        if group_name:
            group_attachment = CfnUserPoolUserToGroupAttachment(
                self,
                "UserToGroupAttachment",
                user_pool_id=user_pool.user_pool_id,
                group_name=group_name,
                username=username,
            )
            group_attachment.node.add_dependency(create_user_cr)
            group_attachment.node.add_dependency(set_password_cr)
            group_attachment.node.add_dependency(user_pool)

    @property
    def username(self) -> str:
        return self._username

    @property
    def password(self) -> SecretValue:
        return self._password
