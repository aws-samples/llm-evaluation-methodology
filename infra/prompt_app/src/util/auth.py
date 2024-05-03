# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Main entry point for example prompt template engineering app based on Streamlit
"""
# Python Built-Ins:
import json

# External Dependencies:
import boto3
from streamlit_cognito_auth import CognitoAuthenticator


class CognitoAuth:
    @staticmethod
    def get_authenticator(secret_id: str) -> CognitoAuthenticator:
        """Get Cognito parameters from Secrets Manager and return a CognitoAuthenticator object."""
        # Get Cognito parameters from Secrets Manager
        secretsmanager_client = boto3.client("secretsmanager")
        response = secretsmanager_client.get_secret_value(
            SecretId=secret_id,
        )
        secret_string = json.loads(response["SecretString"])
        pool_id = secret_string["pool_id"]
        app_client_id = secret_string["app_client_id"]
        app_client_secret = secret_string["app_client_secret"]

        # Initialise CognitoAuthenticator
        authenticator = CognitoAuthenticator(
            pool_id=pool_id,
            app_client_id=app_client_id,
            app_client_secret=app_client_secret,
        )

        return authenticator
