# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK construct for a SageMaker Studio domain for demos and workshops
"""
# Python Built-Ins:
import os
import random
from typing import Optional

# External Dependencies:
from aws_cdk import Duration, RemovalPolicy, SecretValue, Stack
from aws_cdk import aws_cloudfront as cf
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_ecs_patterns
from aws_cdk import aws_iam as iam
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk.aws_cloudfront_origins import HttpOrigin
from aws_cdk.aws_ec2 import Port
from aws_cdk.aws_ecr_assets import DockerImageAsset, Platform
from aws_cdk.aws_elasticloadbalancingv2 import (
    ApplicationLoadBalancer,
    ApplicationProtocol,
    ListenerCondition,
    ListenerAction,
)
from aws_cdk.aws_logs import RetentionDays
from aws_cdk import aws_s3 as s3
from constructs import Construct
from upsert_slr import ServiceLinkedRole

# Local Dependencies:
from .cognito import UserPoolUser


FRONTEND_CODE_PATH = os.path.join(os.path.dirname(__file__), "../prompt_app")


class PromptEngineeringApp(Construct):
    _cognito_user_pool: cognito.UserPool
    _cloudfront_distribution: cf.Distribution
    _data_bucket: s3.Bucket
    _demo_cognito_user: Optional[UserPoolUser]
    _ecs_service: aws_ecs_patterns.ApplicationLoadBalancedFargateService

    def __init__(
        self,
        scope: Construct,
        id: str,
        *,
        cognito_user_pool: Optional[cognito.UserPool] = None,
        cognito_demo_username: Optional[str] = "workshop",
        cognito_demo_password: Optional[str] = None,
        container_cpu_units: int = 16384,  # 16vCPU
        container_mem_mib: int = 32768,  # 32GB
    ) -> None:
        super().__init__(scope, id)
        stack = Stack.of(self)

        # Bucket to store datasets/etc
        self._data_bucket = s3.Bucket(
            self,
            "DataBucket",
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            cors=[
                # CORS permissions are required for the A2I/SMGT human review UIs:
                s3.CorsRule(
                    allowed_headers=["*"],
                    allowed_methods=[
                        s3.HttpMethods.DELETE,
                        s3.HttpMethods.GET,
                        s3.HttpMethods.HEAD,
                        s3.HttpMethods.POST,
                        s3.HttpMethods.PUT,
                    ],
                    allowed_origins=["*"],
                    exposed_headers=["Access-Control-Allow-Origin"],
                ),
            ],
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Cognito user pool
        if not cognito_user_pool:
            cognito_user_pool = cognito.UserPool(
                self,
                "UserPool",
                advanced_security_mode=cognito.AdvancedSecurityMode.ENFORCED,
                removal_policy=RemovalPolicy.DESTROY,
                self_sign_up_enabled=False,
                sign_in_aliases=cognito.SignInAliases(username=True, email=True),
                sign_in_case_sensitive=False,
            )
        self._cognito_user_pool = cognito_user_pool
        user_pool_client = cognito.UserPoolClient(
            self,
            "UserPoolClient",
            user_pool=self._cognito_user_pool,
            generate_secret=True,
        )

        # Store Cognito configuration in a secret to use in Streamlit app
        cognito_secret = secretsmanager.Secret(
            self,
            "ParamCognitoSecret",
            secret_object_value={
                "pool_id": SecretValue.unsafe_plain_text(cognito_user_pool.user_pool_id),
                "app_client_id": SecretValue.unsafe_plain_text(
                    user_pool_client.user_pool_client_id
                ),
                "app_client_secret": user_pool_client.user_pool_client_secret,
            },
        )

        # Demo user for cognito, if required:
        if cognito_demo_username:
            if not cognito_demo_password:
                raise ValueError(
                    "cognito_demo_password must be set if cognito_demo_username is not None"
                )
            self._demo_cognito_user = UserPoolUser(
                self,
                "DemoUser",
                username=cognito_demo_username,
                password=cognito_demo_password,
                user_pool=cognito_user_pool,
            )

        # Docker image build and upload to ECR
        container_image = ecs.ContainerImage.from_docker_image_asset(
            DockerImageAsset(
                self, "DockerImage", directory=FRONTEND_CODE_PATH, platform=Platform.LINUX_AMD64
            )
        )

        # In a fresh account, ECS service role may not exist and creating ecs.Cluster does not seem
        # to automatically create it - therefore use the upsert_slr package:
        ecs_slr = ServiceLinkedRole(
            self,
            "AWSServiceRoleForECS",
            aws_service_name="ecs.amazonaws.com",
            description="Role to enable Amazon ECS to manage your cluster.",
        )

        # create fargate ecs cluster
        cluster = ecs.Cluster(
            self, "Cluster", enable_fargate_capacity_providers=True, container_insights=True
        )
        cluster.node.add_dependency(ecs_slr)

        # create fargate task definition
        task_definition = ecs.FargateTaskDefinition(
            self,
            "TaskDefinition",
            cpu=container_cpu_units,
            memory_limit_mib=container_mem_mib,
        )

        # Add container to task definition
        app_container = task_definition.add_container(
            "Container",
            image=container_image,
            cpu=container_cpu_units,
            environment={
                "AWS_DEFAULT_REGION": stack.region,
                "AWS_REGION": stack.region,
                "COGNITO_SECRET_NAME": cognito_secret.secret_name,
            },
            memory_limit_mib=container_mem_mib,
            logging=ecs.AwsLogDriver(
                stream_prefix="llm-eval-wkshp", log_retention=RetentionDays.ONE_WEEK
            ),
            # TODO: The restart on this is way too slow
            # Try e.g. explicitly Denying the task role Bedrock access in IAM, and trying to run an
            # evaluation job in the UI: It takes several minutes for ECS to detect the container's
            # died and re-deploy it... Better error handling or automatic Streamlit restart within
            # the container definition might be better?
            health_check=ecs.HealthCheck(
                # We also define this in the Dockerfile HEALTHCHECK, but ECS doesn't look there:
                command=["CMD-SHELL", "curl --fail http://localhost:8501/_stcore/health || exit 1"],
                start_period=Duration.minutes(3),  # Initial grace time for start-up
                timeout=Duration.seconds(10),
            ),
        )
        app_container.add_port_mappings(
            ecs.PortMapping(container_port=8501, protocol=ecs.Protocol.TCP)
        )

        self._data_bucket.grant_read_write(task_definition.task_role)
        cognito_secret.grant_read(task_definition.task_role)
        task_definition.add_to_task_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "states:DescribeExecution",
                    "states:GetActivityTask",
                    "states:GetExecutionHistory",
                ],
                resources=["*"],
                sid="GetSFnExecutionStatus",
            ),
        )
        task_definition.add_to_task_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=["*"],
                sid="BedrockAccess",
            ),
        )
        task_definition.add_to_task_role_policy(
            iam.PolicyStatement(
                actions=["sagemaker:InvokeEndpoint"],
                resources=[
                    f"arn:{stack.partition}:sagemaker:{stack.region}:{stack.account}:endpoint/*"
                ],
                sid="InvokeSageMakerEndpoints",
            )
        )
        task_definition.add_to_task_role_policy(
            iam.PolicyStatement(
                actions=["sagemaker:ListEndpoints"],
                resources=["*"],
                sid="ListSageMakerEndpoints",
            )
        )
        task_definition.add_to_task_role_policy(
            iam.PolicyStatement(
                actions=["secretsmanager:GetSecretValue"],
                resources=[
                    f"arn:{stack.partition}:secretsmanager:{stack.region}:{stack.account}:secret:openai_key-??????"
                ],
                sid="GetOpenAIApiKey",
            )
        )

        # ECS Fargate service
        # TODO: TLS encyption between CloudFront and ECS, if you need it
        # Setting up the ECS service to offer HTTPS connections requires a valid certificate, which
        # in turn requires a known domain name. There's too many dependencies for us to provide
        # end-to-end automation of such a setup in this workshop-oriented sample, but as a best
        # practice any production solution based on this architecture should close the loop to
        # enable encryption in transit between CloudFront and the ECS service.
        ecs_service = aws_ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "Service",
            cluster=cluster,
            task_definition=task_definition,
            protocol=ApplicationProtocol.HTTP,
        )
        self._ecs_service = ecs_service
        ecs_service.service.connections.allow_from_any_ipv4(
            port_range=Port.tcp(80), description="Allow HTTP from anywhere"
        )

        # We'll generate a randomized key to ensure the ECS service only accepts connections from
        # CloudFront, and not direct:
        app_auth_key = "".join(random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=15))
        origin = HttpOrigin(
            ecs_service.load_balancer.load_balancer_dns_name,
            protocol_policy=cf.OriginProtocolPolicy.HTTP_ONLY,
            custom_headers={"X-Custom-Header": app_auth_key},
        )

        origin_request_policy = cf.OriginRequestPolicy(
            self,
            "OriginRequestPolicy",
            cookie_behavior=cf.OriginRequestCookieBehavior.all(),
            header_behavior=cf.OriginRequestHeaderBehavior.all(),
            query_string_behavior=cf.OriginRequestQueryStringBehavior.all(),
            # (Explicitly setting `origin_request_policy_name` breaks any stack UPDATE ops that
            # require replacement - so we leave it default)
        )

        self._cloudfront_distribution = cf.Distribution(
            self,
            "Distribution",
            default_behavior=cf.BehaviorOptions(
                origin=origin,
                viewer_protocol_policy=cf.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                allowed_methods=cf.AllowedMethods.ALLOW_ALL,
                cache_policy=cf.CachePolicy.CACHING_DISABLED,
                origin_request_policy=origin_request_policy,
            ),
        )

        ecs_service.listener.add_action(
            "alb_listener_action",
            priority=1,
            # Require the static secret key header configured in the CloudFront distribution:
            conditions=[ListenerCondition.http_header("X-Custom-Header", [app_auth_key])],
            action=ListenerAction.forward(target_groups=[ecs_service.target_group]),
        )

        ecs_service.listener.add_action(
            "default",
            action=ListenerAction.fixed_response(
                status_code=403, content_type="text/plain", message_body="Forbidden"
            ),
        )

    @property
    def cognito_user_pool(self) -> cognito.UserPool:
        return self._cognito_user_pool

    @property
    def data_bucket(self) -> s3.IBucket:
        return self._data_bucket

    @property
    def demo_cognito_user(self) -> Optional[UserPoolUser]:
        return self._demo_cognito_user

    @property
    def domain_name(self) -> str:
        return self._cloudfront_distribution.distribution_domain_name

    @property
    def load_balancer(self) -> ApplicationLoadBalancer:
        return self._ecs_service.load_balancer
