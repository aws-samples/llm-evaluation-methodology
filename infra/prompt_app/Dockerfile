# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Note:
# - Emulating --platform=linux/amd64 *does not work on Apple silicon with Ray*, so we rely on the
#   build request to set the target platform (in case you want to test locally on a Mac) rather
#   than overriding it here. For more details see:
#     - https://github.com/ray-project/ray/issues/43069
#     - https://github.com/aws/fmeval/issues/183
FROM public.ecr.aws/docker/library/python:3.10

ARG AWS_DEFAULT_REGION=us-east-1

ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}

WORKDIR /usr/src/app

# Copying only the reqs before install (not the rest of source) helps re-use the pip layer when
# non-requirements changes are made to src/:
COPY src/requirements.txt ./requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ ./

EXPOSE 8501

# (Note Amazon ECS does check this parameter, so we need to define it in CDK too)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "LLM_Evaluation.py", "--server.port=8501", "--server.address=0.0.0.0"]
