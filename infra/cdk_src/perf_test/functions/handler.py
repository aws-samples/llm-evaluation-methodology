# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK Custom Resource Lambda to copy content into an Amazon S3 Bucket"""
# Python Built-Ins:
import os

# External Dependencies:
import boto3

BLOGS_BUCKET = "aws-blogs-artifacts-public"
SRC_PREFIX = "artifacts/ML-FMBT"
MANIFEST = os.path.join(SRC_PREFIX, "manifest.txt")


def get_manifest():
    s3 = boto3.client("s3")
    manifest = (
        s3.get_object(Bucket=BLOGS_BUCKET, Key=f"{SRC_PREFIX}/manifest.txt")["Body"]
        .read()
        .decode("utf-8")
    )
    files = manifest.splitlines()
    records = [{"Bucket": BLOGS_BUCKET, "Key": f"{SRC_PREFIX}/{f}", "Name": f} for f in files]
    return records


def replicate(bucket):
    s3 = boto3.client("s3")
    records = get_manifest()
    n = len(records)
    print(f"Replicating {n} files to {bucket}")
    for idx, record in enumerate(records):
        name = record["Name"]
        print(f"Copying {idx+1}/{n}: {name}")
        s3.copy_object(
            CopySource={"Bucket": record["Bucket"], "Key": record["Key"]},
            Bucket=bucket,
            Key=record["Name"],
        )


def get_physical_id(bucket_name) -> str:
    return f"fmbench-setup-{bucket_name}"


def on_event(event, context):
    print(event)
    request_type = event["RequestType"]
    if request_type == "Create":
        return on_create(event)
    if request_type == "Update":
        return on_update(event)
    if request_type == "Delete":
        return on_delete(event)
    raise Exception("Invalid request type: %s" % request_type)


def on_create(event):
    props = event["ResourceProperties"]
    print("create new resource with props %s" % props)

    bucket_name = props["BucketName"]
    #
    physical_id = get_physical_id(bucket_name)
    replicate(bucket_name)

    return {"PhysicalResourceId": physical_id}


def on_update(event):
    physical_id = event["PhysicalResourceId"]
    props = event["ResourceProperties"]
    print("update resource %s with props %s" % (physical_id, props))
    on_delete(event)
    on_create(event)


def on_delete(event):
    physical_id = event["PhysicalResourceId"]
    print("delete resource %s" % physical_id)
    # delete all files in S3 bucket
    bucket_name = physical_id[len("fmbench-setup-") :]
    s3 = boto3.client("s3")
    #  delete all objects in the bucket
    # get list_objects paginator
    response = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket_name).build_full_result()
    if "Contents" in response:
        for obj in response["Contents"]:
            s3.delete_object(Bucket=bucket_name, Key=obj["Key"])
