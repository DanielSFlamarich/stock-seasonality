import time

import boto3
from date_utils import print_timestamp

"""
AWS related functions and tasks
"""


def run_crawler(glue_crawler: str, region_name, refresh_seconds):
    """
    Runs AWS Glue Crawler to update Glue Catalog,
    so data in new partitions is visible in Athena / Redshift Spectrum.

    Args:
        glue_crawler (str): Crawler Name
        region_name (str): AWS Region
        refresh_seconds (int): Sleep interval while checking crawler state
    """
    glue = boto3.client("glue", region_name=region_name)

    if get_crawler(glue, glue_crawler)["Crawler"]["State"] not in (
        "STARTING",
        "RUNNING",
        "STOPPING",
    ):
        response = glue.start_crawler(Name=glue_crawler)
        print(f"{print_timestamp()}:{response}")

    while get_crawler(glue, glue_crawler)["Crawler"]["State"] in (
        "STARTING",
        "RUNNING",
    ):
        state = get_crawler(glue, glue_crawler)["Crawler"]["State"]
        print(f"{print_timestamp()}:Crawler {state}")
        time.sleep(refresh_seconds)


def get_crawler(glue, glue_crawler):
    response = glue.get_crawler(Name=glue_crawler)
    return response
