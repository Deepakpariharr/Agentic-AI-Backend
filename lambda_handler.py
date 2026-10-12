"""AWS Lambda entry point.

Wraps the existing FastAPI ASGI app with Mangum so it can run behind
API Gateway (HTTP API) without any changes to the app itself.
"""
from aws_lambda_powertools import Logger
from aws_lambda_powertools.logging import correlation_paths
from mangum import Mangum

from main import app

mangum_handler = Mangum(app)
logger=Logger(service="mcpbe")

@logger.inject_lambda_context(
    correlation_id_path=correlation_paths.API_GATEWAY_HTTP, # API Gateway Request Id
    clear_state=True #reset keys between invocations
)

def handler(event,context):
    return mangum_handler(event,context)