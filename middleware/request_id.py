
import time
import uuid

from aws_lambda_powertools import Logger
from fastapi import Request

logger = Logger(service="mcpbe")

async def request_id_middleware(request: Request, call_next):
    request_id = logger.get_correlation_id() if "aws.context" in request.scope else None
    request_id = request_id or uuid.uuid4().hex
    logger.set_correlation_id(request_id)
    
    start=time.perf_counter()
    try:
        response=await call_next(request)
    except Exception:
        logger.exception("request failed",extra={"method":request.method,"path":request.url.path})
        raise
    
    response.headers["X-Request-ID"]=request_id
    logger.info("request completed", extra={
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": round((time.perf_counter() - start) * 1000, 2),
    })
    return response