
import time
import uuid
import logging

from fastapi import Request

logger = logging.getLogger(__name__)

async def request_id_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex
    
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