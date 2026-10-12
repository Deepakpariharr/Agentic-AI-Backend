from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class ResponseModel(BaseModel):
    """Generic response model for all API endpoints"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "examples": [
                {
                    "statusCode": 200,
                    "message": "Success",
                    "data": {
                        "response": "Hello, how can I help you?",
                        "user_id": "user123"
                    }
                },
                {
                    "statusCode": 400,
                    "message": "Validation Error",
                    "data": None
                },
                {
                    "statusCode": 500,
                    "message": "Internal Server Error",
                    "data": None
                }
            ]
        }
    )

    statusCode: int
    message: str
    data: Optional[Any] = None


# Helper functions to create standardized responses
def success_response(data: Any = None, message: str = "Success") -> ResponseModel:
    """Create a successful response"""
    return ResponseModel(
        statusCode=200,
        message=message,
        data=data
    )


def error_response(message: str, status_code: int = 500, data: Any = None) -> ResponseModel:
    """Create an error response"""
    return ResponseModel(
        statusCode=status_code,
        message=message,
        data=data
    )


def validation_error_response(message: str = "Validation Error", data: Any = None) -> ResponseModel:
    """Create a validation error response"""
    return ResponseModel(
        statusCode=400,
        message=message,
        data=data
    )


def unauthorized_response(message: str = "Unauthorized", data: Any = None) -> ResponseModel:
    """Create an unauthorized response"""
    return ResponseModel(
        statusCode=401,
        message=message,
        data=data
    )


def not_found_response(message: str = "Not Found", data: Any = None) -> ResponseModel:
    """Create a not found response"""
    return ResponseModel(
        statusCode=404,
        message=message,
        data=data
    )


def server_error_response(message: str = "Internal Server Error", data: Any = None) -> ResponseModel:
    """Create a server error response"""
    return ResponseModel(
        statusCode=500,
        message=message,
        data=data
    )