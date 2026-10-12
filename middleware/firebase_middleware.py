import json
import logging

from fastapi import Request, Response
from firebase_admin import auth, credentials, initialize_app

from config import DISABLE_AUTH, FIREBASE_ADMIN_KEY
from model.response_models import unauthorized_response

logger = logging.getLogger(__name__)


def _extract_uid_from_state(request: Request):
    state = getattr(request, "state", None)
    if state is None:
        return None

    user = getattr(state, "user", None)
    if isinstance(user, dict):
        return user.get("uid") or user.get("user_id")
    if isinstance(user, str) and user:
        return user

    uid = getattr(state, "uid", None)
    return uid if isinstance(uid, str) and uid else None


def _load_credentials() -> credentials.Certificate:
    """Load the Firebase service account credentials.

    On Lambda the whole service-account JSON comes from SSM (config pulls it at
    cold start); locally it falls back to the file on disk as before.
    """
    if FIREBASE_ADMIN_KEY:
        return credentials.Certificate(json.loads(FIREBASE_ADMIN_KEY))
    return credentials.Certificate("Secret/firebaseAdminKey.json")


if not DISABLE_AUTH:
    cred = _load_credentials()
    try:
        initialize_app(cred)
    except ValueError:
        pass


async def firebase_auth_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    # Health and API docs are public; every other route needs a valid Firebase
    # token. Keep this list in sync with the `Authorizer: NONE` routes in
    # template.yaml, where API Gateway enforces the same rule before Lambda.
    if request.url.path in ["/", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        error_response_obj = unauthorized_response(
            message="Missing or invalid Authorization header"
        )
        return Response(
            content=error_response_obj.model_dump_json(),
            status_code=401,
            media_type="application/json"
        )

    token = auth_header.split(" ")[1]

    try:
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token.get("uid")
        if not user_id:
            raise ValueError("Firebase token missing uid")
        request.state.user = user_id
        request.state.uid = user_id
    except Exception as e:
        logger.warning("Firebase token verification failed", extra={"error": str(e)})
        error_response_obj = unauthorized_response(
            message="Invalid or expired Firebase token"
        )
        return Response(
            content=error_response_obj.model_dump_json(),
            status_code=401,
            media_type="application/json"
        )
    response = await call_next(request)
    return response
