from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def custom_key_func(request: Request):
    state = getattr(request, "state", None)
    if state is not None:
        user = getattr(state, "user", None)
        if isinstance(user, dict):
            uid = user.get("uid") or user.get("user_id")
            if uid:
                return uid
        elif user:
            return user

        uid = getattr(state, "uid", None)
        if uid:
            return uid

    return get_remote_address(request)

limiter = Limiter(key_func=custom_key_func)
