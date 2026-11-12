from __future__ import annotations

from typing import Any


class GatewayError(Exception):
    status_code = 502
    error_type = "upstream_error"
    code = "upstream_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code

    def as_openai(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": None,
                "code": self.code,
            }
        }


class AuthenticationError(GatewayError):
    status_code = 401
    error_type = "authentication_error"
    code = "invalid_api_key"


class NoAvailableAccountError(GatewayError):
    status_code = 503
    error_type = "service_unavailable"
    code = "no_available_account"


class UpstreamAuthError(GatewayError):
    status_code = 401
    error_type = "upstream_authentication_error"
    code = "upstream_auth_expired"


class UpstreamRateLimitError(GatewayError):
    status_code = 429
    error_type = "rate_limit_error"
    code = "upstream_rate_limited"


class UpstreamChallengeError(GatewayError):
    status_code = 403
    error_type = "upstream_challenge"
    code = "upstream_challenge"


class UpstreamError(GatewayError):
    pass
