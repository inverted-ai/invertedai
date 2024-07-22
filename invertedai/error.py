import invertedai as iai
"""
Inspired by OpenAI python-API error handling
"""


class InvertedAIError(Exception):
    """
    Base exception class for Inverted AI Python-API error handling.
    """

    def __init__(
        self,
        message=None,
        http_body=None,
        http_status=None,
        json_body=None,
        headers=None,
        code=None,
    ):
        super(InvertedAIError, self).__init__(message)

        if http_body and hasattr(http_body, "decode"):
            try:
                http_body = http_body.decode("utf-8")
            except BaseException:
                http_body = (
                    "<Could not decode body as utf-8. "
                    "Please report to info@inverted.ai>"
                )

        self._message = message
        self.http_body = http_body
        self.http_status = http_status
        self.json_body = json_body
        self.headers = headers or {}
        self.code = code

    def __str__(self):
        msg = self._message or "<empty message>"
        return msg

    @property
    def user_message(self):
        return self._message

    def __repr__(self):
        return "%s(message=%r, http_status=%r, request_id=%r)" % (
            self.__class__.__name__,
            self._message,
            self.http_status,
        )


class APIError(InvertedAIError):
    """
    Errors which are caused by incorrect API call.
    """
    pass


class TryAgain(InvertedAIError):
    """
    Errors that may be resolved upon resending the request.
    """
    pass


class InvalidAPIKeyError(InvertedAIError):
    """
    Missing API Key or invalid API Key type.
    """
    pass


class APIConnectionError(InvertedAIError):
    """
    API-gateway error due to invalid API Key.
    """

    def __init__(
        self,
        message,
        http_body=None,
        http_status=None,
        json_body=None,
        headers=None,
        code=None,
        should_retry=False,
    ):
        super(APIConnectionError, self).__init__(
            message, http_body, http_status, json_body, headers, code
        )
        self.should_retry = should_retry


class InvalidRequestError(InvertedAIError):
    """
    Bad Request error (for example, malformed request syntax,
    invalid request message framing, or deceptive request routing).
    """

    def __init__(
        self,
        message,
        param,
        code=None,
        http_body=None,
        http_status=None,
        json_body=None,
        headers=None,
    ):
        super(InvalidRequestError, self).__init__(
            message, http_body, http_status, json_body, headers, code
        )
        self.param = param

    def __repr__(self):
        return "%s(message=%r, param=%r, code=%r, http_status=%r, " "request_id=%r)" % (
            self.__class__.__name__,
            self._message,
            self.param,
            self.code,
            self.http_status,
        )

    def __reduce__(self):
        return type(self), (
            self._message,
            self.param,
            self.code,
            self.http_body,
            self.http_status,
            self.json_body,
            self.headers,
        )


class ResourceNotFoundError(InvertedAIError):
    pass


class AuthenticationError(InvertedAIError):
    """
    API-gateway error due to invalid API Key.
    """
    pass


class PermissionError(InvertedAIError):
    pass


class RateLimitError(InvertedAIError):
    pass


class ServiceUnavailableError(InvertedAIError):
    pass


class InvalidAPIType(InvertedAIError):
    pass

class RequestTooLarge(InvertedAIError):
    pass

class RequestTimeoutError(InvertedAIError):
    pass

class ServerTimeoutError(InvertedAIError):
    pass


class SignatureVerificationError(InvertedAIError):
    def __init__(self, message, sig_header, http_body=None):
        super(SignatureVerificationError, self).__init__(message, http_body)
        self.sig_header = sig_header

    def __reduce__(self):
        return type(self), (
            self._message,
            self.sig_header,
            self.http_body,
        )


class InvalidInput(InvertedAIError):
    """
    Invalid Python API input.
    """
    pass


class InvalidInputType(InvalidInput):
    """
    Invalid input type for Python API.
    """
    pass
