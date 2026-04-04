class AuthenticationError(Exception):
    pass


class AuthorizationError(Exception):
    pass


class RateLimitError(Exception):
    pass


class InvalidContentError(Exception):
    pass


class NotFoundError(Exception):
    pass

