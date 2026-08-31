"""Typed failures used by the 05 command-line entry points."""


class ActorError(RuntimeError):
    exit_code = 2


class ConfigError(ActorError):
    exit_code = 3


class DataIntegrityError(ActorError):
    exit_code = 4


class ArtifactCompatibilityError(ActorError):
    exit_code = 5


class OutputExistsError(ActorError):
    exit_code = 6

