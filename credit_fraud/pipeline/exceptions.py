"""Custom exceptions for the pipeline module."""


class InvalidAlgorithmFramework(Exception):
    """Exception raised when an invalid algorithm framework is used.

    This exception is raised when an unsupported or invalid algorithm framework
    is used in the pipeline for training a model.
    """

    pass


class InvalidProcessingFramework(Exception):
    """Exception raised when an invalid processing framework is used.

    This exception is raised when an unsupported or invalid processing framework
    is used in the pipeline for preprocessing data.
    """

    pass
