"Defines the abstract base class for pipeline steps."

from abc import ABC, abstractmethod


class Step(ABC):
    """
    Abstract base class for pipeline steps.
    """

    @abstractmethod
    def __init__(self):
        """
        Initializes the Step object.
        """
        pass

    @abstractmethod
    def build(self):
        """
        Builds the step.
        """
        pass
