from abc import ABC, abstractmethod
from typing import Protocol
from typing import Optional 
from mltracker.ports.modules import Modules

class Iteration(Protocol):
    """
    An iteration represents a single step or cycle in the training or evaluation
    of a model. It typically corresponds to one pass through the training
    dataset (an epoch) or a defined unit of model progression.

    Iterations can own modules that are associated with the training or evaluation
    process itself, rather than the model's core architecture.
    """

    @property
    def epoch(self) -> Optional[int]:
        """
        The numeric identifier of this iteration in the training or evaluation
        process. Usually corresponds to the number of passes completed over
        the dataset.

        Returns:
            int: The epoch number of this iteration.
        """

    @property
    def modules(self) -> Modules:
        """
        An iteration can own a collection of modules not related to the model
        itself, but to it's training or evaluation process. 

        Returns:
            Modules: A collection of modules.
        """

class Iterations(ABC):
    """
    A collection of iterations within the lifecycle of a model.
    Provides an interface for creating, retrieving, and listing iterations.
    """

    @abstractmethod
    def create(self, epoch: int) -> Iteration:
        """
        Adds a record of an iteration in the database with their respective
        modules and returns it.

        Args:
            epoch (int): The epoch number of the iteration to create. 
        """ 

    @abstractmethod
    def get(self, epoch: int) -> Optional[Iteration]:
        """
        Retrieves a list of iterations for a given epoch number.

        Args:
            epoch (int): The epoch number of the iteration to retrieve.

        Returns:
            list[Iteration]: The iteration list.
        """ 

    @abstractmethod
    def list(self) -> list[Iteration]:
        """
        Lists all iterations stored in the collection.

        Returns:
            list[Iteration]: The complete set of iterations.
        """

    @abstractmethod
    def remove(self, iteration: Iteration):
        """
        Removes an iteration from the collection.

        Args:
            iteration (Iteration): The iteration to remove.
        """

    @abstractmethod
    def clear(self):
        """
        Removes all iterations from the collection.
        """