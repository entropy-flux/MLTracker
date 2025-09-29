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
    def epoch(self) -> int:
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
        Modules specific to this iteration that may be used for training, evaluation,
        logging, or other auxiliary tasks. These modules are separate from the
        model’s core modules.

        Returns:
            Modules: The collection of modules associated with this iteration.
        """  

class Iterations(ABC):
    """
    A collection of iterations within the lifecycle of a model’s training or evaluation.
    Provides an interface for creating, retrieving, and listing iterations.
    """

    @abstractmethod
    def create(self, epoch: int) -> Iteration:
        """
        Creates a record of an iteration in the database, retrieving
        an instance of the entity representing it.

        Args:
            epoch (int): The epoch number of the iteration to create.

        Returns:
            Iteration: The entity representing the created iteration.
        """

    @abstractmethod
    def read(self, epoch: int) -> Optional[Iteration]:
        """
        Retrieves an iteration for a given epoch number if any exists.

        Args:
            epoch (int): The epoch number of the iteration to retrieve.

        Returns:
            Optional[Iteration]: The iteration found if any, otherwise None.
        """

    @abstractmethod
    def list(self) -> list[Iteration]:
        """
        Lists all iterations stored in the collection.

        Returns:
            list[Iteration]: The complete set of iterations.
        """
