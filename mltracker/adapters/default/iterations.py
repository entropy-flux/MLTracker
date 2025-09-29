from uuid import UUID, uuid4
from attrs import define
from typing import Optional, List
 
from mltracker.adapters.default.modules import Modules 

@define
class Iteration:
    epoch: int
    modules: Modules 


class Iterations:
    def __init__(self):
        self._iterations: List[Iteration] = []

    def create(self, epoch: int) -> Iteration:
        if any(it.epoch == epoch for it in self._iterations):
            raise ValueError(f"Iteration for epoch {epoch} already exists")
        iteration = Iteration(epoch=epoch, modules=Modules())
        self._iterations.append(iteration)
        return iteration

    def read(self, epoch: int) -> Optional[Iteration]:
        return next((it for it in self._iterations if it.epoch == epoch), None)

    def list(self) -> List[Iteration]:
        return list(self._iterations) 