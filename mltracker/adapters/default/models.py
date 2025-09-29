from uuid import UUID
from uuid import uuid4
from attrs import define 
from typing import Optional 
from mltracker.adapters.default.metrics import Metrics
from mltracker.adapters.default.modules import Modules
from mltracker.adapters.default.iterations import Iterations
 
@define
class Model:
    id: UUID
    name: str
    hash: str
    epoch: int
    metrics: Metrics
    modules: Modules
    iterations: Iterations

    def __eq__(self, __value: object):
        if not isinstance(__value, self.__class__):
            return False
        return self.id == __value.id

    def __hash__(self):
        return hash(self.id)


class Models:
    def __init__(self):
        self.collection: set[Model] = set()

    def create(self, hash: str, name: Optional[str] = None) -> Model:
        if any(model.hash == hash for model in self.collection):
            raise ValueError(f"Model with hash '{hash}' already exists")

        model = Model(
            id=uuid4(),
            name=name,
            hash=hash,
            epoch=0,
            metrics=Metrics(),
            modules=Modules(),
            iterations=Iterations()
        )

        self.collection.add(model)
        return model

    def read(self, hash: str) -> Optional[Model]:
        return next((model for model in self.collection if model.hash == hash), None)

    def update(self, hash: str, name: str) -> Optional[Model]:
        model = self.read(hash)
        if model:
            model.name = name
        return model

    def list(self) -> list[Model]:
        return list(self.collection)