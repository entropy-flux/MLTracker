from typing import Optional, List
from tinydb import TinyDB, where
from attrs import define

from mltracker.ports.iterations import Iterations as Collection
from mltracker.adapters.tinydb.modules import Modules

@define
class Iteration:
    model_hash: str
    epoch: int
    modules: Modules
    table: any

class Iterations(Collection):
    def __init__(self, database: TinyDB, experiment_name: str, model_name: str):
        self.database = database
        self.experiment_name = experiment_name
        self.root_name = f"{model_name}/iterations"
        self.table = self.database.table(f"{self.experiment_name}/{self.root_name}")

    def create(self, epoch: int) -> Iteration:
        if self.table.search(where('epoch') == epoch):
            raise ValueError(f"Iteration for epoch {epoch} already exists")
        self.table.insert({'epoch': epoch})
        return Iteration(
            model_hash=self.root_name,
            epoch=epoch,
            modules=Modules(self.database, self.experiment_name, self.root_name),
            table=self.table
        )

    def read(self, epoch: int) -> Optional[Iteration]:
        data = self.table.get(where('epoch') == epoch)
        if not data:
            return None
        return Iteration(
            model_hash=self.root_name,
            epoch=data['epoch'],
            modules=Modules(self.database, self.experiment_name, self.root_name),
            table=self.table
        )

    def list(self) -> List[Iteration]:
        iterations = []
        for data in self.table.all():
            iterations.append(Iteration(
                model_hash=self.root_name,
                epoch=data['epoch'],
                modules=Modules(self.database, self.experiment_name, self.root_name),
                table=self.table
            ))
        return iterations
 