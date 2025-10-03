from uuid import UUID, uuid4
from typing import Optional, Any, List
from typing import override
from dataclasses import dataclass
from tinydb import TinyDB, where
from mltracker.ports.iterations import Iterations as Collection
from mltracker.adapters.modules import Modules

@dataclass
class Iteration:
    id: UUID
    epoch: int
    modules: Modules

class Iterations(Collection): 

    def __init__(self, db: TinyDB, path: str):
        self.db = db
        self.path  = path
        self.table = self.db.table(path)

    def build(self, id: UUID | str, epoch: int) -> Iteration:
        return Iteration(
            id=id if isinstance(id, UUID) else UUID(id),
            epoch=epoch, 
            modules = Modules(self.db, f"/iterations/{str(id)}/modules"),
        )

    @override
    def create(self, epoch: int) -> Iteration:
        id = uuid4()
        self.table.insert({
            "id": str(id), 
            "epoch": epoch, 
        })
        return self.build(id, epoch)

    @override
    def get(self, epoch: int) -> Optional[Iteration]:
        result = self.table.get(where("epoch") == epoch)
        if result:
            return self.build(result["id"], result["epoch"])
        return None

    @override
    def list(self) -> list[Iteration]:
        records = self.table.all()
        return [ self.build(
            id=record["id"], 
            epoch=record["epoch"], 
        ) for record in records ]

    @override
    def remove(self, iteration: Iteration): 
        iteration.modules.clear()
        self.table.remove(where("id") == str(iteration.id)) 

    @override
    def clear(self):
        for item in self.table.all():
            iteration = self.build(item["id"], item["epoch"])
            iteration.modules.clear()
        self.db.drop_table(self.path)