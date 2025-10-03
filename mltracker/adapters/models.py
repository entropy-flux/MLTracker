from uuid import uuid4, UUID
from typing import Optional
from typing import override
from dataclasses import dataclass
from tinydb import TinyDB, where 
from mltracker.ports.models import Models as Collection 
from mltracker.adapters.metrics import Metrics
from mltracker.adapters.modules import Modules
from mltracker.adapters.iterations import Iterations
from mltracker.adapters.epochs import Epochs

@dataclass
class Model:
    id: UUID
    hash: str
    name: Optional[str]
    metrics: Metrics
    modules: Modules
    iterations: Iterations
    epochs: Epochs

    @property
    def epoch(self) -> int:
        return self.epochs.get()
    
    @epoch.setter
    def epoch(self, value: int):
        self.epochs.set(value)

class Models(Collection):

    def __init__(self, db: TinyDB, path: str):
        self.db = db
        self.path = path
        self.table = self.db.table(path)

    def build(self, id: str | UUID, hash: str, name: Optional[str]) -> Model:
        return Model(
            id = id if isinstance(id, UUID) else UUID(id),  
            hash = hash, 
            name = name,
            metrics = Metrics(self.db, f"/models/{str(id)}/metrics"),
            modules = Modules(self.db, f"/models/{str(id)}/modules"),
            iterations = Iterations(self.db, f"/models/{str(id)}/iterations"),
            epochs = Epochs(self.db, f"/models/{str(id)}/epochs")
        )
    
    @override
    def create(self, hash: str, name: Optional[str] = None) -> Model:    
        if self.table.get(where("hash") == hash):
            raise ValueError(f"Experiment with hash '{hash}' already exists")
        
        id = uuid4() 
        self.table.insert({
            "id": str(id),  
            "hash": hash,
            "name": name
        })
        return self.build(id, hash, name)
        
    @override
    def read(self, hash: str) -> Optional[Model]:
        result = self.table.get(where("hash") == hash)
        if result:
            return self.build(result["id"], result["hash"], result["name"])
        return None

    @override
    def update(self, hash: str, name: str): 
        result = self.table.get(where("hash") == hash) 
        if result: 
            self.table.update({"name": name}, doc_ids=[result.doc_id])

    @override
    def delete(self, id: UUID):
        result = self.table.get(where("id") == str(id))
        if result:
            model = self.build(result["id"], result["hash"], result["name"])
            model.metrics.clear() 
            model.modules.clear()
            model.iterations.clear()
            model.epochs.drop()
            self.table.remove(doc_ids=[result.doc_id])

    @override
    def list(self) -> list[Model]:
        models = []
        for item in self.table.all():
            model = self.build(item["id"], item["hash"], item["name"])
            models.append(model)
        return models 
    
    @override
    def clear(self):   
        for item in self.table.all():
            model = self.build(item["id"], item["hash"], item["name"])
            model.metrics.clear() 
            model.modules.clear()
            model.iterations.clear()
            model.epochs.drop()
        self.db.drop_table(self.path)