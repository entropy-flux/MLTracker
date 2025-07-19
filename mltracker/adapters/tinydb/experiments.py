from uuid import uuid4, UUID
from typing import override
from typing import Optional 
from tinydb import TinyDB, where
 
from mltracker.ports.experiments import Experiment
from mltracker.ports.experiments import Experiments as Repository  



from uuid import UUID, uuid4
from attrs import define, field
from typing import Optional
from mltracker.exceptions import Conflict
from mltracker.adapters.default.models import Models

@define
class Experiment:
    id: UUID
    name: str
    models: Models

    def __eq__(self, __value: object):
        if not isinstance(__value, self.__class__):
            return False
        return self.id == __value.id
        
    def __hash__(self):
        return hash(self.id)


class Experiments:
    def __init__(self):
        self.collection = set[Experiment]()

    def create(self, name: str) -> Experiment:
        if any(experiment.name == name for experiment in self.collection):
            raise Conflict(f"Experiment '{name}' already exists")

        experiment = Experiment(
            id=uuid4(),
            name=name,
            models=Models()
        )        

        self.collection.add(experiment)
        return experiment
    

    def read(self, name: str) -> Optional[Experiment]: 
        return next(
            (experiment for experiment in self.collection if experiment.name == name),
            None
        )




class Experiments:
    def __init__(self, database: TinyDB):
        self.database = database
        self.table = self.database.table('experiments')    
        
    
    @override
    def create(self, name: str) -> Experiment:
        if self.table.search(where('name') == name):
            raise ValueError(f"Experiment with name {name} already exists")
        id = uuid4()
        self.table.insert({'id': str(id), 'name': name})
        return Experiment(
            id=id, 
            name=name,  
        )
    

    @override    
    def read(self, name) -> Optional[Experiment]: 
        data = self.table.get(where('name') == name)  
        return Experiment(
            id=UUID(data['id']),
            name=data['name'],  
        ) if data else None
    

    @override
    def update(self, id: UUID, name: str) -> Experiment:
        self.table.update({'name': name}, where('id') == str(id))
        return Experiment(
            id=id, 