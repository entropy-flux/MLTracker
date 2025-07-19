from typing import Literal

from mltracker.ports.experiments import Experiments, Experiment 
from mltracker.ports.models import Models, Model 
from mltracker.adapters.default.experiments import Experiments as DefaultExperimentsCollection
from mltracker.adapters.tinydb.experiments import Experiments as TinyDBExperrimentsCollection

def getallexperiments(*, backend: Literal['default', 'tinydb'] = 'default') -> Experiments: 
    if backend == 'default':
        return DefaultExperimentsCollection()
    elif backend == 'tinydb':
        import os
        from tinydb import TinyDB        
        os.makedirs("data", exist_ok=True) 
        database = TinyDB('data/db.json')
        return TinyDBExperrimentsCollection(database)

def getexperiment(name: str, *, backend: Literal['default', 'tinydb'] = 'default') -> Experiment:
    collection = getallexperiments(backend=backend)
    experiment = collection.read(name)
    if not experiment:
        experiment = collection.create(name)
    return experiment

def getallmodels(experiment: str,*, backend: Literal['default', 'tinydb'] = 'default') -> Models:
    owner = getexperiment(experiment, backend=backend)
    return owner.models