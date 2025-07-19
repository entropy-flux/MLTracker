from mltracker.ports.experiments import Experiments, Experiment 
from mltracker.ports.models import Models, Model 
from mltracker.adapters.default.experiments import Experiments as DefaultExperimentsCollection

def getallexperiments() -> Experiments: 
    return DefaultExperimentsCollection()

def getexperiment(name: str) -> Experiment:
    collection = getallexperiments()
    experiment = collection.read(name)
    if not experiment:
        experiment = collection.create(name)
    return experiment

def getallmodels(experiment: str) -> Models:
    owner = getexperiment(experiment)
    return owner.models