import os
from pytest import fixture
from mltracker import getallexperiments
from mltracker.ports import Experiments

@fixture(scope='function')
def experiments() -> Experiments: 
    db_path = 'data/db.json'
    if os.path.exists(db_path):
        os.remove(db_path)
    return getallexperiments(backend='tinydb')

def test_experiments(experiments: Experiments):
    experiment = experiments.create('test')
    assert experiment.name == 'test'
    
    experiment = experiments.read('test')
    assert experiment.name == 'test'

def test_models(experiments: Experiments):
    experiment = experiments.create('test')
    model = experiment.models.create('1234', 'mlp-classifier')
    assert model.hash == '1234'
    assert model.name == 'mlp-classifier'
    assert model.epoch == 0
    model.epoch += 1
    model.epoch += 1
    model.epoch += 1
    model = experiment.models.create('12345', 'mlp-classifier-2')
    model = experiment.models.read('1234')
    assert model.epoch == 3
    model = experiment.models.read('12345')
    assert model.epoch == 0 

    
def test_metrics(experiments: Experiments):
    experiment = experiments.create('test')
    model = experiment.models.create('1234', 'mlp-classifier')
    model.metrics.add('loss', 1.55)
    model.metrics.add('accuracy', 0.80)

    list = model.metrics.list()
    assert len(list) == 2


def test_modules(experiments: Experiments):
    experiment = experiments.create('test')
    model = experiment.models.create('1234', 'mlp-classifier')
    model.modules.add('Linear',  {'in_features': 784, 'out_features': 10}) 
    model.modules.add('Linear2', {'in_features': 784, 'out_features': 10}) 
    modules = model.modules.list()
    assert(len(modules) == 2)


def test_iterations(experiments: Experiments):
    experiment = experiments.create('test')
    model = experiment.models.create('1234', 'mlp-classifier')

    iterations = model.iterations.list()
    assert len(iterations) == 0

    iter0 = model.iterations.create(0)
    assert iter0.epoch == 0
    assert len(model.iterations.list()) == 1

    iter1 = model.iterations.create(1)
    assert iter1.epoch == 1
    assert len(model.iterations.list()) == 2

    read_iter0 = model.iterations.read(0)
    assert read_iter0.epoch == 0

    read_iter1 = model.iterations.read(1)
    assert read_iter1.epoch == 1
    
    assert model.iterations.read(99) is None

    import pytest
    with pytest.raises(ValueError):
        model.iterations.create(0)

    iteration = model.iterations.read(1)
    iteration.modules.add('Linear', {'in_features': 784, 'out_features': 10})
    iteration.modules.add('Linear2', {'in_features': 784, 'out_features': 10})
 
    modules = iteration.modules.list()
    assert len(modules) == 2 
