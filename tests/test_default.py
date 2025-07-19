from pytest import fixture
from mltracker import getallexperiments
from mltracker.ports import Experiments

@fixture(scope='function')
def experiments() -> Experiments:
    return getallexperiments()

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
    model.modules.add('Linear', {'in_features': 784, 'out_features': 10}) 
    modules = model.modules.list()