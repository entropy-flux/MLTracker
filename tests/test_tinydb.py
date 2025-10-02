from uuid import UUID
import pytest
import os
from tinydb import TinyDB
from tempfile import NamedTemporaryFile
import pytest
from uuid import UUID
from mltracker.adapters.experiments import Experiments

@pytest.fixture
def db(): 
    file = NamedTemporaryFile(delete=False)
    file.close() 
    
    db = TinyDB(file.name)
    yield db   
    db.close()
    if os.path.exists(file.name):
        os.remove(file.name)


def test_experiments(db: TinyDB):
    repository = Experiments(db)

    created_first  = repository.create("experiment1")
    created_second = repository.create("experiment2")
 
    assert isinstance(created_first.id, UUID)
    assert created_first.name == "experiment1"
    assert isinstance(created_second.id, UUID)
    assert created_second.name == "experiment2"
 
    readed_first = repository.read("experiment1")
    assert readed_first is not None
    assert readed_first.id == created_first.id
    assert readed_first.name == created_first.name
 
    assert repository.read("nonexistent") is None 

    all_experiments = repository.list()
    assert len(all_experiments) == 2
    ids = [experiment.id for experiment in all_experiments]
    assert created_first.id in ids and created_second.id in ids

    repository.update(created_first.id, "experiment1_updated")
    updated_first = repository.read("experiment1_updated")
    assert updated_first is not None
    assert updated_first.id == created_first.id
    assert updated_first.name == "experiment1_updated"
    assert repository.read("experiment1") is None

    repository.delete(created_second.id)
    assert repository.read("experiment2") is None
    remaining_experiments = repository.list()
    assert len(remaining_experiments) == 1
    assert remaining_experiments[0].id == created_first.id


def test_experiment_models(db: TinyDB): 
    repository = Experiments(db)
    experiment = repository.create("experiment_with_models")
 
    assert experiment.models.list() == []
 
    model1 = experiment.models.create(hash="hash1", name="model1")
    model2 = experiment.models.create(hash="hash2", name="model2")
 
    read_model1 = experiment.models.read("hash1")
    read_model2 = experiment.models.read("hash2")
    assert read_model1 is not None
    assert read_model1.hash == "hash1"
    assert read_model1.name == "model1"
    assert read_model2.hash == "hash2"
    assert read_model2.name == "model2"
 
    experiment.models.update("hash1", "model1_updated")
    updated_model1 = experiment.models.read("hash1")
    assert updated_model1.name == "model1_updated"
 
    experiment.models.delete(model2.id)
    assert experiment.models.read("hash2") is None
 
    remaining_models = experiment.models.list()
    assert len(remaining_models) == 1
    assert remaining_models[0].id == model1.id


def test_experiment_metrics(db: TinyDB):
    repository = Experiments(db)
    experiment = repository.create("test_metrics")

    model1 = experiment.models.create(hash="hash1", name="model1")
    model2 = experiment.models.create(hash="hash2", name="model2")
    model3 = experiment.models.create(hash="hash3", name="model3")

    model1.metrics.add(name="accuracy", value=0.80, epoch=1, phase="train")
    model1.metrics.add(name="accuracy", value=0.85, epoch=2, phase="train")
    model1.metrics.add(name="accuracy", value=0.88, epoch=3, phase="val")
    model1.metrics.add(name="accuracy", value=0.90, epoch=4, phase="val")
    model1.metrics.add(name="loss", value=0.40, epoch=1, phase="train")
    model1.metrics.add(name="loss", value=0.30, epoch=2, phase="train")
    model1.metrics.add(name="loss", value=0.25, epoch=3, phase="val")
    model1.metrics.add(name="loss", value=0.20, epoch=4, phase="val")
    model1.metrics.add(name="precision", value=0.70, epoch=2, phase="val")
    model1.metrics.add(name="recall", value=0.65, epoch=2, phase="val")
 
    model2.metrics.add(name="accuracy", value=0.50, epoch=1)
    model2.metrics.add(name="loss", value=1.25, epoch=1)
    model2.metrics.add(name="f1", value=0.55, epoch=1, phase="train")
    model2.metrics.add(name="f1", value=0.60, epoch=2, phase="val")

    all_metrics = model1.metrics.list()
    assert len(all_metrics) == 10
    assert all(isinstance(m.id, UUID) for m in all_metrics)
 
    acc_metrics = model1.metrics.list(name="accuracy")
    assert len(acc_metrics) == 4
    assert [m.value for m in acc_metrics] == [0.80, 0.85, 0.88, 0.90]
    assert [m.epoch for m in acc_metrics] == [1, 2, 3, 4]
    assert [m.phase for m in acc_metrics] == ["train", "train", "val", "val"]
 
    loss_metrics = model1.metrics.list(name="loss")
    assert len(loss_metrics) == 4
    assert [m.value for m in loss_metrics] == [0.40, 0.30, 0.25, 0.20]
    assert [m.epoch for m in loss_metrics] == [1, 2, 3, 4]
    assert [m.phase for m in loss_metrics] == ["train", "train", "val", "val"]
 
    prec_metrics = model1.metrics.list(name="precision")
    recall_metrics = model1.metrics.list(name="recall")
    assert len(prec_metrics) == 1 and prec_metrics[0].value == 0.70
    assert len(recall_metrics) == 1 and recall_metrics[0].value == 0.65
 
    to_remove = next(m for m in acc_metrics if m.value == 0.85)
    model1.metrics.remove(to_remove)
    remaining = model1.metrics.list(name="accuracy")
    assert len(remaining) == 3
    assert all(m.value != 0.85 for m in remaining)
 
    to_remove_loss = next(m for m in loss_metrics if m.value == 0.25)
    model1.metrics.remove(to_remove_loss)
    remaining_losses = model1.metrics.list(name="loss")
    assert len(remaining_losses) == 3
    assert all(m.value != 0.25 for m in remaining_losses)
 
    model1.metrics.clear()
    assert model1.metrics.list() == []
  
    m2_all = model2.metrics.list()
    assert len(m2_all) == 4
    assert [m.name for m in m2_all] == ["accuracy", "loss", "f1", "f1"]
 
    assert len(model3.metrics.list()) == 0
    model3.metrics.add(name="auc", value=0.72, epoch=1)
    model3.metrics.add(name="auc", value=0.75, epoch=2, phase="val")

    auc_metrics = model3.metrics.list(name="auc")
    assert len(auc_metrics) == 2
    assert [m.value for m in auc_metrics] == [0.72, 0.75]
 
    assert model1.metrics.list() == [] 
    assert len(model2.metrics.list()) == 4
    assert len(model3.metrics.list()) == 2


def test_model_epochs(db: TinyDB):
    repository = Experiments(db)
    experiment = repository.create("experiment_with_epochs")
    model = experiment.models.create(hash="model_hash", name="model_name")
    model.epoch += 1
    model.epoch += 1

    model = experiment.models.read(hash="model_hash")
    assert model.epoch == 2


def test_model_modules(db: TinyDB): 
    repository = Experiments(db)
    experiment = repository.create("experiment_with_modules")
    model = experiment.models.create(hash="model_hash", name="model_name")

    assert model.modules.list() == []

    model.modules.add(name="module1", attributes={"type": "conv", "layers": 3})
    model.modules.add(name="module2", attributes={"type": "dense", "units": 128})

    all_modules = model.modules.list()
    assert len(all_modules) == 2

    module1 = next(m for m in all_modules if m.name == "module1")
    module2 = next(m for m in all_modules if m.name == "module2")

    assert module1.attributes == {"type": "conv", "layers": 3}
    assert module2.attributes == {"type": "dense", "units": 128}

    model.modules.remove(module1)
    remaining = model.modules.list()
    assert len(remaining) == 1
    assert remaining[0].id == module2.id

    model.modules.clear()
    assert model.modules.list() == []



def test_model_iterations(db: TinyDB):
    repository = Experiments(db)
    experiment = repository.create("experiment_with_iterations")
    model = experiment.models.create(hash="model_hash", name="model_name")

    assert model.iterations.list() == []

    model.iterations.create(epoch=1)
    model.iterations.create(epoch=2)

    all_iterations = model.iterations.list()
    assert len(all_iterations) == 2

    iter1 = model.iterations.get(1)
    iter2 = model.iterations.get(2)

    assert iter1 is not None
    assert iter1.epoch == 1
    assert iter2 is not None
    assert iter2.epoch == 2

    iter1.modules.add(name="moduleA", attributes={"type": "conv"})
    iter1.modules.add(name="moduleB", attributes={"type": "dense"})

    modules_iter1 = iter1.modules.list()
    assert len(modules_iter1) == 2
    names = [m.name for m in modules_iter1]
    assert "moduleA" in names and "moduleB" in names

    model.iterations.remove(iter2)
    remaining_iterations = model.iterations.list()
    assert len(remaining_iterations) == 1
    assert remaining_iterations[0].epoch == 1

    model.iterations.clear()
    assert model.iterations.list() == []
