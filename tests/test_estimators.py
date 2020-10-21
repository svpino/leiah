from leiah.estimators import Estimator


def test_estimator_get_training_job_name():
    estimator = Estimator(model="hello", experiment="world", hyperparameters=dict())
    assert estimator.get_training_job_name() == "hello-world"
