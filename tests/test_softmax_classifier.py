import numpy as np

from src.classification.softmax_classifier import SoftmaxClassifier


def test_softmax_classifier_learns_simple_separable_data() -> None:
    rng = np.random.default_rng(7)

    class0 = rng.normal(loc=-1.0, scale=0.2, size=(40, 2)).astype(np.float32)
    class1 = rng.normal(loc=1.0, scale=0.2, size=(40, 2)).astype(np.float32)

    x = np.vstack([class0, class1]).astype(np.float32)
    y = np.concatenate([
        np.zeros(40, dtype=np.int64),
        np.ones(40, dtype=np.int64),
    ])

    classifier = SoftmaxClassifier(input_dim=2, num_classes=2, seed=7)
    classifier.fit(
        x,
        y,
        epochs=80,
        batch_size=16,
        learning_rate=0.2,
        weight_decay=0.0,
        seed=7,
        verbose=False,
    )

    accuracy = classifier.score(x, y)
    assert accuracy > 0.98
