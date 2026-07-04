import numpy as np

from surrogate_model import SurrogateConfig, hash_instruction, train_surrogate


def test_hash_instruction_deterministic_and_normalized():
    v1 = hash_instruction("open the top drawer")
    v2 = hash_instruction("open the top drawer")
    np.testing.assert_allclose(v1, v2)
    assert np.linalg.norm(v1) == 0.0 or abs(np.linalg.norm(v1) - 1.0) < 1e-6


def test_hash_instruction_differs_for_different_text():
    v1 = hash_instruction("open the top drawer")
    v2 = hash_instruction("press the red button")
    assert not np.allclose(v1, v2)


def test_surrogate_fits_linear_relationship():
    rng = np.random.default_rng(0)
    n, state_dim, action_dim = 512, 6, 4
    states = rng.normal(size=(n, state_dim))
    W = rng.normal(size=(state_dim, action_dim)) * 0.5
    actions = states @ W  # simple linear ground truth, no noise

    cfg = SurrogateConfig(hidden_dim=64, epochs=400, learning_rate=0.02, seed=0)
    model = train_surrogate(states, actions, instructions=None, config=cfg)

    pred = model.predict(states)
    mse = float(np.mean((pred - actions) ** 2))
    assert mse < 0.05  # should fit a noiseless linear map closely


def test_surrogate_predict_shape():
    rng = np.random.default_rng(1)
    states = rng.normal(size=(20, 9))
    actions = rng.normal(size=(20, 7))
    cfg = SurrogateConfig(epochs=5)
    model = train_surrogate(states, actions, config=cfg)
    out = model.predict(rng.normal(size=(3, 9)))
    assert out.shape == (3, 7)


def test_surrogate_with_instructions():
    rng = np.random.default_rng(2)
    states = rng.normal(size=(30, 4))
    actions = rng.normal(size=(30, 2))
    instructions = ["pick up the block"] * 15 + ["open the drawer"] * 15
    cfg = SurrogateConfig(epochs=10)
    model = train_surrogate(states, actions, instructions=instructions, config=cfg)
    out = model.predict(states[:2], instructions=instructions[:2])
    assert out.shape == (2, 2)
