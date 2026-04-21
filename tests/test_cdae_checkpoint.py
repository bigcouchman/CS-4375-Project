from pathlib import Path

import numpy as np

from src.models import CDAE


def test_cdae_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = CDAE(input_channels=3, latent_channels=8, seed=7)

    # Mutate weights so this is not just default initialization.
    model.encoder_conv.weights += 0.123
    model.output_conv.bias += 0.456

    checkpoint_path = tmp_path / "cdae_checkpoint.npz"
    model.save_checkpoint(checkpoint_path)

    restored_model = CDAE(input_channels=3, latent_channels=8, seed=999)
    restored_model.load_checkpoint(checkpoint_path)

    assert np.allclose(model.encoder_conv.weights, restored_model.encoder_conv.weights)
    assert np.allclose(model.encoder_conv.bias, restored_model.encoder_conv.bias)
    assert np.allclose(model.bottleneck_conv.weights, restored_model.bottleneck_conv.weights)
    assert np.allclose(model.decoder_conv.weights, restored_model.decoder_conv.weights)
    assert np.allclose(model.output_conv.bias, restored_model.output_conv.bias)
