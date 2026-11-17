"""Models for predicting MPRA readouts from sequence.

- `cnn`: residual CNN (`ResCNN`) trained from scratch.
- `alphagenome`: pretrained AlphaGenome sequence encoder. It needs the `alphagenome_pytorch`
  package, so it is not imported here; import it as `MPRACollection.model.alphagenome`.
"""

from .cnn import ResBlock, ResCNN

__all__ = ["ResBlock", "ResCNN", "MODELS", "build_model"]

# models that build_model() can create, by the name used in run configuration files
MODELS = {
    "ResCNN": ResCNN,
}


def build_model(name: str, seq_len: int, n_out: int, **kwargs):
    """Create the model `name` from MODELS for sequences of length seq_len and n_out
    readouts; kwargs are passed to the model's constructor."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'; available: {sorted(MODELS)}")
    return MODELS[name](seq_len=seq_len, n_out=n_out, **kwargs)
