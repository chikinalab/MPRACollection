"""Pretrained AlphaGenome sequence encoder.

Model: the convolutional encoder (`alphagenome_pytorch.model.SequenceEncoder`) of the
pretrained AlphaGenome model, loaded from the Hugging Face repository
`gtca/alphagenome_pytorch`. Only the 80 `encoder.*` tensors are read from the weights file;
the transformer, decoder and output heads stored in the same file are skipped.

Input: float tensor of shape (batch, length, 4), one-hot encoded in ACGT order, with N and
other unknown bases as all-zero rows. This differs from the default encoding of
`MPRA_Dataset.to_DataLoader()` (ATGC order, N = 0.25, channels first); pass
`collate_fn=make_collate()` to get batches in the encoder's format.

Output: `encoder(x)` returns `(trunk, intermediates)`. `trunk` has shape
(batch, 1536, ceil(length / 128)): 1536 channels and one position per 128 bp (the last
position also covers the zero padding the pooling layers add at the end). `intermediates`
is a dict of the outputs at bin sizes 1, 2, 4, ..., 64 bp, used by the AlphaGenome decoder.

Needs the `alphagenome_pytorch` and `safetensors` packages (the `alphagenome-ft` environment).
"""

from huggingface_hub import hf_hub_download
from safetensors import safe_open

from alphagenome_pytorch.model import SequenceEncoder

from ..data import OneHotCollate

REPO_ID = "gtca/alphagenome_pytorch"
WEIGHTS_FILE = "model_all_folds.safetensors"  # or model_fold_{0..3}.safetensors
ENCODER_PREFIX = "encoder."
EMBED_DIM = 1536  # channels of the encoder output
BIN_SIZE = 128  # bp per position of the encoder output

# one-hot encoding the encoder was trained with, as OneHotCollate arguments
ONEHOT_ALPHABET = "ACGT"
ONEHOT_VALUE_UNKNOWN = 0.0


def load_pretrained_encoder(
    weights_file: str = WEIGHTS_FILE,
    repo_id: str = REPO_ID,
    device="cpu",
    eval_mode: bool = True,
) -> SequenceEncoder:
    """Create a SequenceEncoder with the pretrained AlphaGenome encoder weights.

    The weights file is downloaded to, or read from, the Hugging Face cache. With
    HF_HUB_OFFLINE=1 (e.g. on compute nodes) it must already be in the cache.

    Args:
        weights_file: file in the repository; model_all_folds.safetensors or
            model_fold_{0..3}.safetensors
        repo_id: Hugging Face repository with the weights
        device: device to move the encoder to
        eval_mode: call eval(), so that BatchNorm uses the stored running statistics.
            Set False to get the encoder in training mode.
    """
    path = hf_hub_download(repo_id, weights_file)
    with safe_open(path, framework="pt") as f:
        state = {k[len(ENCODER_PREFIX):]: f.get_tensor(k) for k in f.keys() if k.startswith(ENCODER_PREFIX)}
    encoder = SequenceEncoder()
    # strict=True raises an error on any missing or unexpected key; the bf16 weights are cast to the fp32 parameters
    encoder.load_state_dict(state, strict=True)
    encoder = encoder.to(device)
    return encoder.eval() if eval_mode else encoder.train()


def make_collate(len_max: int = None) -> OneHotCollate:
    """Collate function giving X of shape (batch, length, 4) in the encoder's one-hot
    encoding. len_max=None pads each batch with N (all-zero rows) to its longest sequence."""
    return OneHotCollate(
        len_max, alphabet=ONEHOT_ALPHABET, value_unknown=ONEHOT_VALUE_UNKNOWN, channels_first=False
    )
