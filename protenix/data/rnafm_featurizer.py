# protenix/data/rnafm_featurizer.py
import torch
import numpy as np
import fm
import logging

logger = logging.getLogger(__name__)

class RNAFMEmbedder:
    """Singleton wrapper around RNA-FM t12 model."""
    _instance = None

    def __new__(cls, device=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model(device)
        return cls._instance

    def _init_model(self, device):
        # pick device only once
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            device = "cpu"
        self.device = device

        # load model + alphabet
        try:
            model, alphabet = fm.pretrained.rna_fm_t12()
        except Exception as e:
            logger.exception("Failed to load RNA-FM:")
            raise
        self.model = model.to(self.device).eval()
        self.alphabet = alphabet

    def embed(self, seq: str) -> np.ndarray:
        """
        Tokenize `seq`, run through RNA-FM, return (L, 640) float32 numpy.
        """
        # build token
        try:
            idxs = [self.alphabet.get_idx(s) for s in seq]
        except KeyError as e:
            raise ValueError(f"Invalid character {e.args[0]} in sequence") from e

        token = torch.tensor([idxs], dtype=torch.long, device=self.device)  # (1,L)
        with torch.no_grad():
            out = self.model(token, repr_layers=[12])
            emb = out["representations"][12]  # (1, L, 640)
        emb = emb.squeeze(0).cpu().detach().numpy().astype(np.float32)
        # free GPU memory right away
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return emb
