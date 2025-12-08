from pat.policy import PolicyEngine
from pat.pipeline.pipeline import _apply_redactions
from pat.fusion import FusionEngine
from pat.detectors.runner import DetectorRunner
from pathlib import Path
import numpy as np
class ZeroEmb:
    def encode_batch(self, texts): return np.zeros((len(texts),4))
    def encode(self, text): return np.zeros(4)
class StaticSev:
    def __init__(self): self.model=True
    def predict(self, fv, pii_type=None): return 0.82, 'HIGH', {'HIGH':0.8}
text = Path('samples/test_blob.txt').read_text(encoding='utf-8')
fused = FusionEngine().fuse(DetectorRunner().run(text), text=text)
decisions = PolicyEngine().decide(fused, {})
sanitized = _apply_redactions(text, decisions)
print(repr(sanitized[350:420]))
