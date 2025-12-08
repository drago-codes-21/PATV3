from pat.pipeline import RedactionPipeline
from pat.fusion import FusionEngine
from pat.policy import PolicyEngine
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
pipeline = RedactionPipeline(detector_runner=DetectorRunner(), fusion_engine=FusionEngine(), policy_engine=PolicyEngine(), severity_model=StaticSev(), embedding_model=ZeroEmb())
res = pipeline.run(text)
needle = '[EMAIL THREAD – CUSTOMER SUPPORT]'
print('Heading in original:', needle in text)
print('Heading in sanitized:', needle in res['sanitized_text'])
print('Any spans covering heading region:')
idx = text.index(needle)
for span in res['pii_spans']:
    if span.start <= idx <= span.end:
        print(span.pii_type, span.start, span.end, repr(span.text))
