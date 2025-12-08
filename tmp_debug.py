from pat.pipeline import RedactionPipeline
from pat.fusion import FusionEngine
from pat.policy import PolicyEngine
from pat.detectors.runner import DetectorRunner
from pat.severity.model import SeverityModel
from pathlib import Path

text = Path('samples/test_blob.txt').read_text(encoding='utf-8')
pipeline = RedactionPipeline(detector_runner=DetectorRunner(), fusion_engine=FusionEngine(), policy_engine=PolicyEngine(), severity_model=SeverityModel())
res = pipeline.run(text)
print('Spans near heading:')
for span in res['pii_spans']:
    if span.start < 200:
        print(span.pii_type, span.start, span.end, span.text[:50])
print('Contains heading in sanitized:', 'EMAIL THREAD' in res['sanitized_text'])
