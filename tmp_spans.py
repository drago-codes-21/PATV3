from pat.detectors.runner import DetectorRunner
from pat.fusion import FusionEngine
from pathlib import Path
text = Path('samples/test_blob.txt').read_text(encoding='utf-8')
runner=DetectorRunner(); fused=FusionEngine().fuse(runner.run(text), text=text)
needle = text.index('Billing Address')
for span in fused:
    if span.start >= needle-20 and span.start < needle+200:
        print(span.pii_type, span.start, span.end, repr(span.text))
