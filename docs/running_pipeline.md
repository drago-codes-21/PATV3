# Running the PAT Pipeline

## CLI (single file)
```
python -m pat.pipeline.run --input input.txt --output sanitized.txt [--policy-config path] [--severity-model path] [--debug]
```
- `--debug` enables detector/severity/policy debug logs via env flags.

## Debug harness
```
python -m pat.pipeline.debug --text "sample input..." [--json-output dbg.json]
```
Prints spans, severity, policy decisions; optional JSON output for inspection.

## Programmatic
```python
from pat.pipeline.core import run_pipeline
result = run_pipeline("text here", context={"channel": "EMAIL_BODY"})
print(result.sanitized_text)
for span in result.spans:
    print(span.pii_type, span.severity_label, getattr(span, "policy_decision", {}))
```

## Notes
- Config/paths from `pat.config.settings`; override via env vars or CLI flags above.
- Models/policy are loaded once per process when using the pipeline helpers.
- VERY_HIGH severity is always masked by policy; non-overlapping spans are enforced from fusion.
