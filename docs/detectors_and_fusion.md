# Detector and Fusion Layer Overview

## Detectors

- **RegexDetector (`regex`)**: config-driven patterns (`pat.config.patterns`) for email, phone, card, IBAN/bank account, IP, URL, government IDs, and context-gated generic numbers. Supports allowlists (safe domains/literals) and validator hooks (Luhn, IBAN, Aadhaar) with metadata on pattern hits.
- **NERDetector (`ml_ner`)**: spaCy-based NER with explicit label mapping, calibration hook, and metadata for raw labels/model. Lazily loads model path from settings.
- **DomainHeuristicsDetector (`domain_heuristic`)**: rule-engine over contextual keywords (`pat.config.heuristics`) with numeric pickers, base scores, and hints (financial/credential/identity). Aggregates multiple rule hits into a single span with matched rule metadata.
- **EmbeddingSimilarityDetector / SemanticSimilarityDetector (`semantic`)**: prototype-based semantic confirmer using shared embedding model. Works on windows around prior detections plus value-heavy sentences; emits similarity breakdown per type in metadata.
- **MLTokenClassifierDetector (`ml_token`)**: mpnet embedding prototype classifier for token/value spans; uses regex candidates when tokenizer unavailable and emits prototype metadata.

All detectors emit `DetectorResult` with aligned `start/end`, `pii_type`, `confidence/score`, `detector_name`, and optional metadata.

## Fusion

- **FusionEngine** enforces sorted, non-overlapping spans, merging overlapping or adjacent compatible spans (names, addresses, numeric identifiers). Context window size and debug logging are configurable via settings/env.
- Aggregates detector evidence into `FusedSpan` carrying `pii_type` (primary), `all_types`, detector sets/scores, merged metadata (including type scores), span text, and left/right context slices.
- Type resolution uses detector-weighted scores with priority ordering favoring credentials/financial/contact over generic numbers; confidence gains a small bonus per agreeing detector.
- Adjacency grouping handles multi-token names/addresses to avoid fragmented spans, and generic numeric candidates are upgraded when corroborated by stronger detectors.
