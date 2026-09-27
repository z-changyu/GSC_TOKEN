# Dataset protocol

ScienceQA (`derek-thomas/ScienceQA`) is primary. Store choices and map the answer index to its text while preserving the original answer/context/explanation in metadata.

GQA (`vikhyatk/gqa-val`) is relational/compositional. The adapter accepts flattened records and defensively flattens grouped QA lists; verify the downloaded schema before generating the manifest.

A-OKVQA (`HuggingFaceM4/A-OKVQA`) is knowledge-intensive. Preserve choices, direct answers, rationales, and all other source fields in metadata. Official evaluation should consume all annotator answers; the unified `ground_truth` is only a canonical convenience value.

Manifests contain stable IDs in fixed order. Subsets use seed 2026 and are shared across methods. Raw files are never deleted or rewritten.

