# Local Validation Samples

This directory is reserved for team-authored local smoke validation samples for
AWQ W4A16 work. It is not an official benchmark, and it must not use the
official competition evaluation dataset or copied public MMBench TSV content.

Future `samples.jsonl` rows should be JSON objects with lightweight metadata:

```json
{"sample_id":"local-001","image_path":"resources/local_validation/images/local-001.png","question":"What should the model verify?","reference_answer":"A short expected answer, if available.","expected_behavior":"A concise behavior check, if a fixed answer is not appropriate.","category":"ocr","notes":"Why this sample is useful for smoke validation."}
```

Recommended fields:

- `sample_id`
- `image_path`
- `question`
- `reference_answer` or `expected_behavior`
- `category`
- `notes`

Data rules:

- Do not commit large images.
- Do not commit sensitive, private, licensed, or customer data.
- Do not use the official competition evaluation dataset.
- Do not report local smoke results as an official benchmark.
- Do not use local smoke results to claim official performance gains.
- Use these samples only for smoke, regression, and sanity checks.

This phase creates only this README. It does not create `samples.jsonl`, image
files, model outputs, benchmark outputs, or quantized artifacts.
