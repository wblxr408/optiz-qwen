# Scripts

This directory is reserved for executable workflow entrypoints such as:

- baseline runner
- evaluation launcher
- profiling launcher
- data-preparation helpers
- benchmark collection

At the current stage, only the skeleton exists because the external assets are still missing.

## Model download

The teacher-provided official model source is ModelScope:

- https://www.modelscope.cn/models/Qwen/Qwen3.5-2B

Create the local development environment with the organizer's minimal runtime
requirements first:

```bash
conda create -n optiz-qwen python=3.11 -y
conda run -n optiz-qwen python -m pip install -r configs/requirements/dndx_public.txt
```

For local Mac development, install extra helper packages separately. These are
not part of the organizer's minimal dependency list:

```bash
conda run -n optiz-qwen python -m pip install -r configs/requirements/local_dev_extra.txt
```

Then place weights in the repository's ignored raw-resource path:

```bash
conda run -n optiz-qwen bash scripts/download_qwen35_2b_modelscope.sh
```

The target directory is `resources/model_weights/raw/Qwen3.5-2B/`.
