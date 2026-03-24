# Contributing

Thanks for your interest in improving ChurnGuard ML API.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m app.train
pytest -q
python -m uvicorn app.api:app --reload
```

## Pull request checklist

- Keep the API contract stable unless the README is updated.
- Run `python -m app.train` after model-related changes.
- Run `pytest -q` before opening a pull request.
- Update docs when changing endpoints, features, or setup steps.

## Suggested improvement areas

- Model explainability
- Real dataset integration
- Deployment manifests
- Observability and logging
