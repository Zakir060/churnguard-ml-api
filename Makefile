install:
	pip install -r requirements.txt

train:
	python -m app.train

serve:
	uvicorn app.api:app --reload

test:
	pytest -q
