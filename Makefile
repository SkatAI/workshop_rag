install:
#install
	pip install --upgrade pip && pip install -r requirements.txt

format:
# format
	black -l 100 **/*.py

lint:
# lint
	pylint --disable=R,C,W0105 script/*.py

test:
# test
	pytest -vv

precommit: format lint
