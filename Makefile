.PHONY: lint test

# Lint source code

lint:
	# stop the build if there are Python syntax errors or undefined names
	flake8 --filename=*.py --count --show-source --max-line-length=127 --statistics
	# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
	flake8 --filename=*.py --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Run tests

test:
	python -m pytest -vvv tests
