.PHONY: lint test

# Lint source code

lint:
	# stop the build if there are Python syntax errors or undefined names
	flake8 --filename=*.py --count --show-source --max-line-length=119 --statistics
	# exit-zero treats all errors as warnings
	flake8 --filename=*.py --count --exit-zero --max-line-length=119 --max-complexity=10 --statistics

# Run tests

test:
	python -m pytest -vvv tests
