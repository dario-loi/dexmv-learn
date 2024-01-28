
install:
	@echo "Getting all dependencies up to date, ensure that an environment called dexmv-learn exists"
	pip install -e .
	pip install -e mjrl/.
