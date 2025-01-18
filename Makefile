all: setup run

setup:
	@python3 -m venv venv && \
	. ./venv/bin/activate && \
	pip3 install -r requirements.txt

KNN:
	@. ./venv/bin/activate && \
	python3 KNN.py

FNN:
	@. ./venv/bin/activate && \
	python3 FNN.py


fclean:
	@rm -rf ./venv
	
