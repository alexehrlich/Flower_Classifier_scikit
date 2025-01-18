all: setup run

setup:
	@python3 -m venv venv && \
	source ./venv/bin/activate && \
	pip3 install -r requirements.txt

KNN:
	@source ./venv/bin/activate && \
	python3 KNN.py

FNN:
	@source ./venv/bin/activate && \
	python3 FNN.py


fclean:
	@rm -rf ./venv
	
