# Makefile for HR Assistant

.PHONY: install run lint

install:
	pip install -r requirements.txt

run:
	python src/main.py

lint:
	flake8 .
