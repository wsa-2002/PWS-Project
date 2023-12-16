default: help

.PHONY: database backend

help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

database:
	cd database && docker-compose up -d

backend:
	cd backend && poetry run uvicorn main:app

install:
	cd backend && poetry install --no-root
	@if [ ! -f backend/.env ]; then\
		cp backend/.env.example backend/.env;\
	fi