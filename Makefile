##
## EPITECH PROJECT, 2026
## my_torch
## File description:
## Makefile
##

UV              :=      uv

ANALYZER        :=      my_torch_analyzer
GENERATOR       :=      my_torch_generator
BINARIES        :=      $(ANALYZER) $(GENERATOR)

.PHONY: all install run tests_run clean fclean re help

all: $(BINARIES)

install:
	$(UV) sync

$(ANALYZER): install
	@printf '#!/bin/sh\nexec uv run --quiet python -m my_torch.analyser.analyzer "$$@"\n' > $@
	@chmod +x $@
	@echo "$@ ready"

$(GENERATOR): install
	@printf '#!/bin/sh\nexec uv run --quiet python -m my_torch.generator.generator "$$@"\n' > $@
	@chmod +x $@
	@echo "$@ ready"

run: $(ANALYZER)
	./$(ANALYZER) --help

tests_run: install
	$(UV) run pytest -v

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2> /dev/null || true
	@rm -rf .pytest_cache .ruff_cache

fclean: clean
	@rm -f $(BINARIES)
	@rm -rf .venv build dist *.egg-info

re: fclean all

help:
	@echo "all        build the binaries"
	@echo "install    sync the uv environment"
	@echo "run        run the analyzer with --help"
	@echo "tests_run  run the test suite"
	@echo "clean      remove caches"
	@echo "fclean     clean + remove binaries and the venv"
	@echo "re         fclean + all"
