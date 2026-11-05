##
## EPITECH PROJECT, 2026
## my_torch
## File description:
## Makefile
##

UV              :=      uv
PYTHON          :=      python3

ANALYZER        :=      my_torch_analyzer
GENERATOR       :=      my_torch_generator
BINARIES        :=      $(ANALYZER) $(GENERATOR)

# Default OpenBLAS threads of the analyzer: mini-batch matrices are small, and
# more threads spend longer coordinating than computing. Override at run time
# with OPENBLAS_NUM_THREADS=n ./my_torch_analyzer ...
BLAS_THREADS    :=      4

.PHONY: all install run tests_run clean fclean re help

all: $(BINARIES)

install:
	$(UV) sync

$(ANALYZER):
	@printf '#!/bin/sh\nexport OPENBLAS_NUM_THREADS="$${OPENBLAS_NUM_THREADS:-$(BLAS_THREADS)}"\nexport PYTHONPATH="$$(dirname "$$0")/src$${PYTHONPATH:+:$$PYTHONPATH}"\nexec $(PYTHON) -m my_torch.analyser.analyzer "$$@"\n' > $@
	@chmod +x $@
	@echo "$@ ready"

$(GENERATOR):
	@printf '#!/bin/sh\nexport PYTHONPATH="$$(dirname "$$0")/src$${PYTHONPATH:+:$$PYTHONPATH}"\nexec $(PYTHON) -m my_torch.generator.generator "$$@"\n' > $@
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
	@echo "install    sync the uv development environment"
	@echo "run        run the analyzer with --help"
	@echo "tests_run  run the test suite"
	@echo "clean      remove caches"
	@echo "fclean     clean + remove binaries and the venv"
	@echo "re         fclean + all"
