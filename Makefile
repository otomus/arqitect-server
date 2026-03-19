VENV := /Users/oronmozes/Documents/projects/sentient-server/.venv
PYTHON := $(VENV)/bin/python3
PID_DIR := .pids
LOG := brain.log

.PHONY: init start stop restart status logs seed-deps setup

init:
	@$(PYTHON) -m arqitect.cli.main init

seed-deps:
	@echo "Installing community dependencies..."
	@$(PYTHON) -c "from arqitect.brain.community import seed_dependencies; seed_dependencies()" 2>&1 || echo "Seed deps skipped (community not available)"

start: seed-deps $(PID_DIR)
	@echo "Starting brain..."
	@$(PYTHON) -m arqitect.brain.brain --daemon >> $(LOG) 2>&1 & echo $$! > $(PID_DIR)/brain.pid
	@echo "Starting MCP server..."
	@$(PYTHON) -m arqitect.mcp.server >> $(LOG) 2>&1 & echo $$! > $(PID_DIR)/mcp.pid
	@echo "Starting bridge..."
	@$(PYTHON) -m arqitect.bridge.server >> $(LOG) 2>&1 & echo $$! > $(PID_DIR)/bridge.pid
	@echo "All services started. PIDs in $(PID_DIR)/"

stop:
	@for svc in brain mcp bridge; do \
		if [ -f $(PID_DIR)/$$svc.pid ]; then \
			pid=$$(cat $(PID_DIR)/$$svc.pid); \
			if kill -0 $$pid 2>/dev/null; then \
				kill $$pid && echo "Stopped $$svc ($$pid)"; \
			else \
				echo "$$svc ($$pid) not running"; \
			fi; \
			rm -f $(PID_DIR)/$$svc.pid; \
		fi; \
	done
	@# Kill any stray arqitect processes not tracked by PID files
	@pkill -f 'arqitect\.brain\.brain' 2>/dev/null && echo "Killed stray brain process(es)" || true
	@pkill -f 'arqitect\.mcp\.server' 2>/dev/null && echo "Killed stray mcp process(es)" || true
	@pkill -f 'arqitect\.bridge\.server' 2>/dev/null && echo "Killed stray bridge process(es)" || true

restart: stop
	@sleep 2
	@$(MAKE) start

status:
	@for svc in brain mcp bridge; do \
		if [ -f $(PID_DIR)/$$svc.pid ]; then \
			pid=$$(cat $(PID_DIR)/$$svc.pid); \
			if kill -0 $$pid 2>/dev/null; then \
				echo "$$svc: running ($$pid)"; \
			else \
				echo "$$svc: dead (stale pid $$pid)"; \
			fi; \
		else \
			echo "$$svc: not started"; \
		fi; \
	done

logs:
	@tail -f $(LOG)

setup:
	@echo "=== Arqitect Server Setup ==="
	@echo "1. Creating virtual environment..."
	@python3 -m venv $(VENV) 2>/dev/null || echo "Venv already exists"
	@echo "2. Installing server dependencies..."
	@$(VENV)/bin/pip install -r requirements.txt --quiet 2>/dev/null || echo "No requirements.txt found"
	@echo "3. Syncing community manifest..."
	@$(PYTHON) -c "from arqitect.brain.community import sync_manifest; sync_manifest()" 2>&1 || echo "Manifest sync skipped"
	@echo "4. Seeding community tools..."
	@$(PYTHON) -c "from arqitect.brain.community import seed_tools; seed_tools()" 2>&1 || echo "Tool seeding skipped"
	@echo "5. Installing community dependencies..."
	@$(MAKE) seed-deps
	@echo "6. Starting services..."
	@$(MAKE) start
	@echo "=== Setup complete ==="

$(PID_DIR):
	@mkdir -p $(PID_DIR)
