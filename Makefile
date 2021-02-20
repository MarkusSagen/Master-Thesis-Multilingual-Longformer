
include .env

export USER_ID      := $(shell id -u)
export USER_NAME    := $(shell whoami)
export PROJECT_DIR  := $(shell pwd)
export COMPOSE_CMD  := docker-compose -f docker-compose.yaml -p ${PROJECT_NAME}_${USER_NAME}
export PKG_DIR      := pkg

# Enable running e3k on machines with no GPU
ifeq (${GPU_IDS}, none)
	export RUNTIME := runc
else
	export RUNTIME := nvidia
endif

# Enable pulling in dependencies in private repos
ifneq (${PRIVATE_DEPS}, none)
	clone_private_deps := for item in ${PRIVATE_DEPS}; do \
            git clone $$item ${PKG_DIR}/$$item; \
	    echo $$item; \
	done
else
	clone_private_deps := echo "Nothing to clone"
endif

.PHONY: build
build:
	mkdir -p ${PKG_DIR}
	$(call clone_private_deps)
	$(COMPOSE_CMD) build
	rm -rf ${PKG_DIR}

.PHONY: logs
logs:
	${COMPOSE_CMD} logs

.PHONY: up
up:
	$(COMPOSE_CMD) up --detach

.PHONY: down
down:
	$(COMPOSE_CMD) down

.PHONY: repl
repl:
	${COMPOSE_CMD} exec repl python3 $(run)

.PHONY: ipython
ipython:
	${COMPOSE_CMD} exec repl ipython $(run)

.PHONY: shell
shell:
	${COMPOSE_CMD} exec repl bash
