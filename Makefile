.PHONY: all build test clean lint install-tools tidy bench cover vet fmt check gosec

GOPATH := $(shell go env GOPATH)
BIN_DIR := $(GOPATH)/bin
GOLANGCI_LINT := $(BIN_DIR)/golangci-lint
GOSEC := $(BIN_DIR)/gosec

all: test build

build:
	@echo "Building..."
	@go build ./...

test:
	@echo "Running tests..."
	@go test -v ./...

clean:
	@echo "Cleaning..."
	@go clean
	@rm -f coverage.txt

$(GOLANGCI_LINT):
	@echo "Installing golangci-lint..."
	@go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

$(GOSEC):
	@echo "Installing gosec..."
	@go install github.com/securego/gosec/v2/cmd/gosec@latest

lint: $(GOLANGCI_LINT)
	@echo "Running linter..."
	@$(GOLANGCI_LINT) run --timeout=5m

gosec: $(GOSEC)
	@echo "Running security scanner..."
	@$(GOSEC) -exclude-dir=.github -exclude-dir=examples ./...

install-tools: $(GOLANGCI_LINT) $(GOSEC)
	@echo "All tools installed"

tidy:
	@echo "Running go mod tidy..."
	@go mod tidy

bench:
	@echo "Running benchmarks..."
	@go test -bench=. -benchmem ./...

cover:
	@echo "Running tests with coverage..."
	@go test -race -coverprofile=coverage.txt -covermode=atomic ./...
	@go tool cover -html=coverage.txt

vet:
	@echo "Running go vet..."
	@go vet ./...

fmt:
	@echo "Running go fmt..."
	@go fmt ./...

check: fmt vet lint gosec test
	@echo "All checks passed!"
