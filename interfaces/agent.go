// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package interfaces

import (
	"context"
)

// Agent defines the interface that an agent must implement to be used as a tool or with a runner
type Agent interface {
	// GetName returns the name of the agent
	GetName() string

	// GetDescription returns the description of the agent
	GetDescription() string
}

// Runner defines the interface for executing an agent
type Runner interface {
	// Run executes the agent with the given input and returns the result
	Run(ctx context.Context, agent any, input string) (any, error)
}

// RunResult defines the interface for agent execution results
type RunResult interface {
	// GetFinalOutput returns the final output of the agent execution
	GetFinalOutput() string
}
