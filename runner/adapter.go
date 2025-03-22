// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package runner

import (
	"context"
	"fmt"

	"github.com/ryichk/ai-agents-sdk-go/agent"
)

// Adapter is an adapter that implements the Runner interface
type Adapter struct{}

// NewAdapter creates a new Adapter instance
func NewAdapter() *Adapter {
	return &Adapter{}
}

// Run executes the agent
func (r *Adapter) Run(ctx context.Context, agentIF any, input string) (any, error) {
	// Check if the agent is of type *agent.Agent
	a, ok := agentIF.(*agent.Agent)
	if !ok {
		return nil, fmt.Errorf("agent must be of type *agent.Agent")
	}

	// Delegate the processing to Runner.RunWithConfig
	if DefaultProvider == nil {
		return nil, ErrModelProviderRequired
	}

	config := DefaultRunConfig()
	config.ModelProvider = DefaultProvider

	result, err := RunWithConfig(ctx, a, input, config)
	if err != nil {
		return nil, err
	}

	// Wrap the result to satisfy the interfaces.RunResult interface
	return &GetResult{Result: result}, nil
}

// GetResult is a helper that calls the GetFinalOutput method from the Result
// Implements interfaces.RunResult interface
type GetResult struct {
	Result *Result
}

// GetFinalOutput gets the FinalOutput from the Result
func (g *GetResult) GetFinalOutput() string {
	if g.Result == nil {
		return ""
	}
	return g.Result.FinalOutput
}
