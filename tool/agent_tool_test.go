// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package tool

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

// MockAgent implements interfaces.Agent interface for testing
type MockAgent struct {
	name        string
	description string
}

func (a *MockAgent) GetName() string {
	return a.name
}

func (a *MockAgent) GetDescription() string {
	return a.description
}

// MockRunner implements interfaces.Runner interface for testing
type MockRunner struct {
	result any
	err    error
}

func (r *MockRunner) Run(ctx context.Context, agent any, input string) (any, error) {
	return r.result, r.err
}

// MockResult implements interfaces.RunResult interface for testing
type MockResult struct {
	output string
}

func (r *MockResult) GetFinalOutput() string {
	return r.output
}

func TestNewAgentTool(t *testing.T) {
	// Create mock agent and runner for testing
	mockAgent := &MockAgent{
		name:        "TestAgent",
		description: "Test agent description",
	}
	mockRunner := &MockRunner{
		result: &MockResult{output: "Mock result"},
	}

	// Create an AgentTool
	agentTool, err := NewAgentTool(mockAgent, mockRunner)
	assert.NoError(t, err, "No error should be returned")
	assert.NotNil(t, agentTool, "AgentTool should not be nil")

	// Verify that the name and description are set correctly
	assert.Equal(t, "TestAgent", agentTool.Name(), "Tool name should match agent name")
	assert.Equal(t, "Test agent description", agentTool.Description(), "Tool description should match agent description")

	// Create an AgentTool with custom options
	customTool, err := NewAgentTool(
		mockAgent,
		mockRunner,
		AgentToolOption{
			Name:        "CustomName",
			Description: "Custom description",
		},
	)
	assert.NoError(t, err, "No error should be returned")
	assert.Equal(t, "CustomName", customTool.Name(), "Tool name should match custom name")
	assert.Equal(t, "Custom description", customTool.Description(), "Tool description should match custom description")
}

func TestAgentToolInvoke(t *testing.T) {
	// Create mock agent and runner for testing
	mockAgent := &MockAgent{
		name:        "TestAgent",
		description: "Test agent description",
	}
	mockRunner := &MockRunner{
		result: &MockResult{output: "Mock result"},
	}

	// Create an AgentTool
	agentTool, err := NewAgentTool(mockAgent, mockRunner)
	assert.NoError(t, err, "No error should be returned")

	// Invoke the tool
	result, err := agentTool.Invoke(context.Background(), `{"input": "test input"}`)
	assert.NoError(t, err, "Tool invocation should not return an error")
	assert.Equal(t, "Mock result", result, "Tool result should match mock result")

	// Create a tool with custom output extractor
	customTool, err := NewAgentTool(
		mockAgent,
		mockRunner,
		AgentToolOption{
			OutputExtractor: func(result any) (string, error) {
				return "Custom extracted output", nil
			},
		},
	)
	assert.NoError(t, err, "No error should be returned")

	// Invoke the tool with custom output extractor
	customResult, err := customTool.Invoke(context.Background(), `{"input": "test input"}`)
	assert.NoError(t, err, "Tool invocation should not return an error")
	assert.Equal(t, "Custom extracted output", customResult, "Tool result should match custom output")
}
