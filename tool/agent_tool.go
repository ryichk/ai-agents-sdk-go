// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package tool

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/ryichk/ai-agents-sdk-go/interfaces"
)

// AgentTool wraps an agent as a tool
type AgentTool struct {
	name        string
	description string
	agent       interfaces.Agent
	runner      interfaces.Runner
	// Optional custom output extractor function
	outputExtractor func(result any) (string, error)
}

func (t *AgentTool) Name() string {
	return t.name
}

func (t *AgentTool) Description() string {
	return t.description
}

// ParamsJSONSchema returns the JSON schema for the tool parameters
func (t *AgentTool) ParamsJSONSchema() map[string]any {
	// When an agent is used as a tool, the input is simply a text string
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"input": map[string]any{
				"type":        "string",
				"description": "The input to send to the agent",
			},
		},
		"required": []string{"input"},
	}
}

// Invoke executes the agent with the provided input
func (t *AgentTool) Invoke(ctx context.Context, paramsJSON string) (string, error) {
	// Parse parameters
	var params struct {
		Input string `json:"input"`
	}
	if err := json.Unmarshal([]byte(paramsJSON), &params); err != nil {
		return "", fmt.Errorf("failed to parse parameters: %w", err)
	}

	// Execute the agent
	result, err := t.runner.Run(ctx, t.agent, params.Input)
	if err != nil {
		return "", fmt.Errorf("failed to run agent: %w", err)
	}

	// Use custom output extractor if provided
	if t.outputExtractor != nil {
		return t.outputExtractor(result)
	}

	// Default: return the final output
	if runResult, ok := result.(interfaces.RunResult); ok {
		return runResult.GetFinalOutput(), nil
	}

	return fmt.Sprintf("%v", result), nil
}

// AgentToolOption represents options for creating an AgentTool
type AgentToolOption struct {
	// Name for the tool (optional, defaults to agent's name)
	Name string
	// Description for the tool (optional, defaults to agent's description)
	Description string
	// Custom function to extract output from agent result (optional)
	OutputExtractor func(result any) (string, error)
}

// NewAgentTool converts an agent to a tool
//
// This function allows using an agent as a tool that can be called by other agents.
// This is different from handoffs in two ways:
//  1. With tools, the target agent receives specific input (not the conversation history)
//  2. With tools, the original agent continues the conversation after the tool call
//     (instead of the target agent taking over)
//
// Args:
//   - a: The agent to convert to a tool
//   - r: The runner implementation to execute the agent
//   - options: Optional configuration for the agent tool
//
// Returns:
//   - A Tool that invokes the underlying agent
//   - An error if the agent cannot be converted to a tool
func NewAgentTool(a interfaces.Agent, r interfaces.Runner, options ...AgentToolOption) (Tool, error) {
	if a == nil {
		return nil, fmt.Errorf("agent cannot be nil")
	}
	if r == nil {
		return nil, fmt.Errorf("runner cannot be nil")
	}

	// Create the tool
	agentTool := &AgentTool{
		agent:       a,
		runner:      r,
		name:        a.GetName(),
		description: a.GetDescription(),
	}

	// Apply custom options
	if len(options) > 0 {
		option := options[0]
		if option.Name != "" {
			agentTool.name = option.Name
		}
		if option.Description != "" {
			agentTool.description = option.Description
		}
		if option.OutputExtractor != nil {
			agentTool.outputExtractor = option.OutputExtractor
		}
	}

	return agentTool, nil
}
