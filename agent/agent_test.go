// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package agent

import (
	"context"
	"reflect"
	"testing"

	"github.com/ryichk/ai-agents-sdk-go/guardrail"
	"github.com/ryichk/ai-agents-sdk-go/handoff"
	"github.com/ryichk/ai-agents-sdk-go/model"
	"github.com/stretchr/testify/assert"
)

type MockTool struct {
	name        string
	description string
}

func (t *MockTool) Name() string {
	return t.name
}

func (t *MockTool) Description() string {
	return t.description
}

func (t *MockTool) ParamsJSONSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"param": map[string]any{
				"type":        "string",
				"description": "Test parameter",
			},
		},
	}
}

func (t *MockTool) Invoke(ctx context.Context, args string) (string, error) {
	return "Mock tool result", nil
}

type MockInputGuardrail struct {
	name        string
	description string
	checkFunc   func(ctx context.Context, input string) (guardrail.InputGuardrailResult, error)
}

func (g *MockInputGuardrail) Name() string {
	return g.name
}

func (g *MockInputGuardrail) Description() string {
	return g.description
}

func (g *MockInputGuardrail) Check(ctx context.Context, input string) (guardrail.InputGuardrailResult, error) {
	return g.checkFunc(ctx, input)
}

type MockOutputGuardrail struct {
	name        string
	description string
	checkFunc   func(ctx context.Context, output string) (guardrail.OutputGuardrailResult, error)
}

func (g *MockOutputGuardrail) Name() string {
	return g.name
}

func (g *MockOutputGuardrail) Description() string {
	return g.description
}

func (g *MockOutputGuardrail) Check(ctx context.Context, output string) (guardrail.OutputGuardrailResult, error) {
	return g.checkFunc(ctx, output)
}

func TestNewAgent(t *testing.T) {
	agent := New("Test Agent", "This is a test agent")

	assert.Equal(t, "Test Agent", agent.Name, "Agent name is incorrect")
	assert.Equal(t, "This is a test agent", agent.Instructions, "Agent instructions are incorrect")
	assert.Empty(t, agent.Tools, "Tools of a new agent should be empty")
	assert.Empty(t, agent.Handoffs, "Handoffs of a new agent should be empty")
	assert.Empty(t, agent.InputGuardrails, "Input guardrails of a new agent should be empty")
	assert.Empty(t, agent.OutputGuardrails, "Output guardrails of a new agent should be empty")
}

func TestAddTool(t *testing.T) {
	agent := New("Test Agent", "This is a test agent")

	tool1 := &MockTool{name: "tool1", description: "Tool 1"}
	agent.AddTool(tool1)
	assert.Len(t, agent.Tools, 1, "Agent should have 1 tool added")
	assert.Equal(t, tool1, agent.Tools[0], "Added tool should match")

	tool2 := &MockTool{name: "tool2", description: "Tool 2"}
	agent.AddTool(tool2)
	assert.Len(t, agent.Tools, 2, "Agent should have 2 tools added")
	assert.Equal(t, tool2, agent.Tools[1], "Second added tool should match")
}

func TestAddHandoff(t *testing.T) {
	agent1 := New("Agent 1", "This is agent 1")
	agent2 := New("Agent 2", "This is agent 2")

	handoff1 := handoff.NewHandoff(agent2, "Handoff to Agent 2")

	agent1.AddHandoff(handoff1)
	assert.Len(t, agent1.Handoffs, 1, "Agent should have 1 handoff added")
	assert.Equal(t, handoff1, agent1.Handoffs[0], "Added handoff should match")

	agent3 := New("Agent 3", "This is agent 3")
	handoff2 := handoff.NewHandoff(agent3, "Handoff to Agent 3")
	agent1.AddHandoff(handoff2)
	assert.Len(t, agent1.Handoffs, 2, "Agent should have 2 handoffs added")
	assert.Equal(t, handoff2, agent1.Handoffs[1], "Second added handoff should match")
}

func TestAddHandoffs(t *testing.T) {
	agent1 := New("Agent 1", "This is agent 1")
	agent2 := New("Agent 2", "This is agent 2")
	agent3 := New("Agent 3", "This is agent 3")

	handoff1 := handoff.NewHandoff(agent2, "Handoff to Agent 2")
	handoff2 := handoff.NewHandoff(agent3, "Handoff to Agent 3")

	agent1.AddHandoffs(handoff1, handoff2)
	assert.Len(t, agent1.Handoffs, 2, "Agent should have 2 handoffs added")
	assert.Equal(t, handoff1, agent1.Handoffs[0], "First handoff should match")
	assert.Equal(t, handoff2, agent1.Handoffs[1], "Second handoff should match")
}

func TestAddGuardrails(t *testing.T) {
	agent := New("Test Agent", "This is a test agent")

	inputGuardrail := &MockInputGuardrail{
		name:        "Test Input Guardrail",
		description: "Test input guardrail",
		checkFunc: func(ctx context.Context, input string) (guardrail.InputGuardrailResult, error) {
			return guardrail.InputGuardrailResult{Allowed: true}, nil
		},
	}
	agent.AddInputGuardrail(inputGuardrail)
	assert.Len(t, agent.InputGuardrails, 1, "Agent should have 1 input guardrail added")
	assert.Equal(t, inputGuardrail, agent.InputGuardrails[0], "Added input guardrail should match")

	outputGuardrail := &MockOutputGuardrail{
		name:        "Test Output Guardrail",
		description: "Test output guardrail",
		checkFunc: func(ctx context.Context, output string) (guardrail.OutputGuardrailResult, error) {
			return guardrail.OutputGuardrailResult{Allowed: true}, nil
		},
	}
	agent.AddOutputGuardrail(outputGuardrail)
	assert.Len(t, agent.OutputGuardrails, 1, "Agent should have 1 output guardrail added")
	assert.Equal(t, outputGuardrail, agent.OutputGuardrails[0], "Added output guardrail should match")
}

func TestSetModel(t *testing.T) {
	agent := New("Test Agent", "This is a test agent")

	agent.SetModel("gpt-4o")
	assert.Equal(t, "gpt-4o", agent.Model, "Model should be set correctly")

	// Change model
	agent.SetModel("gpt-4-turbo")
	assert.Equal(t, "gpt-4-turbo", agent.Model, "Model should be changed correctly")
}

func TestSetModelSettings(t *testing.T) {
	agent := New("Test Agent", "This is a test agent")

	settings := model.Settings{
		Temperature: 0.8,
		MaxTokens:   500,
		TopP:        0.9,
	}
	agent.SetModelSettings(settings)

	assert.Equal(t, settings, agent.ModelSettings, "Model settings should be set correctly")
	assert.Equal(t, 0.8, agent.ModelSettings.Temperature, "Temperature should be set correctly")
	assert.Equal(t, 500, agent.ModelSettings.MaxTokens, "MaxTokens should be set correctly")
	assert.Equal(t, 0.9, agent.ModelSettings.TopP, "TopP should be set correctly")
}

func TestSetOutputType(t *testing.T) {
	agent := New("Test Agent", "This is a test agent")

	type TestOutput struct {
		Name    string `json:"name"`
		Success bool   `json:"success"`
	}
	outputType := reflect.TypeOf(TestOutput{})
	agent.SetOutputType(outputType)

	assert.Equal(t, outputType, agent.OutputType, "Output type should be set correctly")
	assert.Equal(t, "TestOutput", agent.OutputType.Name(), "Output type name should be set correctly")
}

func TestCompleteAgent(t *testing.T) {
	agent := New("Complete Agent", "This is a complete agent")

	tool1 := &MockTool{name: "tool1", description: "Tool 1"}
	agent.AddTool(tool1)

	agent2 := New("Agent 2", "This is agent 2")
	handoff1 := handoff.NewHandoff(agent2, "Handoff to Agent 2")
	agent.AddHandoff(handoff1)

	inputGuardrail := &MockInputGuardrail{
		name:        "Test Input Guardrail",
		description: "Test input guardrail",
		checkFunc: func(ctx context.Context, input string) (guardrail.InputGuardrailResult, error) {
			return guardrail.InputGuardrailResult{Allowed: true}, nil
		},
	}
	agent.AddInputGuardrail(inputGuardrail)

	agent.SetModel("gpt-4o")
	settings := model.Settings{
		Temperature: 0.7,
		MaxTokens:   1000,
	}
	agent.SetModelSettings(settings)

	assert.Equal(t, "Complete Agent", agent.Name)
	assert.Equal(t, "This is a complete agent", agent.Instructions)
	assert.Len(t, agent.Tools, 1)
	assert.Len(t, agent.Handoffs, 1)
	assert.Len(t, agent.InputGuardrails, 1)
	assert.Equal(t, "gpt-4o", agent.Model)
	assert.Equal(t, 0.7, agent.ModelSettings.Temperature)
	assert.Equal(t, 1000, agent.ModelSettings.MaxTokens)
}
