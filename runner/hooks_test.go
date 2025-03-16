// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package runner

import (
	"context"
	"testing"

	"github.com/ryichk/ai-agents-sdk-go/agent"
	"github.com/ryichk/ai-agents-sdk-go/model"
	"github.com/ryichk/ai-agents-sdk-go/testutil"
	"github.com/ryichk/ai-agents-sdk-go/tool"
	"github.com/stretchr/testify/assert"
)

func TestRunWithHooks(t *testing.T) {
	fakeModel := NewFakeModel()
	fakeModel.SetNextOutput([]model.Message{GetTextMessage("This is a test response")})

	testAgent := agent.New("test-agent", "Test instructions")
	testAgent.SetModel("gpt-4o")

	hooks := &testutil.TestHooks{}
	testAgent.SetHooks(newAgentHooksAdapter(hooks))

	ctx := context.Background()
	config := RunConfig{
		ModelProvider: fakeModel,
		MaxTurns:      5,
	}
	result, err := RunWithConfig(ctx, testAgent, "Hello", config)

	assert.NoError(t, err, "RunWithConfig should not return an error")
	assert.NotNil(t, result, "Result should not be nil")

	assert.Equal(t, 1, hooks.StartCount, "OnStart should be called once")
	assert.Equal(t, 1, hooks.EndCount, "OnEnd should be called once")

	assert.Equal(t, 0, hooks.HandoffCount, "OnHandoff should not be called")

	assert.Equal(t, 0, hooks.ToolStartCount, "OnToolStart should not be called")
	assert.Equal(t, 0, hooks.ToolEndCount, "OnToolEnd should not be called")
}

// TestRunWithToolCallHooks tests that hooks are called correctly during tool calls
func TestRunWithToolCallHooks(t *testing.T) {
	fakeModel := NewFakeModel()

	testTool := NewFunctionTool("test-tool", "Tool execution result")

	// Set up multiple turn outputs
	fakeModel.AddMultipleTurnOutputs([][]model.Message{
		// First turn: Tool call
		{GetFunctionToolCall("test-tool", `{"a":"b"}`)},
		// Second turn: Final response
		{GetTextMessage("Final response after tool call")},
	})

	testAgent := agent.New("test-agent", "Test instructions")
	testAgent.SetModel("gpt-4o")
	testAgent.AddTool(testTool)

	hooks := &testutil.TestHooks{}
	testAgent.SetHooks(newAgentHooksAdapter(hooks))

	ctx := context.Background()
	config := RunConfig{
		ModelProvider: fakeModel,
		MaxTurns:      5,
	}
	result, err := RunWithConfig(ctx, testAgent, "Use the test tool", config)

	assert.NoError(t, err, "RunWithConfig should not return an error")
	assert.NotNil(t, result, "Result should not be nil")

	assert.Equal(t, 1, hooks.StartCount, "OnStart should be called once")
	assert.Equal(t, 1, hooks.EndCount, "OnEnd should be called once")

	assert.Equal(t, 1, hooks.ToolStartCount, "OnToolStart should be called once")
	assert.Equal(t, 1, hooks.ToolEndCount, "OnToolEnd should be called once")

	assert.Equal(t, 0, hooks.HandoffCount, "OnHandoff should not be called")
}

type agentHooksAdapter struct {
	hooks *testutil.TestHooks
}

func newAgentHooksAdapter(hooks *testutil.TestHooks) *agentHooksAdapter {
	return &agentHooksAdapter{
		hooks: hooks,
	}
}

func (a *agentHooksAdapter) OnStart(ctx context.Context, agent *agent.Agent) error {
	return a.hooks.OnStart(ctx, agent)
}

func (a *agentHooksAdapter) OnEnd(ctx context.Context, agent *agent.Agent, output any) error {
	return a.hooks.OnEnd(ctx, agent, output)
}

func (a *agentHooksAdapter) OnHandoff(ctx context.Context, agent *agent.Agent, source *agent.Agent) error {
	return a.hooks.OnHandoff(ctx, agent, source)
}

func (a *agentHooksAdapter) OnToolStart(ctx context.Context, agent *agent.Agent, t tool.Tool) error {
	return a.hooks.OnToolStart(ctx, agent, t)
}

func (a *agentHooksAdapter) OnToolEnd(ctx context.Context, agent *agent.Agent, t tool.Tool, result string) error {
	return a.hooks.OnToolEnd(ctx, agent, t, result)
}

// TestRunWithHandoffHooks tests that hooks are called correctly during handoffs
func TestRunWithHandoffHooks(t *testing.T) {
	fakeModel1 := NewFakeModel()
	fakeModel2 := NewFakeModel()

	agent1 := agent.New("agent1", "Test instructions for agent1")
	agent1.SetModel("gpt-4o")

	agent2 := agent.New("agent2", "Test instructions for agent2")
	agent2.SetModel("gpt-4o")

	handoffName := "handoff"
	testHandoff := testutil.NewTestHandoff(handoffName, agent2)
	agent1.AddHandoff(testHandoff)

	handoffCall := model.ToolCall{
		ID:   "handoff_123",
		Type: "function",
		Function: model.FunctionCall{
			Name:      handoffName,
			Arguments: `{"input": "Handoff input"}`,
		},
	}
	handoffMessage := model.Message{
		Role:      "assistant",
		Content:   "Handing off to agent2",
		ToolCalls: []model.ToolCall{handoffCall},
	}
	fakeModel1.SetNextOutput([]model.Message{handoffMessage})

	fakeModel2.SetNextOutput([]model.Message{GetTextMessage("Response from agent2")})

	hooks1 := &testutil.TestHooks{}
	hooks2 := &testutil.TestHooks{}
	agent1.SetHooks(newAgentHooksAdapter(hooks1))
	agent2.SetHooks(newAgentHooksAdapter(hooks2))

	handoffCallbackExecuted := false
	handoffInputReceived := ""

	ctx := context.Background()
	config := RunConfig{
		ModelProvider: fakeModel1,
		MaxTurns:      10,
		HandoffCallback: func(ctx context.Context, targetAgent *agent.Agent, sourceAgent *agent.Agent, inputJSON string) error {
			handoffCallbackExecuted = true
			handoffInputReceived = inputJSON
			return nil
		},
	}
	result, err := RunWithConfig(ctx, agent1, "Start with agent1", config)

	assert.NoError(t, err, "RunWithConfig should not return an error")
	assert.NotNil(t, result, "Result should not be nil")

	// Verify handoff callback was executed
	assert.True(t, handoffCallbackExecuted, "HandoffCallback should be executed")
	assert.Equal(t, `{"input": "Handoff input"}`, handoffInputReceived, "HandoffCallback should receive correct input")

	// Verify agent1's OnStart was called
	assert.Equal(t, 1, hooks1.StartCount, "agent1 OnStart should be called once")
	assert.Equal(t, 0, hooks1.EndCount, "agent1 OnEnd should not be called")
	assert.Equal(t, 0, hooks1.HandoffCount, "agent1 OnHandoff should not be called")

	// Verify agent2's OnStart and OnEnd were called
	assert.Equal(t, 1, hooks2.StartCount, "agent2 OnStart should be called once")
	assert.Equal(t, 1, hooks2.EndCount, "agent2 OnEnd should be called once")
	assert.Equal(t, 1, hooks2.HandoffCount, "agent2 OnHandoff should be called once")
}
