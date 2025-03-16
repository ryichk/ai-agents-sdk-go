// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package agent

import (
	"context"
	"testing"

	"github.com/ryichk/ai-agents-sdk-go/testutil"
	"github.com/ryichk/ai-agents-sdk-go/tool"
	"github.com/stretchr/testify/assert"
)

type testHooksAdapter struct {
	hooks testutil.TestHooks
}

func newTestHooksAdapter() *testHooksAdapter {
	return &testHooksAdapter{
		hooks: testutil.TestHooks{},
	}
}

func (a *testHooksAdapter) OnStart(ctx context.Context, agent *Agent) error {
	return a.hooks.OnStart(ctx, agent)
}

func (a *testHooksAdapter) OnEnd(ctx context.Context, agent *Agent, output any) error {
	return a.hooks.OnEnd(ctx, agent, output)
}

func (a *testHooksAdapter) OnHandoff(ctx context.Context, agent *Agent, source *Agent) error {
	return a.hooks.OnHandoff(ctx, agent, source)
}

func (a *testHooksAdapter) OnToolStart(ctx context.Context, agent *Agent, t tool.Tool) error {
	return a.hooks.OnToolStart(ctx, agent, t)
}

func (a *testHooksAdapter) OnToolEnd(ctx context.Context, agent *Agent, t tool.Tool, result string) error {
	return a.hooks.OnToolEnd(ctx, agent, t, result)
}

func (a *testHooksAdapter) GetStartCount() int {
	return a.hooks.StartCount
}

func (a *testHooksAdapter) GetEndCount() int {
	return a.hooks.EndCount
}

func (a *testHooksAdapter) GetHandoffCount() int {
	return a.hooks.HandoffCount
}

func (a *testHooksAdapter) GetToolStartCount() int {
	return a.hooks.ToolStartCount
}

func (a *testHooksAdapter) GetToolEndCount() int {
	return a.hooks.ToolEndCount
}

func TestBaseAgentHooks(t *testing.T) {
	hooks := &BaseAgentHooks{}
	ctx := context.Background()
	agent1 := New("agent1", "test instructions")
	agent2 := New("agent2", "test instructions")
	dummyTool := testutil.NewTestTool("dummy", "dummy tool", "test result")

	err := hooks.OnStart(ctx, agent1)
	assert.NoError(t, err, "BaseAgentHooks.OnStart should not return an error")

	err = hooks.OnEnd(ctx, agent1, "test output")
	assert.NoError(t, err, "BaseAgentHooks.OnEnd should not return an error")

	err = hooks.OnHandoff(ctx, agent1, agent2)
	assert.NoError(t, err, "BaseAgentHooks.OnHandoff should not return an error")

	err = hooks.OnToolStart(ctx, agent1, dummyTool)
	assert.NoError(t, err, "BaseAgentHooks.OnToolStart should not return an error")

	err = hooks.OnToolEnd(ctx, agent1, dummyTool, "test result")
	assert.NoError(t, err, "BaseAgentHooks.OnToolEnd should not return an error")
}

func TestAgentHooksInterface(t *testing.T) {
	hooks := newTestHooksAdapter()
	ctx := context.Background()
	agent1 := New("agent1", "test instructions")
	agent2 := New("agent2", "test instructions")
	dummyTool := testutil.NewTestTool("dummy", "dummy tool", "test result")

	agent1.SetHooks(hooks)

	err := hooks.OnStart(ctx, agent1)
	assert.NoError(t, err, "TestHooks.OnStart should not return an error")
	assert.Equal(t, 1, hooks.GetStartCount(), "StartCount should be incremented")

	err = hooks.OnEnd(ctx, agent1, "test output")
	assert.NoError(t, err, "TestHooks.OnEnd should not return an error")
	assert.Equal(t, 1, hooks.GetEndCount(), "EndCount should be incremented")

	err = hooks.OnHandoff(ctx, agent1, agent2)
	assert.NoError(t, err, "TestHooks.OnHandoff should not return an error")
	assert.Equal(t, 1, hooks.GetHandoffCount(), "HandoffCount should be incremented")

	err = hooks.OnToolStart(ctx, agent1, dummyTool)
	assert.NoError(t, err, "TestHooks.OnToolStart should not return an error")
	assert.Equal(t, 1, hooks.GetToolStartCount(), "ToolStartCount should be incremented")

	err = hooks.OnToolEnd(ctx, agent1, dummyTool, "test result")
	assert.NoError(t, err, "TestHooks.OnToolEnd should not return an error")
	assert.Equal(t, 1, hooks.GetToolEndCount(), "ToolEndCount should be incremented")
}

func TestAgentWithHooks(t *testing.T) {
	hooks := newTestHooksAdapter()
	agent := New("test", "test instructions")

	_, ok := agent.Hooks.(*BaseAgentHooks)
	assert.True(t, ok, "Agent should have BaseAgentHooks by default")

	agent.SetHooks(hooks)
	assert.Equal(t, hooks, agent.Hooks, "Agent should have the custom hooks set")
}
