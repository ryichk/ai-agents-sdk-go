// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package agent

import (
	"context"

	"github.com/ryichk/ai-agents-sdk-go/tool"
)

// Hooks is the interface for agent lifecycle hooks
type Hooks interface {
	// OnStart is called when the agent starts execution
	OnStart(ctx context.Context, agent *Agent) error

	// OnEnd is called when the agent finishes execution
	OnEnd(ctx context.Context, agent *Agent, output any) error

	// OnHandoff is called when the agent hands off to another agent
	OnHandoff(ctx context.Context, nextAgent *Agent, currentAgent *Agent) error

	// OnToolStart is called when a tool starts execution
	OnToolStart(ctx context.Context, agent *Agent, tool tool.Tool) error

	// OnToolEnd is called when a tool finishes execution
	OnToolEnd(ctx context.Context, agent *Agent, tool tool.Tool, output string) error
}

// BaseAgentHooks provides a basic implementation of the Hooks interface
type BaseAgentHooks struct{}

func (h *BaseAgentHooks) OnStart(ctx context.Context, agent *Agent) error {
	return nil
}

func (h *BaseAgentHooks) OnEnd(ctx context.Context, agent *Agent, output any) error {
	return nil
}

func (h *BaseAgentHooks) OnHandoff(ctx context.Context, nextAgent *Agent, currentAgent *Agent) error {
	return nil
}

func (h *BaseAgentHooks) OnToolStart(ctx context.Context, agent *Agent, tool tool.Tool) error {
	return nil
}

func (h *BaseAgentHooks) OnToolEnd(ctx context.Context, agent *Agent, tool tool.Tool, output string) error {
	return nil
}
