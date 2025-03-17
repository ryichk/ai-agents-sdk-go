// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package agent

import (
	"reflect"

	"github.com/ryichk/ai-agents-sdk-go/guardrail"
	"github.com/ryichk/ai-agents-sdk-go/handoff"
	"github.com/ryichk/ai-agents-sdk-go/model"
	"github.com/ryichk/ai-agents-sdk-go/tool"

)

type CloneOption func(*Agent)

func WithName(name string) CloneOption {
	return func(a *Agent) {
		a.Name = name
	}
}

func WithInstructions(instructions string) CloneOption {
	return func(a *Agent) {
		a.Instructions = instructions
	}
}

func WithHandoffDescription(desc string) CloneOption {
	return func(a *Agent) {
		a.HandoffDescription = desc
	}
}

func WithModel(model string) CloneOption {
	return func(a *Agent) {
		a.Model = model
	}
}

func WithModelSettings(settings model.Settings) CloneOption {
	return func(a *Agent) {
		a.ModelSettings = settings
	}
}

func WithTools(tools []tool.Tool) CloneOption {
	return func(a *Agent) {
		a.Tools = make([]tool.Tool, len(tools))
		copy(a.Tools, tools)
	}
}

func WithHandoffs(handoffs []handoff.Handoff) CloneOption {
	return func(a *Agent) {
		a.Handoffs = make([]handoff.Handoff, len(handoffs))
		copy(a.Handoffs, handoffs)
	}
}

func WithOutputType(outputType reflect.Type) CloneOption {
	return func(a *Agent) {
		a.OutputType = outputType
	}
}

// WithHooks sets the hooks of the agent
func WithHooks(hooks Hooks) CloneOption {
	return func(a *Agent) {
		a.Hooks = hooks
	}
}
func WithInputGuardrails(guardrails []guardrail.InputGuardrail) CloneOption {
	return func(a *Agent) {
		a.InputGuardrails = make([]guardrail.InputGuardrail, len(guardrails))
		copy(a.InputGuardrails, guardrails)
	}
}

func WithOutputGuardrails(guardrails []guardrail.OutputGuardrail) CloneOption {
	return func(a *Agent) {
		a.OutputGuardrails = make([]guardrail.OutputGuardrail, len(guardrails))
		copy(a.OutputGuardrails, guardrails)
	}
}

func WithDynamicInstructions(fn InstructionsFunc) CloneOption {
	return func(a *Agent) {
		a.dynamicInstructions = fn
	}
}

func WithAsyncDynamicInstructions(fn AsyncInstructionsFunc) CloneOption {
	return func(a *Agent) {
		a.asyncDynamicInstructions = fn
	}
}

// Make a copy of the agent, with the given arguments changed.
// For example, you could do:
// ```
// newAgent := agent.Clone(
//
//	agent.WithInstructions("You are a helpful assistant."),
//
// )
// ```
func (a *Agent) Clone(opts ...CloneOption) *Agent {
	cloned := &Agent{
		Name:               a.Name,
		Instructions:       a.Instructions,
		HandoffDescription: a.HandoffDescription,
		Model:              a.Model,
		ModelSettings:      a.ModelSettings,
		Tools:              make([]tool.Tool, len(a.Tools)),
		Handoffs:           make([]handoff.Handoff, len(a.Handoffs)),
		InputGuardrails:    make([]guardrail.InputGuardrail, len(a.InputGuardrails)),
		OutputGuardrails:   make([]guardrail.OutputGuardrail, len(a.OutputGuardrails)),
		OutputType:         a.OutputType,
		Hooks:              a.Hooks,
	}

	copy(cloned.Tools, a.Tools)
	copy(cloned.Handoffs, a.Handoffs)
	copy(cloned.InputGuardrails, a.InputGuardrails)
	copy(cloned.OutputGuardrails, a.OutputGuardrails)

	// Apply any options to modify the cloned agent
	for _, opt := range opts {
		opt(cloned)
	}

	return cloned
}
