// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package agent

import (
	"context"
	"reflect"

	"github.com/ryichk/ai-agents-sdk-go/guardrail"
	"github.com/ryichk/ai-agents-sdk-go/handoff"
	"github.com/ryichk/ai-agents-sdk-go/model"
	"github.com/ryichk/ai-agents-sdk-go/tool"
)

// InstructionsFunc is a function type that generates dynamic instructions
type InstructionsFunc func(ctx context.Context) string

// AsyncInstructionsFunc is a function type that generates dynamic instructions asynchronously
type AsyncInstructionsFunc func(ctx context.Context) (string, error)

// Agent represents an AI model configured with instructions, tools, guardrails, and handoffs
type Agent struct {
	Name string

	// Instructions are used as a "system prompt" when the agent is called
	Instructions string

	// HandoffDesc is the description used when this agent is handed off from another agent
	HandoffDesc string

	Handoffs []handoff.Handoff

	// Model is the name of the model used by this agent
	Model string

	ModelSettings model.Settings

	Tools []tool.Tool

	InputGuardrails []guardrail.InputGuardrail

	OutputGuardrails []guardrail.OutputGuardrail

	OutputType reflect.Type

	// Hooks handle lifecycle events of the agent
	Hooks Hooks

	// dynamicInstructions is a function that generates dynamic instructions
	dynamicInstructions InstructionsFunc

	// asyncDynamicInstructions is a function that generates dynamic instructions asynchronously
	asyncDynamicInstructions AsyncInstructionsFunc
}

func New(name string, instructions string) *Agent {
	return &Agent{
		Name:         name,
		Instructions: instructions,
		Tools:        make([]tool.Tool, 0),
		Handoffs:     make([]handoff.Handoff, 0),
		Hooks:        &BaseAgentHooks{},
	}
}

func (a *Agent) AddTool(tool tool.Tool) {
	a.Tools = append(a.Tools, tool)
}

func (a *Agent) AddHandoff(handoff handoff.Handoff) {
	a.Handoffs = append(a.Handoffs, handoff)
}

func (a *Agent) AddHandoffs(handoffs ...handoff.Handoff) {
	a.Handoffs = append(a.Handoffs, handoffs...)
}

func (a *Agent) AddInputGuardrail(guardrail guardrail.InputGuardrail) {
	a.InputGuardrails = append(a.InputGuardrails, guardrail)
}

func (a *Agent) AddOutputGuardrail(guardrail guardrail.OutputGuardrail) {
	a.OutputGuardrails = append(a.OutputGuardrails, guardrail)
}

func (a *Agent) SetModel(model string) {
	a.Model = model
}

func (a *Agent) SetModelSettings(settings model.Settings) {
	a.ModelSettings = settings
}

func (a *Agent) SetOutputType(outputType reflect.Type) {
	a.OutputType = outputType
}

func (a *Agent) SetHooks(hooks Hooks) {
	a.Hooks = hooks
}

func (a *Agent) SetDynamicInstructions(f InstructionsFunc) {
	a.dynamicInstructions = f
}

func (a *Agent) SetAsyncDynamicInstructions(f AsyncInstructionsFunc) {
	a.asyncDynamicInstructions = f
}

// GetInstructions returns the current instructions
func (a *Agent) GetInstructions(ctx context.Context) (string, error) {
	if a.asyncDynamicInstructions != nil {
		return a.asyncDynamicInstructions(ctx)
	}
	if a.dynamicInstructions != nil {
		return a.dynamicInstructions(ctx), nil
	}
	return a.Instructions, nil
}

// Clone creates a new copy of the agent
func (a *Agent) Clone() *Agent {
	cloned := &Agent{
		Name:             a.Name,
		Instructions:     a.Instructions,
		HandoffDesc:      a.HandoffDesc,
		Model:            a.Model,
		ModelSettings:    a.ModelSettings,
		Tools:            make([]tool.Tool, len(a.Tools)),
		Handoffs:         make([]handoff.Handoff, len(a.Handoffs)),
		InputGuardrails:  make([]guardrail.InputGuardrail, len(a.InputGuardrails)),
		OutputGuardrails: make([]guardrail.OutputGuardrail, len(a.OutputGuardrails)),
		OutputType:       a.OutputType,
		Hooks:            a.Hooks,
	}

	copy(cloned.Tools, a.Tools)
	copy(cloned.Handoffs, a.Handoffs)
	copy(cloned.InputGuardrails, a.InputGuardrails)
	copy(cloned.OutputGuardrails, a.OutputGuardrails)

	return cloned
}
