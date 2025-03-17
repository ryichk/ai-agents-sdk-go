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

// An agent is an AI model configured with instructions, tools, guardrails, and handoffs and more.
//
// We strongly recommend passing `instructions`, which is the "system prompt" for the agent.
// In addition, you can pass `description`, which is a human-readable description of the agent, used
// when the agent is used inside tools/handoffs.
//
// Agents are generic on the context type. The context is a (mutable) object you create.
// It is passed to tool functions, handoffs, guardrails, etc.
type Agent struct {
	// The name of the agent.
	Name string

	// The instructions for the agent. Will be used as the "system prompt" when this agent is invoked.
	// Describes what the agent should do, and how it responds.
	Instructions string

	// A description of the agent.
	// This is used when the agent is used as a handoff, so that an LLM knows what it does and when to invoke it.
	HandoffDescription string

	// Handoffs are sub-agents that the agent can delegate to.
	// You can provide a list of handoffs, and the agent can choose to delegate to them if relevant.
	// Allows for separation of concerns and modularity.
	Handoffs []handoff.Handoff

	// The Model implementation to use when invoking the LLM.
	// By default, if not set, the agent will use the default model configured in `runner.DefaultRunConfig`.
	Model string

	// Configures model-specific tuning parameters (e.g. Temperature, TopP).
	ModelSettings model.Settings

	// A list of tools that the agent can use.
	Tools []tool.Tool

	// A list of checks than run in parallel to the agent's execution, before generating a response.
	// Runs only if the agent is the first agent in the chain.
	InputGuardrails []guardrail.InputGuardrail

	// A list of checks that run on the final output of the agent, after generating a response.
	// Runs only if the agent produces a final output.
	OutputGuardrails []guardrail.OutputGuardrail

	// The type of the output object. If not provided, the output will be `string`.
	OutputType reflect.Type

	// A interface that receives callbacks on various lifecycle events for this agent.
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

// Make a copy of the agent, with the given arguments changed.
func (a *Agent) Clone() *Agent {
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

	return cloned
}
