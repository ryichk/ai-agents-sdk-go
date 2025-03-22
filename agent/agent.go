// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package agent

import (
	"context"
	"fmt"
	"reflect"

	"github.com/ryichk/ai-agents-sdk-go/guardrail"
	"github.com/ryichk/ai-agents-sdk-go/handoff"
	"github.com/ryichk/ai-agents-sdk-go/interfaces"
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
	//
	// Note: In the Python SDK, this field can directly hold either a string or a callable function.
	// In Go, due to static typing constraints, we use separate fields (dynamicInstructions and
	// asyncDynamicInstructions) to handle dynamic instruction generation.
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
	// This is part of the Go implementation's approach to handle callable instructions
	// that are synchronous (non-async) in the Python SDK.
	dynamicInstructions InstructionsFunc

	// asyncDynamicInstructions is a function that generates dynamic instructions asynchronously
	// This is part of the Go implementation's approach to handle coroutine functions
	// (async def) for instructions in the Python SDK.
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

// SetDynamicInstructions sets a function to dynamically generate instructions
// In the Python SDK, this functionality is achieved by directly assigning
// a callable to the instructions field.
func (a *Agent) SetDynamicInstructions(f InstructionsFunc) {
	a.dynamicInstructions = f
}

// SetAsyncDynamicInstructions sets an async function to dynamically generate instructions
// In the Python SDK, this functionality is achieved by directly assigning
// an async callable (coroutine function) to the instructions field.
func (a *Agent) SetAsyncDynamicInstructions(f AsyncInstructionsFunc) {
	a.asyncDynamicInstructions = f
}

// GetName returns the agent name
// Implements interfaces.Agent interface
func (a *Agent) GetName() string {
	return a.Name
}

// GetDescription returns the agent description
// Implements interfaces.Agent interface
func (a *Agent) GetDescription() string {
	if a.HandoffDescription != "" {
		return a.HandoffDescription
	}
	return fmt.Sprintf("Agent %s", a.Name)
}

// AsTool converts this agent to a tool that can be used by other agents.
//
// This is different from a handoff in two ways:
//  1. In handoffs, the new agent receives the conversation history.
//     In this tool, the new agent receives generated input.
//  2. In handoffs, the new agent takes over the conversation.
//     In this tool, the new agent is called as a tool, and the conversation is continued by the original agent.
func (a *Agent) AsTool(runner any, options ...tool.AgentToolOption) (tool.Tool, error) {
	if runner == nil {
		return nil, fmt.Errorf("runner cannot be nil")
	}

	// Check if runner implements the Runner interface
	runnerIF, ok := runner.(interfaces.Runner)
	if !ok {
		return nil, fmt.Errorf("runner must implement interfaces.Runner interface")
	}

	return tool.NewAgentTool(a, runnerIF, options...)
}

// GetSystemPrompt returns the current system prompt (instructions)
// This is equivalent to the `get_system_prompt` method in the Python SDK.
// While the Python version can inspect the type of instructions at runtime,
// the Go version uses predefined fields for different types of instruction sources.
func (a *Agent) GetSystemPrompt(ctx context.Context) (string, error) {
	if a.asyncDynamicInstructions != nil {
		return a.asyncDynamicInstructions(ctx)
	}
	if a.dynamicInstructions != nil {
		return a.dynamicInstructions(ctx), nil
	}
	return a.Instructions, nil
}
