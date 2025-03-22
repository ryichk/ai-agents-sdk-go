// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package runner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"time"

	"github.com/ryichk/ai-agents-sdk-go/agent"
	"github.com/ryichk/ai-agents-sdk-go/handoff"
	"github.com/ryichk/ai-agents-sdk-go/model"
	"github.com/ryichk/ai-agents-sdk-go/tool"
	"github.com/ryichk/ai-agents-sdk-go/tracing"
)

// Default value for maximum turns in the agent loop
const DefaultMaxTurns = 10

var (
	ErrMaxTurnsExceeded         = errors.New("maximum turns exceeded")
	ErrModelProviderRequired    = errors.New("model provider is required")
	ErrGuardrailTripwire        = errors.New("guardrail tripwire triggered")
	ErrAgentMissingInstructions = errors.New("agent has no instructions")
	ErrInvalidHandoffInput      = errors.New("invalid handoff input")
	ErrInvalidOutputFormat      = errors.New("invalid output format")
)

var DefaultProvider model.Provider

// Message represents a chat message
type Message struct {
	// Role is the role of the message (system, user, assistant, tool)
	Role string

	// Content is the content of the message
	Content string

	ToolCalls []model.ToolCall

	ToolCallID string

	Name string
}

// Usage represents token usage
type Usage struct {
	// PromptTokens is the number of tokens in the prompt
	PromptTokens int

	// CompletionTokens is the number of tokens in the completion
	CompletionTokens int

	// TotalTokens is the total number of tokens
	TotalTokens int
}

// Result represents the result of an agent execution
type Result struct {
	// FinalOutput is the final output (string or JSON string)
	FinalOutput string

	// StructuredOutput is the structured output (based on agent's OutputType)
	StructuredOutput any

	// LastAgent is the last agent that was executed
	LastAgent *agent.Agent

	// History is the message history
	History []Message

	// Usage is the token usage
	Usage Usage
}

// RunConfig represents agent execution configuration
type RunConfig struct {
	// Model is the name of the model
	Model string

	ModelProvider model.Provider

	MaxTurns int

	// StepDelay is the delay between agent steps
	StepDelay time.Duration

	// HandoffCallback is a callback function that is called when a handoff occurs
	// It receives the current context, target agent, source agent, and handoff input JSON
	HandoffCallback func(ctx context.Context, targetAgent *agent.Agent, sourceAgent *agent.Agent, inputJSON string) error

	// HandoffInputFilter is a function that filters the input being passed to the target agent during handoff
	HandoffInputFilter handoff.InputFilter
}

// DefaultRunConfig returns the default execution configuration
func DefaultRunConfig() RunConfig {
	return RunConfig{
		Model:    "gpt-4o",
		MaxTurns: DefaultMaxTurns,
	}
}

// Run executes the agent
func Run(ctx context.Context, agent *agent.Agent, input string) (*Result, error) {
	return RunWithConfig(ctx, agent, input, DefaultRunConfig())
}

// RunWithConfig executes the agent with configuration
func RunWithConfig(ctx context.Context, a *agent.Agent, input string, config RunConfig) (*Result, error) {
	// Validate inputs and setup initial state
	if err := validateInputsAndSetup(a, &config); err != nil {
		return nil, fmt.Errorf("validation error: %w", err)
	}

	ctx, span := setupTracing(ctx, a, input, config)
	defer func() {
		if span != nil {
			span.End()
		}
	}()

	// Create execution state
	execState := &executionState{
		agent:            a,
		currentAgent:     a,
		originalInput:    input,
		config:           config,
		messages:         prepareMessages(a, input),
		resultMessages:   []model.Message{},
		usage:            Usage{},
		startTime:        time.Now(),
		ctx:              ctx,
		span:             span,
		stepCounter:      0,
		finalOutput:      "",
		structuredOutput: nil,
	}

	// Apply input guardrails
	if err := applyInputGuardrails(ctx, execState); err != nil {
		recordTracingError(ctx, execState.startTime, "", err)
		return nil, err
	}

	// Call agent start hook
	if err := a.Hooks.OnStart(ctx, a); err != nil {
		recordTracingError(ctx, execState.startTime, "", fmt.Errorf("error in OnStart hook: %w", err))
		return nil, fmt.Errorf("error in OnStart hook: %w", err)
	}

	// Add user input to history
	if input != "" {
		execState.resultMessages = append(execState.resultMessages, model.Message{
			Role:    "user",
			Content: input,
		})
	}

	// Run agent loop
	result, err := runAgentExecutionLoop(execState)
	if err != nil {
		// Special case for max turns exceeded
		if errors.Is(err, ErrMaxTurnsExceeded) {
			recordTracingError(ctx, execState.startTime, "", ErrMaxTurnsExceeded)
			return nil, ErrMaxTurnsExceeded
		}

		recordTracingError(ctx, execState.startTime, "", err)
		return nil, fmt.Errorf("agent execution error: %w", err)
	}

	// Set successful execution attributes in tracing
	if span != nil {
		span.SetAttribute("output", result.FinalOutput)
		span.SetAttribute("duration_ms", time.Since(execState.startTime).Milliseconds())
		span.SetAttribute("success", true)
		span.SetAttribute("turns_used", execState.stepCounter)

		if result.LastAgent != a {
			span.SetAttribute("final_agent", result.LastAgent.Name)
		}
	}

	return result, nil
}

// executionState tracks the state during agent execution
type executionState struct {
	agent            *agent.Agent
	currentAgent     *agent.Agent
	originalInput    string
	config           RunConfig
	messages         []model.Message
	resultMessages   []model.Message
	usage            Usage
	startTime        time.Time
	ctx              context.Context
	span             tracing.Span
	stepCounter      int
	finalOutput      string
	structuredOutput any
}

// setupTracing initializes tracing for agent execution
func setupTracing(ctx context.Context, a *agent.Agent, input string, config RunConfig) (context.Context, tracing.Span) {
	var span tracing.Span

	// Start agent execution root span if needed
	if tracing.GetActiveSpan(ctx) == nil {
		span, ctx = tracing.StartSpan(ctx, "agent_run", map[string]any{
			"span_type":  "agent",
			"agent_name": a.Name,
			"input":      input,
			"agent_id":   a.Name,
			"model":      config.Model,
			"max_turns":  config.MaxTurns,
		})
	} else {
		span = tracing.GetActiveSpan(ctx)
	}

	return ctx, span
}

// applyInputGuardrails executes all input guardrails
func applyInputGuardrails(ctx context.Context, state *executionState) error {
	if len(state.agent.InputGuardrails) == 0 {
		return nil
	}

	_, guardrailsCtx := tracing.StartSpan(ctx, "input_guardrails", map[string]any{
		"span_type":  "guardrails",
		"agent_name": state.agent.Name,
		"input":      state.originalInput,
	})
	defer func() {
		if span := tracing.GetActiveSpan(guardrailsCtx); span != nil {
			span.End()
		}
	}()

	for _, g := range state.agent.InputGuardrails {
		result, err := g.Check(guardrailsCtx, state.originalInput)
		if err != nil {
			return fmt.Errorf("input guardrail error: %w", err)
		}

		if !result.Allowed {
			if span := tracing.GetActiveSpan(guardrailsCtx); span != nil {
				span.SetAttribute("guardrail_triggered", true)
				span.SetAttribute("guardrail_message", result.Message)
			}
			return fmt.Errorf("%w: %s", ErrGuardrailTripwire, result.Message)
		}
	}

	return nil
}

// runAgentExecutionLoop runs the main agent loop
func runAgentExecutionLoop(state *executionState) (*Result, error) {
	// Run the main agent loop until we have a final result or exceed max turns
	for state.stepCounter < state.config.MaxTurns {
		// Apply step delay if needed
		if state.stepCounter > 0 && state.config.StepDelay > 0 {
			time.Sleep(state.config.StepDelay)
		}

		// Execute single step
		stepResult, err := runSingleTurn(state)
		if err != nil {
			return nil, fmt.Errorf("step %d error: %w", state.stepCounter+1, err)
		}

		// Process step result
		if err := processStepResult(state, stepResult); err != nil {
			return nil, err
		}

		// Check if we have a final output
		if state.finalOutput != "" {
			break
		}

		state.stepCounter++
	}

	// Check if max turns exceeded
	if state.finalOutput == "" {
		return nil, ErrMaxTurnsExceeded
	}

	// Create final result
	result := &Result{
		FinalOutput:      state.finalOutput,
		StructuredOutput: state.structuredOutput,
		LastAgent:        state.currentAgent,
		History:          convertModelMessages(state.resultMessages),
		Usage:            state.usage,
	}

	// Call agent end hook
	var finalOutputInterface any = state.finalOutput
	if state.structuredOutput != nil {
		finalOutputInterface = state.structuredOutput
	}

	if err := state.currentAgent.Hooks.OnEnd(state.ctx, state.currentAgent, finalOutputInterface); err != nil {
		return nil, fmt.Errorf("error in OnEnd hook: %w", err)
	}

	return result, nil
}

// runSingleTurn executes a single step/turn of the agent
func runSingleTurn(state *executionState) (*stepResult, error) {
	// Create step context with tracing
	stepCtx, stepSpan := createStepContext(state)
	defer func() {
		if stepSpan != nil {
			stepSpan.End()
		}
	}()

	// Get system prompt
	instructions, err := state.currentAgent.GetSystemPrompt(stepCtx)
	if err != nil {
		stepSpan.SetAttribute("error", err.Error())
		return nil, fmt.Errorf("failed to get instructions: %w", err)
	}

	// Update system message with current instructions if needed
	if len(state.messages) > 0 && state.messages[0].Role == "system" {
		state.messages[0].Content = instructions
	}

	// Process agent step (LLM call + tool execution)
	stepResult, err := processAgentStep(stepCtx, state)
	if err != nil {
		stepSpan.SetAttribute("error", err.Error())
		return nil, err
	}

	// Add tracing attributes for success
	stepSpan.SetAttribute("success", true)
	if stepResult.nextAgent != nil {
		stepSpan.SetAttribute("handoff_to", stepResult.nextAgent.Name)
	}
	if stepResult.finalOutput != "" {
		stepSpan.SetAttribute("has_final_output", true)
	}

	return stepResult, nil
}

// createStepContext creates a context for a single agent step with tracing
func createStepContext(state *executionState) (context.Context, tracing.Span) {
	messagesJSON, _ := json.Marshal(convertMessagesToTracingMessages(state.messages))

	span, stepCtx := tracing.StartSpan(state.ctx, fmt.Sprintf("agent_step_%d", state.stepCounter+1), map[string]any{
		"span_type":      "agent",
		"step":           state.stepCounter + 1,
		"messages":       string(messagesJSON),
		"agent_name":     state.currentAgent.Name,
		"messages_count": len(state.messages),
	})

	return stepCtx, span
}

// processAgentStep executes a full agent step including LLM call and tool handling
func processAgentStep(ctx context.Context, state *executionState) (*stepResult, error) {
	settings := model.DefaultSettings()
	modelName := state.config.Model

	// Prepare tools definitions
	settings.Tools = buildToolDefinitions(state.currentAgent)
	settings.Custom = make(map[string]any)

	// Use agent's model if specified
	if state.currentAgent.Model != "" {
		modelName = state.currentAgent.Model
	}
	settings.Custom["model"] = modelName

	// LLM call tracing
	_, llmCtx := tracing.StartSpan(ctx, "llm_call", map[string]any{
		"span_type": "agent",
		"model":     modelName,
		"agent":     state.currentAgent.Name,
	})

	// Call LLM
	response, err := state.config.ModelProvider.CreateChatCompletion(
		llmCtx,
		state.messages,
		settings,
	)

	// End LLM call tracing
	if span := tracing.GetActiveSpan(llmCtx); span != nil {
		if err != nil {
			span.SetAttribute("error", err.Error())
		} else {
			span.SetAttribute("success", true)
			span.SetAttribute("token_usage", response.Usage.TotalTokens)
		}
		span.End()
	}

	if err != nil {
		return nil, fmt.Errorf("LLM call failed: %w", err)
	}

	// Accumulate usage
	accumulateUsage(&state.usage, convertUsage(response.Usage))

	// Process response
	if len(response.Message.ToolCalls) > 0 {
		return processToolCallsAndHandoffs(ctx, state.currentAgent, response.Message)
	}

	// Process final output
	finalOutput := response.Message.Content
	var structuredOutput any

	// Try to parse as structured output if output type is defined
	if state.currentAgent.OutputType != nil {
		structValue := reflect.New(state.currentAgent.OutputType).Interface()
		if err := json.Unmarshal([]byte(finalOutput), structValue); err != nil {
			return nil, fmt.Errorf("%w: %s", ErrInvalidOutputFormat, err.Error())
		}
		// Get value from pointer
		structuredOutput = reflect.ValueOf(structValue).Elem().Interface()
	}

	// Apply output guardrails
	if len(state.currentAgent.OutputGuardrails) > 0 {
		checkedOutput, err := applyOutputGuardrails(ctx, state.currentAgent, finalOutput)
		if err != nil {
			return nil, err
		}
		finalOutput = checkedOutput
	}

	return &stepResult{
		finalOutput:      finalOutput,
		structuredOutput: structuredOutput,
		usage:            convertUsage(response.Usage),
		messages:         []model.Message{response.Message},
	}, nil
}

// buildToolDefinitions builds tool definitions for the agent
func buildToolDefinitions(a *agent.Agent) []map[string]any {
	toolDefs := make([]map[string]any, 0, len(a.Tools)+len(a.Handoffs))

	// Add regular tools
	for _, t := range a.Tools {
		toolDefs = append(toolDefs, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        t.Name(),
				"description": t.Description(),
				"parameters":  t.ParamsJSONSchema(),
			},
		})
	}

	// Add handoff tools
	for _, h := range a.Handoffs {
		toolDefs = append(toolDefs, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        h.ToolName(),
				"description": h.ToolDescription(),
				"parameters":  h.InputJSONSchema(),
			},
		})
	}

	return toolDefs
}

// processStepResult processes the result of a single step
func processStepResult(state *executionState, stepResult *stepResult) error {
	// Update messages with step result
	state.messages = append(state.messages, stepResult.messages...)
	state.resultMessages = append(state.resultMessages, stepResult.messages...)

	// Handle handoff if needed
	if stepResult.nextAgent != nil {
		if err := handleAgentHandoff(state, stepResult); err != nil {
			return err
		}
		return nil
	}

	// Handle final output
	if stepResult.finalOutput != "" {
		state.finalOutput = stepResult.finalOutput
		state.structuredOutput = stepResult.structuredOutput
		return nil
	}

	// Continue with updated state
	return nil
}

// handleAgentHandoff processes a handoff to another agent
func handleAgentHandoff(state *executionState, stepResult *stepResult) error {
	// Get handoff input directly from step result
	handoffInput := stepResult.handoffInput

	// Start handoff tracing
	handoffCtx, span := setupHandoffTracing(state, stepResult)
	defer func() {
		if span != nil {
			span.End()
		}
	}()

	// Process handoff callbacks and hooks
	if err := processHandoffCallbacks(handoffCtx, state, stepResult, handoffInput); err != nil {
		return err
	}

	// Apply input filter if provided
	if err := applyHandoffInputFilter(handoffCtx, state, stepResult, handoffInput); err != nil {
		return err
	}

	// Update messages and state for the new agent
	if err := updateMessagesForNewAgent(handoffCtx, state, stepResult); err != nil {
		return err
	}

	// Call agent start hook for new agent
	if err := state.currentAgent.Hooks.OnStart(handoffCtx, state.currentAgent); err != nil {
		return fmt.Errorf("error in OnStart hook for next agent: %w", err)
	}

	// Tracing: Handoff successful
	if span := tracing.GetActiveSpan(handoffCtx); span != nil {
		span.SetAttribute("success", true)
	}

	return nil
}

// setupHandoffTracing sets up tracing for agent handoff
func setupHandoffTracing(state *executionState, stepResult *stepResult) (context.Context, tracing.Span) {
	span, handoffCtx := tracing.StartSpan(state.ctx, "handoff", map[string]any{
		"span_type":          "handoff",
		"current_agent_name": state.currentAgent.Name,
		"next_agent_name":    stepResult.nextAgent.Name,
		"input":              stepResult.handoffInput,
	})

	return handoffCtx, span
}

// processHandoffCallbacks processes handoff callbacks and hooks
func processHandoffCallbacks(ctx context.Context, state *executionState, stepResult *stepResult, handoffInput string) error {
	// Find the handoff object from source agent's handoffs
	var targetHandoff handoff.Handoff
	for _, h := range state.currentAgent.Handoffs {
		if h.TargetAgent() == stepResult.nextAgent {
			targetHandoff = h
			break
		}
	}

	// Execute handoff callback if provided
	if targetHandoff != nil {
		if err := processTargetHandoff(ctx, targetHandoff, state, stepResult, handoffInput); err != nil {
			return err
		}
	}

	// Execute the config handoff callback if provided
	if state.config.HandoffCallback != nil {
		if err := state.config.HandoffCallback(ctx, stepResult.nextAgent, state.currentAgent, handoffInput); err != nil {
			if span := tracing.GetActiveSpan(ctx); span != nil {
				span.SetAttribute("error", err.Error())
			}
			return fmt.Errorf("handoff callback failed: %w", err)
		}
	}

	// Call agent's handoff hook
	if err := stepResult.nextAgent.Hooks.OnHandoff(ctx, stepResult.nextAgent, state.currentAgent); err != nil {
		if span := tracing.GetActiveSpan(ctx); span != nil {
			span.SetAttribute("error", err.Error())
		}
		return fmt.Errorf("error in OnHandoff hook: %w", err)
	}

	return nil
}

// processTargetHandoff processes the target handoff's OnHandoff method
func processTargetHandoff(ctx context.Context, targetHandoff handoff.Handoff, state *executionState, stepResult *stepResult, handoffInput string) error {
	// Prepare input data
	inputData := &handoff.InputData{
		InputHistory:    []map[string]any{},
		PreHandoffItems: []map[string]any{},
		NewItems:        []map[string]any{},
		Metadata: map[string]any{
			"handoff_input": handoffInput,
			"source_agent":  state.currentAgent.Name,
			"target_agent":  stepResult.nextAgent.Name,
		},
	}

	// Call the handoff's OnHandoff method
	if err := targetHandoff.OnHandoff(ctx, inputData, handoffInput); err != nil {
		if span := tracing.GetActiveSpan(ctx); span != nil {
			span.SetAttribute("error", err.Error())
		}
		return fmt.Errorf("handoff callback failed: %w", err)
	}

	return nil
}

// applyHandoffInputFilter applies the input filter if provided
func applyHandoffInputFilter(ctx context.Context, state *executionState, stepResult *stepResult, handoffInput string) error {
	if state.config.HandoffInputFilter != nil {
		inputData := &handoff.InputData{
			InputHistory:    []map[string]any{},
			PreHandoffItems: []map[string]any{},
			NewItems:        []map[string]any{},
			Metadata: map[string]any{
				"handoff_input": handoffInput,
				"source_agent":  state.currentAgent.Name,
				"target_agent":  stepResult.nextAgent.Name,
			},
		}

		filtered, err := state.config.HandoffInputFilter(ctx, inputData)
		if err != nil {
			return fmt.Errorf("handoff input filter failed: %w", err)
		}

		// Log filtered metadata
		if span := tracing.GetActiveSpan(ctx); span != nil && len(filtered.Metadata) > 0 {
			metadataJSON, _ := json.Marshal(filtered.Metadata)
			span.SetAttribute("filtered_metadata", string(metadataJSON))
		}
	}

	return nil
}

// updateMessagesForNewAgent updates the messages with the new agent's instructions
func updateMessagesForNewAgent(ctx context.Context, state *executionState, stepResult *stepResult) error {
	// Update messages (replace system message with new agent's)
	newInstructions, err := stepResult.nextAgent.GetSystemPrompt(ctx)
	if err != nil {
		return fmt.Errorf("failed to get instructions for next agent: %w", err)
	}

	// Create new messages with system message from target agent
	newMessages := []model.Message{{
		Role:    "system",
		Content: newInstructions,
	}}

	// Copy non-system messages
	for _, msg := range state.messages {
		if msg.Role != "system" {
			newMessages = append(newMessages, msg)
		}
	}

	// Update state
	state.currentAgent = stepResult.nextAgent
	state.messages = newMessages

	return nil
}

// applyOutputGuardrails applies output guardrails to the output
func applyOutputGuardrails(ctx context.Context, a *agent.Agent, output string) (string, error) {
	_, guardrailsCtx := tracing.StartSpan(ctx, "output_guardrails", map[string]any{
		"span_type":  "guardrails",
		"agent_name": a.Name,
		"output":     output,
	})

	modifiedOutput := output

	for _, g := range a.OutputGuardrails {
		result, err := g.Check(guardrailsCtx, modifiedOutput)
		if err != nil {
			if span := tracing.GetActiveSpan(guardrailsCtx); span != nil {
				span.SetAttribute("error", err.Error())
				span.End()
			}
			return "", fmt.Errorf("output guardrail error: %w", err)
		}

		if !result.Allowed {
			if span := tracing.GetActiveSpan(guardrailsCtx); span != nil {
				span.SetAttribute("guardrail_triggered", true)
				span.SetAttribute("guardrail_message", result.Message)
				span.End()
			}
			return "", fmt.Errorf("%w: %s", ErrGuardrailTripwire, result.Message)
		}

		if result.ModifiedOutput != "" {
			modifiedOutput = result.ModifiedOutput
		}
	}

	if span := tracing.GetActiveSpan(guardrailsCtx); span != nil {
		span.SetAttribute("success", true)
		if modifiedOutput != output {
			span.SetAttribute("output_modified", true)
		}
		span.End()
	}

	return modifiedOutput, nil
}

// validateInputsAndSetup validates the inputs and sets up default values
func validateInputsAndSetup(a *agent.Agent, config *RunConfig) error {
	if a.Instructions == "" {
		return ErrAgentMissingInstructions
	}

	if config.ModelProvider == nil {
		return ErrModelProviderRequired
	}

	if config.MaxTurns <= 0 {
		config.MaxTurns = DefaultMaxTurns
	}

	return nil
}

// recordTracingError records an error in the active span
func recordTracingError(ctx context.Context, startTime time.Time, output string, err error) {
	span := tracing.GetActiveSpan(ctx)
	if span != nil {
		span.SetAttribute("output", output)
		span.SetAttribute("duration_ms", time.Since(startTime).Milliseconds())
		span.SetAttribute("success", false)
		span.SetAttribute("error", err.Error())
		span.End()
	}
}

// RunSync executes the agent synchronously
func RunSync(agent *agent.Agent, input string) (*Result, error) {
	if DefaultProvider == nil {
		return nil, ErrModelProviderRequired
	}

	config := DefaultRunConfig()
	config.ModelProvider = DefaultProvider

	return RunWithConfig(context.Background(), agent, input, config)
}

// stepResult represents the result of an agent step
type stepResult struct {
	finalOutput      string
	structuredOutput any
	nextAgent        *agent.Agent
	messages         []model.Message
	usage            Usage
	handoffInput     string
}

// processToolCallsAndHandoffs processes tool calls and handoffs from LLM response
func processToolCallsAndHandoffs(ctx context.Context, a *agent.Agent, message model.Message) (*stepResult, error) {
	// Check if there are any tool calls
	if len(message.ToolCalls) == 0 {
		return nil, fmt.Errorf("no tool calls found in message")
	}

	// Start tool execution tracing
	_, toolsCtx := tracing.StartSpan(ctx, "tool_execution", map[string]any{
		"span_type":  "tools",
		"agent_name": a.Name,
		"tool_count": len(message.ToolCalls),
	})
	defer func() {
		if span := tracing.GetActiveSpan(toolsCtx); span != nil {
			span.End()
		}
	}()

	// Check for handoff tool call first
	for _, tc := range message.ToolCalls {
		// Check if this is a handoff tool call
		for _, h := range a.Handoffs {
			if h.ToolName() == tc.Function.Name {
				// Handle handoff
				shouldHandoff, err := h.ShouldHandoff(ctx, tc.Function.Arguments)
				if err != nil {
					return nil, fmt.Errorf("failed to check handoff: %w", err)
				}

				if shouldHandoff {
					// Handoff input JSON schema validation
					if schema := h.InputJSONSchema(); len(schema) > 0 {
						_, err := handoff.ValidateJSON(tc.Function.Arguments, schema)
						if err != nil {
							return nil, fmt.Errorf("%w: %s", ErrInvalidHandoffInput, err.Error())
						}
					}

					// Get target agent
					targetAgent := h.TargetAgent().(*agent.Agent)

					// Create step result with handoff information
					return &stepResult{
						nextAgent:    targetAgent,
						messages:     []model.Message{message},
						handoffInput: tc.Function.Arguments,
					}, nil
				}
			}
		}
	}

	// Execute regular tools
	toolResponses := []model.Message{}
	for _, tc := range message.ToolCalls {
		var toolResponse string
		var err error

		// Find matching tool
		var foundTool tool.Tool
		for _, t := range a.Tools {
			if t.Name() == tc.Function.Name {
				foundTool = t
				break
			}
		}

		if foundTool != nil {
			// Execute tool
			toolResponse, err = executeToolWithTracing(toolsCtx, a, foundTool, tc.Function.Arguments)
			if err != nil {
				return nil, fmt.Errorf("tool execution error: %w", err)
			}
		} else {
			toolResponse = fmt.Sprintf("Error: Tool '%s' not found", tc.Function.Name)
		}

		// Add tool response to messages
		toolResponses = append(toolResponses, model.Message{
			Role:       "tool",
			ToolCallID: tc.ID,
			Content:    toolResponse,
		})
	}

	// Return step result with tool responses
	return &stepResult{
		messages: append([]model.Message{message}, toolResponses...),
	}, nil
}

// executeToolWithTracing executes a tool with tracing
func executeToolWithTracing(ctx context.Context, a *agent.Agent, tool tool.Tool, args string) (string, error) {
	_, toolCtx := tracing.StartSpan(ctx, "tool_call", map[string]any{
		"span_type": "tool",
		"tool_name": tool.Name(),
		"tool_args": args,
	})
	defer func() {
		if span := tracing.GetActiveSpan(toolCtx); span != nil {
			span.End()
		}
	}()

	// Call tool start hook
	if err := a.Hooks.OnToolStart(ctx, a, tool); err != nil {
		return "", fmt.Errorf("error in OnToolStart hook: %w", err)
	}

	// Execute tool
	result, err := tool.Invoke(toolCtx, args)
	if err != nil {
		if span := tracing.GetActiveSpan(toolCtx); span != nil {
			span.SetAttribute("error", err.Error())
		}
		return "", fmt.Errorf("tool execution error: %w", err)
	}

	// Call tool end hook
	if err := a.Hooks.OnToolEnd(ctx, a, tool, result); err != nil {
		return "", fmt.Errorf("error in OnToolEnd hook: %w", err)
	}

	// Record successful execution in tracing
	if span := tracing.GetActiveSpan(toolCtx); span != nil {
		span.SetAttribute("success", true)
		responsePreview := result
		if len(responsePreview) > 100 {
			responsePreview = responsePreview[:100] + "..."
		}
		span.SetAttribute("response_preview", responsePreview)
	}

	return result, nil
}

// convertUsage converts from model.Usage to runner.Usage
func convertUsage(usage model.Usage) Usage {
	return Usage{
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		TotalTokens:      usage.TotalTokens,
	}
}

// convertModelMessages converts model messages to Runner messages
func convertModelMessages(messages []model.Message) []Message {
	result := make([]Message, len(messages))
	for i, msg := range messages {
		result[i] = Message{
			Role:       msg.Role,
			Content:    msg.Content,
			ToolCalls:  msg.ToolCalls,
			ToolCallID: msg.ToolCallID,
			Name:       msg.Name,
		}
	}
	return result
}

// prepareMessages prepares message history
func prepareMessages(agent *agent.Agent, input string) []model.Message {
	var messages []model.Message

	// Add system message
	if agent.Instructions != "" {
		messages = append(messages, model.Message{
			Role:    "system",
			Content: agent.Instructions,
		})
	}

	// Add user message
	if input != "" {
		messages = append(messages, model.Message{
			Role:    "user",
			Content: input,
		})
	}

	return messages
}

// convertMessagesToTracingMessages converts model messages to tracing messages
func convertMessagesToTracingMessages(messages []model.Message) []map[string]any {
	tracingMessages := make([]map[string]any, len(messages))
	for i, msg := range messages {
		tracingMessage := map[string]any{
			"role":    msg.Role,
			"content": msg.Content,
		}
		if msg.Name != "" {
			tracingMessage["name"] = msg.Name
		}
		if len(msg.ToolCalls) > 0 {
			toolCallsJSON, _ := json.Marshal(msg.ToolCalls)
			tracingMessage["tool_calls"] = string(toolCallsJSON)
		}
		if msg.ToolCallID != "" {
			tracingMessage["tool_call_id"] = msg.ToolCallID
		}
		tracingMessages[i] = tracingMessage
	}
	return tracingMessages
}

// accumulateUsage adds usage stats from a step to the total usage
func accumulateUsage(totalUsage *Usage, stepUsage Usage) {
	totalUsage.PromptTokens += stepUsage.PromptTokens
	totalUsage.CompletionTokens += stepUsage.CompletionTokens
	totalUsage.TotalTokens += stepUsage.TotalTokens
}
