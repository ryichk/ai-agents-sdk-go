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
		return nil, err
	}

	if err := checkInputGuardrails(ctx, a, input); err != nil {
		return nil, err
	}

	// Call agent start hook
	if err := a.Hooks.OnStart(ctx, a); err != nil {
		return nil, fmt.Errorf("error in OnStart hook: %w", err)
	}

	// Prepare initial messages
	messages := prepareMessages(a, input)

	// Setup tracing
	ctx, startTime := setupTracing(ctx, a, input)

	// Agent loop
	_, finalResult, err := runAgentLoop(ctx, a, messages, input, config, startTime)
	if err != nil {
		return nil, err
	}

	// Tracing: Record success
	span := tracing.GetActiveSpan(ctx)
	if span != nil {
		span.SetAttribute("output", finalResult.FinalOutput)
		span.SetAttribute("duration_ms", time.Since(startTime).Milliseconds())
		span.SetAttribute("success", true)
		span.End()
	}

	return finalResult, nil
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

// setupTracing initializes tracing for the agent run
func setupTracing(ctx context.Context, a *agent.Agent, input string) (context.Context, time.Time) {
	startTime := time.Now()

	// Start agent execution root span if needed
	if tracing.GetActiveSpan(ctx) == nil {
		_, ctx = tracing.StartSpan(ctx, "agent_run", map[string]any{
			"span_type":  "agent",
			"agent_name": a.Name,
			"input":      input,
		})
	}

	return ctx, startTime
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

// runAgentLoop executes the agent loop until completion or max turns
func runAgentLoop(ctx context.Context, a *agent.Agent, messages []model.Message, input string, config RunConfig, startTime time.Time) ([]model.Message, *Result, error) {
	var currentAgent = a
	var resultMessages []model.Message
	var usage Usage
	var finalOutput string
	var structuredOutput any
	var lastStepResult *stepResult

	// Add user input to history
	if input != "" {
		resultMessages = append(resultMessages, model.Message{
			Role:    "user",
			Content: input,
		})
	}

	for i := range config.MaxTurns {
		// Apply step delay if needed
		if i > 0 && config.StepDelay > 0 {
			time.Sleep(config.StepDelay)
		}

		// Create step context with tracing
		stepCtx := createStepContext(ctx, messages, i)

		// Process agent step
		stepResult, err := processAgentLoop(stepCtx, currentAgent, messages, config)
		if err != nil {
			recordTracingError(ctx, startTime, "", err)
			return nil, nil, err
		}

		// End step span
		if span := tracing.GetActiveSpan(stepCtx); span != nil {
			span.End()
		}

		lastStepResult = stepResult

		// Accumulate usage statistics
		accumulateUsage(&usage, stepResult.usage)

		// Process step result (handoff, final output, or continue)
		messages, resultMessages, currentAgent, finalOutput, structuredOutput, err =
			processStepResult(ctx, currentAgent, messages, resultMessages, stepResult, config, startTime)

		if err != nil {
			return nil, nil, err
		}

		// If we have final output, break the loop
		if finalOutput != "" {
			break
		}
	}

	// Check if max turns exceeded
	if finalOutput == "" && lastStepResult != nil {
		recordTracingError(ctx, startTime, "", ErrMaxTurnsExceeded)
		return nil, nil, ErrMaxTurnsExceeded
	}

	// Process final result
	result, err := processFinalResult(ctx, currentAgent, finalOutput, structuredOutput, resultMessages, usage)
	if err != nil {
		recordTracingError(ctx, startTime, finalOutput, err)
		return nil, nil, err
	}

	return resultMessages, result, nil
}

// createStepContext creates a context for a single agent step with tracing
func createStepContext(ctx context.Context, messages []model.Message, stepIndex int) context.Context {
	messagesJSON, _ := json.Marshal(convertMessagesToTracingMessages(messages))
	_, stepCtx := tracing.StartSpan(ctx, fmt.Sprintf("agent_step_%d", stepIndex+1), map[string]any{
		"span_type": "agent",
		"step":      stepIndex + 1,
		"messages":  string(messagesJSON),
	})
	return stepCtx
}

// accumulateUsage adds usage stats from a step to the total usage
func accumulateUsage(totalUsage *Usage, stepUsage Usage) {
	totalUsage.PromptTokens += stepUsage.PromptTokens
	totalUsage.CompletionTokens += stepUsage.CompletionTokens
	totalUsage.TotalTokens += stepUsage.TotalTokens
}

// processStepResult handles the result from a single agent step
func processStepResult(
	ctx context.Context,
	currentAgent *agent.Agent,
	messages []model.Message,
	resultMessages []model.Message,
	stepResult *stepResult,
	config RunConfig,
	startTime time.Time,
) ([]model.Message, []model.Message, *agent.Agent, string, any, error) {
	// Handle agent handoff if needed
	if stepResult.nextAgent != nil {
		newMessages, err := processNextAgent(ctx, currentAgent, stepResult.nextAgent, messages, stepResult, config)
		if err != nil {
			recordTracingError(ctx, startTime, "", err)
			return nil, nil, nil, "", nil, err
		}
		return newMessages, append(resultMessages, stepResult.messages...), stepResult.nextAgent, "", nil, nil
	}

	// Handle final output if present
	if stepResult.finalOutput != "" {
		checkedOutput, err := handleFinalOutput(ctx, currentAgent, stepResult.finalOutput)
		if err != nil {
			recordTracingError(ctx, startTime, "", err)
			return nil, nil, nil, "", nil, err
		}

		return messages, append(resultMessages, stepResult.messages...), currentAgent,
			checkedOutput, stepResult.structuredOutput, nil
	}

	// Continue with updated messages
	updatedMessages := append(messages, stepResult.messages...)
	updatedResultMessages := append(resultMessages, stepResult.messages...)

	return updatedMessages, updatedResultMessages, currentAgent, "", nil, nil
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

// agentStep executes one step of the agent loop
func agentStep(ctx context.Context, a *agent.Agent, messages []model.Message, config RunConfig) (*stepResult, error) {
	settings := model.DefaultSettings()
	modelName := config.Model
	toolDefs := make([]map[string]any, 0, len(a.Tools))
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
	settings.Tools = toolDefs
	settings.Custom = make(map[string]any)
	if a.Model != "" {
		modelName = a.Model
	}
	settings.Custom["model"] = modelName

	handoffDefinitions := make([]map[string]any, 0, len(a.Handoffs))
	for _, h := range a.Handoffs {
		handoffDefinitions = append(handoffDefinitions, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        h.ToolName(),
				"description": h.ToolDescription(),
				"parameters":  h.InputJSONSchema(),
			},
		})
	}

	if len(handoffDefinitions) > 0 {
		settings.Tools = append(settings.Tools, handoffDefinitions...)
	}

	response, err := config.ModelProvider.CreateChatCompletion(
		ctx,
		messages,
		settings,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to call model: %w", err)
	}

	if len(response.Message.ToolCalls) > 0 {
		return processToolCallsAndHandoffs(ctx, a, response.Message)
	}

	// Final output - Try to parse structured output
	finalOutput := response.Message.Content
	var structuredOutput any

	if a.OutputType != nil {
		// Try to parse output as structured data
		structValue := reflect.New(a.OutputType).Interface()
		if err := json.Unmarshal([]byte(finalOutput), structValue); err != nil {
			return nil, fmt.Errorf("%w: %s", ErrInvalidOutputFormat, err.Error())
		}
		// Get value from pointer
		structuredOutput = reflect.ValueOf(structValue).Elem().Interface()
	}

	return &stepResult{
		finalOutput:      finalOutput,
		structuredOutput: structuredOutput,
		usage:            convertUsage(response.Usage),
		messages:         []model.Message{response.Message},
	}, nil
}

// stepResult represents the result of an agent step
type stepResult struct {
	finalOutput      string
	structuredOutput any
	nextAgent        *agent.Agent
	messages         []model.Message
	usage            Usage
}

// processToolCallsAndHandoffs processes tool calls and handoffs
func processToolCallsAndHandoffs(ctx context.Context, a *agent.Agent, response model.Message) (*stepResult, error) {
	var nextAgent *agent.Agent
	var resultMessages []model.Message

	resultMessages = append(resultMessages, response)

	for _, toolCall := range response.ToolCalls {
		// Check if it's a handoff tool
		isHandoff := false
		var targetHandoff handoff.Handoff
		for _, h := range a.Handoffs {
			if h.ToolName() == toolCall.Function.Name {
				isHandoff = true
				targetHandoff = h
				break
			}
		}

		if isHandoff && targetHandoff != nil {
			// Handle handoff
			shouldHandoff, err := targetHandoff.ShouldHandoff(ctx, "")
			if err != nil {
				return nil, fmt.Errorf("failed to check handoff: %w", err)
			}
			if shouldHandoff {
				// Handoff input JSON schema validation
				if schema := targetHandoff.InputJSONSchema(); len(schema) > 0 {
					_, err := handoff.ValidateJSON(toolCall.Function.Arguments, schema)
					if err != nil {
						return nil, fmt.Errorf("%w: %s", ErrInvalidHandoffInput, err.Error())
					}
				}

				// Handoff callback execution
				inputData := &handoff.InputData{
					InputHistory:    []map[string]any{},
					PreHandoffItems: []map[string]any{},
					NewItems:        []map[string]any{},
					Metadata:        map[string]any{},
				}
				if err := targetHandoff.OnHandoff(ctx, inputData, toolCall.Function.Arguments); err != nil {
					return nil, fmt.Errorf("handoff callback failed: %w", err)
				}

				// Set target agent
				nextAgent = targetHandoff.TargetAgent().(*agent.Agent)
			}
		} else {
			// Execute tool
			toolResult, err := executeToolCall(ctx, a, toolCall)
			if err != nil {
				return nil, fmt.Errorf("failed to execute tool: %w", err)
			}

			// Add tool result to messages
			resultMessages = append(resultMessages, model.Message{
				Role:       "tool",
				Content:    toolResult,
				ToolCallID: toolCall.ID,
			})
		}
	}

	return &stepResult{
		nextAgent: nextAgent,
		messages:  resultMessages,
		usage:     Usage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150},
	}, nil
}

// executeToolCall executes a tool call
func executeToolCall(ctx context.Context, a *agent.Agent, toolCall model.ToolCall) (string, error) {
	// Find tool
	var selectedTool tool.Tool
	for _, t := range a.Tools {
		if t.Name() == toolCall.Function.Name {
			selectedTool = t
			break
		}
	}

	if selectedTool == nil {
		return "", fmt.Errorf("tool '%s' not found", toolCall.Function.Name)
	}

	// Call tool start hook
	if err := a.Hooks.OnToolStart(ctx, a, selectedTool); err != nil {
		return "", fmt.Errorf("error in OnToolStart hook: %w", err)
	}

	// Tracing: Start tool call
	_, toolCtx := tracing.StartSpan(ctx, "tool_call", map[string]any{
		"span_type":    "tool",
		"tool_name":    selectedTool.Name(),
		"tool_call_id": toolCall.ID,
		"arguments":    toolCall.Function.Arguments,
	})

	// Execute tool
	result, err := selectedTool.Invoke(toolCtx, toolCall.Function.Arguments)

	if err != nil {
		span := tracing.GetActiveSpan(toolCtx)
		if span != nil {
			span.SetAttribute("error", err.Error())
			span.End()
		}
		return "", fmt.Errorf("failed to invoke tool '%s': %w", toolCall.Function.Name, err)
	}

	// Tracing: Tool call completed
	span := tracing.GetActiveSpan(toolCtx)
	if span != nil {
		span.SetAttribute("result", result)
		span.End()
	}

	// Call tool end hook
	if err := a.Hooks.OnToolEnd(ctx, a, selectedTool, result); err != nil {
		return "", fmt.Errorf("error in OnToolEnd hook: %w", err)
	}

	return result, nil
}

// convertUsage converts model usage to Runner usage
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

// checkInputGuardrails checks input guardrails
func checkInputGuardrails(ctx context.Context, agent *agent.Agent, input string) error {
	for _, g := range agent.InputGuardrails {
		result, err := g.Check(ctx, input)
		if err != nil {
			return fmt.Errorf("failed to check input guardrail: %w", err)
		}
		if !result.Allowed {
			return fmt.Errorf("%w: %s", ErrGuardrailTripwire, result.Message)
		}
	}
	return nil
}

// checkOutputGuardrails checks output guardrails
func checkOutputGuardrails(ctx context.Context, agent *agent.Agent, output string) (string, error) {
	modifiedOutput := output

	for _, g := range agent.OutputGuardrails {
		result, err := g.Check(ctx, modifiedOutput)
		if err != nil {
			return "", fmt.Errorf("failed to check output guardrail: %w", err)
		}
		if !result.Allowed {
			return "", fmt.Errorf("%w: %s", ErrGuardrailTripwire, result.Message)
		}
		if result.ModifiedOutput != "" {
			modifiedOutput = result.ModifiedOutput
		}
	}

	return modifiedOutput, nil
}

// handleHandoff processes handoff
func handleHandoff(ctx context.Context, currentAgent *agent.Agent, nextAgent *agent.Agent, messages []model.Message, stepResult *stepResult) ([]model.Message, error) {
	// Tracing: Record handoff
	_, handoffCtx := tracing.StartSpan(ctx, "handoff", map[string]any{
		"span_type":          "handoff",
		"current_agent_name": currentAgent.Name,
		"next_agent_name":    nextAgent.Name,
		"reason":             "Handoff requested",
	})

	// Call handoff hook
	if err := nextAgent.Hooks.OnHandoff(ctx, nextAgent, currentAgent); err != nil {
		span := tracing.GetActiveSpan(handoffCtx)
		if span != nil {
			span.SetAttribute("success", false)
			span.SetAttribute("error", err.Error())
			span.End()
		}
		return nil, fmt.Errorf("error in OnHandoff hook: %w", err)
	}

	// Update messages (replace system message with new agent's)
	newMessages := prepareMessages(nextAgent, "")
	// Add messages other than system messages
	for _, msg := range messages {
		if msg.Role != "system" {
			newMessages = append(newMessages, msg)
		}
	}
	// Add latest message
	newMessages = append(newMessages, stepResult.messages...)

	// Tracing: Handoff completed
	span := tracing.GetActiveSpan(handoffCtx)
	if span != nil {
		span.SetAttribute("success", true)
		span.End()
	}

	return newMessages, nil
}

func handleFinalOutput(ctx context.Context, currentAgent *agent.Agent, output string) (string, error) {
	checkedOutput, err := checkOutputGuardrails(ctx, currentAgent, output)
	if err != nil {
		return "", err
	}
	return checkedOutput, nil
}

func processAgentLoop(ctx context.Context, currentAgent *agent.Agent, messages []model.Message, config RunConfig) (*stepResult, error) {
	// Execute agent step
	stepResult, err := agentStep(ctx, currentAgent, messages, config)
	if err != nil {
		return nil, err
	}

	return stepResult, nil
}

// processNextAgent processes moving to the next agent
func processNextAgent(ctx context.Context, currentAgent *agent.Agent, nextAgent *agent.Agent, messages []model.Message, stepResult *stepResult, config RunConfig) ([]model.Message, error) {
	// Get the handoff tool call if any
	var handoffInput string
	if len(stepResult.messages) > 0 && len(stepResult.messages[0].ToolCalls) > 0 {
		// Assuming the first tool call is the handoff
		handoffInput = stepResult.messages[0].ToolCalls[0].Function.Arguments
	}

	// Execute handoff callback if provided
	if config.HandoffCallback != nil {
		if err := config.HandoffCallback(ctx, nextAgent, currentAgent, handoffInput); err != nil {
			return nil, fmt.Errorf("handoff callback failed: %w", err)
		}
	}

	newMessages, err := handleHandoff(ctx, currentAgent, nextAgent, messages, stepResult)
	if err != nil {
		return nil, err
	}

	// Apply input filter if provided
	if config.HandoffInputFilter != nil {
		// Create input data structure for the filter
		inputData := &handoff.InputData{
			InputHistory:    []map[string]any{},
			PreHandoffItems: []map[string]any{},
			NewItems:        []map[string]any{},
			Metadata: map[string]any{
				"handoff_input": handoffInput,
				"source_agent":  currentAgent.Name,
				"target_agent":  nextAgent.Name,
			},
		}

		filtered, err := config.HandoffInputFilter(ctx, inputData)
		if err != nil {
			return nil, fmt.Errorf("handoff input filter failed: %w", err)
		}

		// Apply filtered data to messages (simplified implementation)
		// In a real implementation, you would need to convert the filtered data back to messages
		if len(filtered.Metadata) > 0 {
			// Just as an example, you might add metadata as a system message
			metadataJSON, _ := json.Marshal(filtered.Metadata)
			newMessages = append([]model.Message{{
				Role:    "system",
				Content: fmt.Sprintf("Handoff metadata: %s", string(metadataJSON)),
			}}, newMessages...)
		}
	}

	// Call agent start hook for new agent
	if err := nextAgent.Hooks.OnStart(ctx, nextAgent); err != nil {
		return nil, fmt.Errorf("error in OnStart hook: %w", err)
	}

	return newMessages, nil
}

// processFinalResult processes final result
func processFinalResult(ctx context.Context, currentAgent *agent.Agent, finalOutput string, structuredOutput any, resultMessages []model.Message, usage Usage) (*Result, error) {
	// Call agent end hook
	var finalOutputInterface any = finalOutput
	if structuredOutput != nil {
		finalOutputInterface = structuredOutput
	}
	if err := currentAgent.Hooks.OnEnd(ctx, currentAgent, finalOutputInterface); err != nil {
		return nil, fmt.Errorf("error in OnEnd hook: %w", err)
	}

	return &Result{
		FinalOutput:      finalOutput,
		StructuredOutput: structuredOutput,
		LastAgent:        currentAgent,
		History:          convertModelMessages(resultMessages),
		Usage:            usage,
	}, nil
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
