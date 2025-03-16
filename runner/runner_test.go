package runner

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/ryichk/ai-agents-sdk-go/agent"
	"github.com/ryichk/ai-agents-sdk-go/guardrail"
	"github.com/ryichk/ai-agents-sdk-go/handoff"
	"github.com/ryichk/ai-agents-sdk-go/model"
)

type TestOutputStruct struct {
	Bar string `json:"bar"`
}

func TestSimpleFirstRun(t *testing.T) {
	ctx := context.Background()

	fakeModel := NewFakeModel()
	fakeModel.SetNextOutput([]model.Message{GetTextMessage("first")})

	testAgent := agent.New("test", "test instructions")

	config := RunConfig{
		Model:         "gpt-4o",
		ModelProvider: fakeModel,
		MaxTurns:      10,
	}

	result, err := RunWithConfig(ctx, testAgent, "test", config)
	assert.NoError(t, err, "Agent execution should not return an error")

	assert.Equal(t, "first", result.FinalOutput, "Final output does not match")
	assert.Len(t, result.History, 2, "Should generate 2 history items")
	assert.Equal(t, 150, result.Usage.TotalTokens, "Total token count does not match")

	fakeModel.SetNextOutput([]model.Message{GetTextMessage("second")})

	result, err = RunWithConfig(ctx, testAgent, "another message", config)
	assert.NoError(t, err, "Second execution should not return an error")
	assert.Equal(t, "second", result.FinalOutput, "Second final output does not match")
	assert.Len(t, result.History, 2, "Second execution should generate 2 history items")
}

func TestToolCallRuns(t *testing.T) {
	ctx := context.Background()

	fakeModel := NewFakeModel()

	testTool := NewFunctionTool("foo", "tool_result")

	testAgent := agent.New("test", "test instructions")
	testAgent.AddTool(testTool)

	// Set up outputs for multiple turns
	fakeModel.AddMultipleTurnOutputs([][]model.Message{
		// 1st turn: Tool call
		{GetTextMessage("a_message"), GetFunctionToolCall("foo", `{"a":"b"}`)},
		// 2nd turn: Text message
		{GetTextMessage("done")},
	})

	config := RunConfig{
		Model:         "gpt-4o",
		ModelProvider: fakeModel,
		MaxTurns:      10,
	}

	result, err := RunWithConfig(ctx, testAgent, "user_message", config)
	assert.NoError(t, err, "Agent execution should not return an error")

	assert.Equal(t, "done", result.FinalOutput, "Final output does not match")
	assert.Equal(t, 300, result.Usage.TotalTokens, "Total token count should be 300 after 2 calls")

	// History should contain 4 items
	// 1. User input
	// 2. First message
	// 3. Tool call
	// 4. Tool result
	// 5. Final message
	assert.Equal(t, 4, len(result.History), "Should generate 4 history items")

	if len(result.History) >= 4 {
		assert.Equal(t, "user", result.History[0].Role, "First item should be a user message")
		assert.Equal(t, "assistant", result.History[1].Role, "Second item should be an assistant message")
		assert.Equal(t, "tool", result.History[2].Role, "Third item should be a tool result")
		assert.Equal(t, "assistant", result.History[3].Role, "Fourth item should be the final message")
	}
}

func TestHandoffs(t *testing.T) {
	ctx := context.Background()

	fakeModel := NewFakeModel()

	testTool := NewFunctionTool("some_function", "result")

	agent1 := agent.New("agent1", "agent1 instructions")
	agent2 := agent.New("agent2", "agent2 instructions")
	agent3 := agent.New("agent3", "agent3 instructions")

	handoff1 := handoff.NewHandoff(agent1, "Handoff to agent1")
	handoff2 := handoff.NewHandoff(agent2, "Handoff to agent2")
	agent3.AddHandoffs(handoff1, handoff2)
	agent3.AddTool(testTool)

	// Set up outputs for multiple turns
	fakeModel.AddMultipleTurnOutputs([][]model.Message{
		// 1st turn: Tool call
		{GetFunctionToolCall("some_function", `{"a":"b"}`)},
		// 2nd turn: Message and handoff
		{GetTextMessage("a_message"), model.Message{
			Role:    "assistant",
			Content: "",
			ToolCalls: []model.ToolCall{
				{
					ID:   "handoff_call",
					Type: "function",
					Function: model.FunctionCall{
						Name:      handoff1.ToolName(),
						Arguments: "{}",
					},
				},
			},
		}},
		// 3rd turn: Text message
		{GetTextMessage("done")},
	})

	config := RunConfig{
		Model:         "gpt-4o",
		ModelProvider: fakeModel,
		MaxTurns:      10,
	}

	result, err := RunWithConfig(ctx, agent3, "user_message", config)
	assert.NoError(t, err, "Agent execution should not return an error")

	assert.Equal(t, "done", result.FinalOutput, "Final output does not match")
	assert.Equal(t, 450, result.Usage.TotalTokens, "Total token count should be 450 after 3 calls")
}

func TestStructuredOutput(t *testing.T) {
	ctx := context.Background()

	fakeModel := NewFakeModel()

	barTool := NewFunctionTool("bar", "bar_result")
	fooTool := NewFunctionTool("foo", "foo_result")

	agent1 := agent.New("agent1", "agent1 instructions")
	agent1.AddTool(barTool)
	agent1.SetOutputType(reflect.TypeOf(TestOutputStruct{}))

	agent2 := agent.New("agent2", "agent2 instructions")
	agent2.AddTool(fooTool)

	handoff1 := handoff.NewHandoff(agent1, "Handoff to agent1")
	agent2.AddHandoff(handoff1)

	// Set up outputs for multiple turns
	fakeModel.AddMultipleTurnOutputs([][]model.Message{
		// 1st turn: Tool call
		{GetFunctionToolCall("foo", `{"bar":"baz"}`)},
		// 2nd turn: Message and handoff
		{GetTextMessage("a_message"), model.Message{
			Role:    "assistant",
			Content: "",
			ToolCalls: []model.ToolCall{
				{
					ID:   "handoff_call",
					Type: "function",
					Function: model.FunctionCall{
						Name:      handoff1.ToolName(),
						Arguments: "{}",
					},
				},
			},
		}},
		// 3rd turn: Tool call
		{GetFunctionToolCall("bar", `{"bar":"baz"}`)},
		// 4th turn: Final output
		{GetTextMessage(`{"bar":"baz"}`)},
	})

	config := RunConfig{
		Model:         "gpt-4o",
		ModelProvider: fakeModel,
		MaxTurns:      10,
	}

	result, err := RunWithConfig(ctx, agent2, "user_message", config)
	assert.NoError(t, err, "Agent execution should not return an error")

	assert.Equal(t, `{"bar":"baz"}`, result.FinalOutput, "Final output does not match")
	assert.Equal(t, 600, result.Usage.TotalTokens, "Total token count should be 600 after 4 calls")
}

func TestHandoffFilters(t *testing.T) {
	ctx := context.Background()

	fakeModel := NewFakeModel()

	filterFunc := func(ctx context.Context, inputData *handoff.InputData) (*handoff.InputData, error) {
		// delete new items
		inputData.NewItems = nil
		return inputData, nil
	}

	agent1 := agent.New("agent1", "agent1 instructions")
	agent2 := agent.New("agent2", "agent2 instructions")

	h := handoff.NewHandoff(agent1, "Test handoff")
	filteredHandoff := handoff.NewFilteredHandoff(h, filterFunc)
	agent2.AddHandoff(filteredHandoff)

	// Set up outputs for multiple turns
	fakeModel.AddMultipleTurnOutputs([][]model.Message{
		// 1st turn: Message, additional message, and handoff
		{
			GetTextMessage("1"),
			GetTextMessage("2"),
			model.Message{
				Role:    "assistant",
				Content: "",
				ToolCalls: []model.ToolCall{
					{
						ID:   "handoff_call",
						Type: "function",
						Function: model.FunctionCall{
							Name:      filteredHandoff.ToolName(),
							Arguments: "{}",
						},
					},
				},
			},
		},
		// 2nd turn: Final message
		{GetTextMessage("last")},
	})

	config := RunConfig{
		Model:         "gpt-4o",
		ModelProvider: fakeModel,
		MaxTurns:      10,
	}

	result, err := RunWithConfig(ctx, agent2, "user_message", config)
	assert.NoError(t, err, "Agent execution should not return an error")

	assert.Equal(t, "last", result.FinalOutput, "Final output does not match")
	assert.Len(t, result.History, 3, "Filtered history should have 3 items")
}

func TestGuardrailTripwire(t *testing.T) {
	ctx := context.Background()

	fakeModel := NewFakeModel()
	fakeModel.SetNextOutput([]model.Message{GetTextMessage("response")})

	inputGuardrail := guardrail.NewInputGuardrail(
		"input_blocker",
		"Block specific inputs",
		func(ctx context.Context, input string) (guardrail.InputGuardrailResult, error) {
			return guardrail.InputGuardrailResult{
				Allowed: false,
				Message: "Input not allowed",
			}, nil
		},
	)

	outputGuardrail := guardrail.NewOutputGuardrail(
		"output_blocker",
		"Block specific outputs",
		func(ctx context.Context, output string) (guardrail.OutputGuardrailResult, error) {
			return guardrail.OutputGuardrailResult{
				Allowed: false,
				Message: "Output not allowed",
			}, nil
		},
	)

	// Agent with input guardrail only
	inputGuardedAgent := agent.New("input_guarded", "test instructions")
	inputGuardedAgent.AddInputGuardrail(inputGuardrail)

	// Agent with output guardrail only
	outputGuardedAgent := agent.New("output_guarded", "test instructions")
	outputGuardedAgent.AddOutputGuardrail(outputGuardrail)

	config := RunConfig{
		Model:         "gpt-4o",
		ModelProvider: fakeModel,
		MaxTurns:      10,
	}

	// Input guardrail test
	_, err := RunWithConfig(ctx, inputGuardedAgent, "bad input", config)
	assert.Error(t, err, "Input guardrail should trip")
	assert.Contains(t, err.Error(), "Input not allowed", "Error message does not match")

	// Output guardrail test
	_, err = RunWithConfig(ctx, outputGuardedAgent, "test input", config)
	assert.Error(t, err, "Output guardrail should trip")
	assert.Contains(t, err.Error(), "Output not allowed", "Error message does not match")
}

func TestMaxTurnsExceeded(t *testing.T) {
	ctx := context.Background()

	fakeModel := NewFakeModel()

	// Set up model output that only returns tool calls
	toolCallOutput := []model.Message{GetFunctionToolCall("test_tool", "{}")}
	outputs := make([][]model.Message, 12) // More than the maximum number of turns
	for i := range 12 {
		outputs[i] = toolCallOutput
	}
	fakeModel.AddMultipleTurnOutputs(outputs)

	testTool := NewFunctionTool("test_tool", "result")

	testAgent := agent.New("test", "test instructions")
	testAgent.AddTool(testTool)

	// Set maximum number of turns
	config := RunConfig{
		Model:         "gpt-4o",
		ModelProvider: fakeModel,
		MaxTurns:      5,
	}

	_, err := RunWithConfig(ctx, testAgent, "test input", config)
	assert.Error(t, err, "Max turns exceeded should cause error")
	assert.Equal(t, ErrMaxTurnsExceeded, err, "Error type does not match")
}

func TestHandoffOnInput(t *testing.T) {
	ctx := context.Background()

	fakeModel := NewFakeModel()

	var callOutput string

	callback := func(ctx context.Context, inputData *handoff.InputData, inputJSON string) error {
		var data map[string]any
		if err := json.Unmarshal([]byte(inputJSON), &data); err != nil {
			t.Fatalf("failed to unmarshal inputJSON: %v", err)
		}
		callOutput = data["bar"].(string)
		return nil
	}

	agent1 := agent.New("agent1", "agent1 instructions")
	agent2 := agent.New("agent2", "agent2 instructions")

	// Create JSON schema for handoff
	schema := handoff.JSONSchema{
		"type": "object",
		"properties": map[string]any{
			"bar": map[string]any{
				"type": "string",
			},
		},
		"required": []string{"bar"},
	}

	// Create handoff with callback
	handoff1 := handoff.NewHandoffWithOptions(agent1, "Test handoff", handoff.Options{
		InputJSONSchema: schema,
		OnHandoff:       callback,
	})
	agent2.AddHandoff(handoff1)

	// Set up outputs for multiple turns
	fakeModel.AddMultipleTurnOutputs([][]model.Message{
		// 1st turn: Message and handoff (with arguments)
		{
			GetTextMessage("1"),
			GetTextMessage("2"),
			model.Message{
				Role:    "assistant",
				Content: "",
				ToolCalls: []model.ToolCall{
					{
						ID:   "handoff_call",
						Type: "function",
						Function: model.FunctionCall{
							Name:      handoff1.ToolName(),
							Arguments: `{"bar":"test_input"}`,
						},
					},
				},
			},
		},
		// 2nd turn: Final message
		{GetTextMessage("last")},
	})

	config := RunConfig{
		Model:         "gpt-4o",
		ModelProvider: fakeModel,
		MaxTurns:      10,
	}

	result, err := RunWithConfig(ctx, agent2, "user_message", config)
	assert.NoError(t, err, "Agent execution should not return an error")

	assert.Equal(t, "last", result.FinalOutput, "Final output does not match")
	assert.Equal(t, "test_input", callOutput, "Callback should be called with correct input")
}

func TestInvalidHandoffInputJSON(t *testing.T) {
	ctx := context.Background()

	fakeModel := NewFakeModel()

	agent1 := agent.New("agent1", "agent1 instructions")
	agent2 := agent.New("agent2", "agent2 instructions")

	// Create JSON schema for handoff (with required fields)
	schema := handoff.JSONSchema{
		"type": "object",
		"properties": map[string]any{
			"reason": map[string]any{
				"type": "string",
			},
		},
		"required": []string{"reason"},
	}

	// Create handoff with schema
	handoff1 := handoff.NewHandoffWithOptions(agent1, "Test handoff", handoff.Options{
		InputJSONSchema: schema,
	})
	agent2.AddHandoff(handoff1)

	// Handoff with invalid JSON
	fakeModel.AddMultipleTurnOutputs([][]model.Message{
		// Handoff with invalid JSON missing required fields
		{model.Message{
			Role:    "assistant",
			Content: "",
			ToolCalls: []model.ToolCall{
				{
					ID:   "handoff_call",
					Type: "function",
					Function: model.FunctionCall{
						Name:      handoff1.ToolName(),
						Arguments: `{"priority": 1}`,
					},
				},
			},
		}},
	})

	config := RunConfig{
		Model:         "gpt-4o",
		ModelProvider: fakeModel,
		MaxTurns:      10,
	}

	_, err := RunWithConfig(ctx, agent2, "user_message", config)
	assert.Error(t, err, "Invalid JSON input should cause error")
	assert.Contains(t, err.Error(), "missing required field", "Error message should indicate missing required field")
}

func TestRunSync(t *testing.T) {
	fakeModel := NewFakeModel()
	fakeModel.SetNextOutput([]model.Message{GetTextMessage("sync response")})

	testAgent := agent.New("test", "test instructions")

	// Save default provider before test
	oldDefaultProvider := DefaultProvider
	defer func() { DefaultProvider = oldDefaultProvider }()

	// Temporarily set DefaultProvider
	DefaultProvider = fakeModel

	// Synchronous execution
	result, err := RunSync(testAgent, "test input")
	assert.NoError(t, err, "RunSync should not return an error")
	assert.Equal(t, "sync response", result.FinalOutput, "RunSync output does not match")
}

func TestRunWithDelay(t *testing.T) {
	ctx := context.Background()

	fakeModel := NewFakeModel()
	fakeModel.SetNextOutput([]model.Message{GetTextMessage("first")})
	fakeModel.SetNextOutput([]model.Message{GetTextMessage("delayed response")})

	testAgent := agent.New("test", "test instructions")

	// Agent execution configuration (with delay)
	config := RunConfig{
		Model:         "gpt-4o",
		ModelProvider: fakeModel,
		MaxTurns:      10,
		StepDelay:     time.Millisecond * 100, // 100 millisecond delay
	}

	// Record execution start time
	startTime := time.Now()

	// Set up outputs for multiple turns and test delay between steps
	fakeModel.AddMultipleTurnOutputs([][]model.Message{
		{GetFunctionToolCall("dummy", "{}")}, // 1st turn
		{GetTextMessage("delayed response")}, // 2nd turn (after delay)
	})

	// Add dummy tool to increase the number of turns
	testAgent.AddTool(NewFunctionTool("dummy", "dummy_result"))

	// Execute agent
	result, err := RunWithConfig(ctx, testAgent, "test input", config)

	// Calculate execution time
	duration := time.Since(startTime)

	assert.NoError(t, err, "Agent execution should not return an error")
	assert.Equal(t, "delayed response", result.FinalOutput, "Final output does not match")
	assert.True(t, duration >= time.Millisecond*100, "Should have at least 100ms delay")
}
