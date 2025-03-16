package runner

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/ryichk/ai-agents-sdk-go/agent"
	"github.com/ryichk/ai-agents-sdk-go/model"
	"github.com/ryichk/ai-agents-sdk-go/tool"
)

type FakeModel struct {
	nextOutput       []model.Message
	multiTurnOutputs [][]model.Message
	currentTurn      int
	history          []model.Message
	response         string
	shouldError      bool
}

func NewFakeModel() *FakeModel {
	return &FakeModel{
		nextOutput:       nil,
		multiTurnOutputs: nil,
		currentTurn:      0,
		history:          []model.Message{},
	}
}

func (m *FakeModel) SetNextOutput(output []model.Message) {
	m.nextOutput = output
}

func (m *FakeModel) AddMultipleTurnOutputs(outputs [][]model.Message) {
	m.multiTurnOutputs = outputs
	m.currentTurn = 0
}

func (m *FakeModel) CreateChatCompletion(ctx context.Context, messages []model.Message, settings model.Settings) (*model.Response, error) {
	m.history = append(m.history, messages...)

	// Select response
	var output []model.Message
	if m.multiTurnOutputs != nil && m.currentTurn < len(m.multiTurnOutputs) {
		output = m.multiTurnOutputs[m.currentTurn]
		m.currentTurn++
	} else if m.nextOutput != nil {
		output = m.nextOutput
		m.nextOutput = nil
	} else {
		output = []model.Message{{Role: "assistant", Content: "default response"}}
	}

	// Generate simulated token usage
	usage := model.Usage{
		PromptTokens:     100,
		CompletionTokens: 50,
		TotalTokens:      150,
	}

	// Process tool calls
	toolCalls := make([]model.ToolCall, 0)
	for _, msg := range output {
		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			toolCalls = append(toolCalls, msg.ToolCalls...)
		}
	}

	// Combine all messages into a single response message
	responseText := combineMessages(output)

	// Include history in response message
	resultMessage := model.Message{
		Role:      "assistant",
		Content:   responseText,
		ToolCalls: toolCalls,
	}

	// Add result message to history
	m.history = append(m.history, resultMessage)

	return &model.Response{
		Message: resultMessage,
		Usage:   usage,
	}, nil
}

func (m *FakeModel) CreateChatCompletionStream(ctx context.Context, messages []model.Message, settings model.Settings) (model.Stream, error) {
	if m.shouldError {
		return nil, fmt.Errorf("fake error")
	}

	response := &model.Response{
		Message: model.Message{
			Role:    "assistant",
			Content: m.response,
		},
	}

	return &FakeStream{
		response: response,
		chunks: []*model.StreamChunk{
			{
				Delta: model.Message{
					Role:    "assistant",
					Content: m.response,
				},
				FinishReason: "stop",
			},
		},
	}, nil
}

type FakeStream struct {
	response *model.Response
	index    int
	chunks   []*model.StreamChunk
}

// Recv receives the next chunk from the stream
func (s *FakeStream) Recv() (*model.StreamChunk, error) {
	if s.index >= len(s.chunks) {
		return nil, io.EOF
	}
	chunk := s.chunks[s.index]
	s.index++
	return chunk, nil
}

func (s *FakeStream) Close() error {
	return nil
}

// combineMessages combines multiple messages into a single text response
func combineMessages(messages []model.Message) string {
	var contents []string
	for _, msg := range messages {
		if msg.Role == "assistant" && msg.Content != "" {
			contents = append(contents, msg.Content)
		}
	}
	return strings.Join(contents, " ")
}

func GetTextMessage(content string) model.Message {
	return model.Message{
		Role:    "assistant",
		Content: content,
	}
}

func GetTextInputItem(content string) model.Message {
	return model.Message{
		Role:    "user",
		Content: content,
	}
}

func GetFunctionToolCall(name string, arguments string) model.Message {
	return model.Message{
		Role:    "assistant",
		Content: "",
		ToolCalls: []model.ToolCall{
			{
				ID:   "call_" + name,
				Type: "function",
				Function: model.FunctionCall{
					Name:      name,
					Arguments: arguments,
				},
			},
		},
	}
}

func GetHandoffToolCall(targetAgent *agent.Agent, inputJSON string) model.Message {
	if inputJSON == "" {
		inputJSON = "{}"
	}

	agentName := targetAgent.Name

	return model.Message{
		Role:    "assistant",
		Content: "",
		ToolCalls: []model.ToolCall{
			{
				ID:   "handoff_" + agentName,
				Type: "function",
				Function: model.FunctionCall{
					Name:      fmt.Sprintf("handoff_%s", agentName),
					Arguments: inputJSON,
				},
			},
		},
	}
}

func NewFunctionTool(name string, result string) tool.Tool {
	return &FunctionTool{
		name:   name,
		result: result,
	}
}

type FunctionTool struct {
	name   string
	result string
}

func (t *FunctionTool) Name() string {
	return t.name
}

func (t *FunctionTool) Description() string {
	return "test tool"
}

func (t *FunctionTool) ParamsJSONSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"a": map[string]any{
				"type": "string",
			},
		},
	}
}

func (t *FunctionTool) Invoke(ctx context.Context, input string) (string, error) {
	return t.result, nil
}

func GetFinalOutputMessage(content string) model.Message {
	// Final output flag is handled internally by Runner
	return model.Message{
		Role:    "assistant",
		Content: content,
	}
}
