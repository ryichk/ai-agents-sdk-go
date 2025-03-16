// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package model

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// OpenAIConfig represents OpenAI provider configuration
type OpenAIConfig struct {
	// APIKey is the OpenAI API key
	APIKey string

	// BaseURL is the custom base URL (optional)
	BaseURL string

	// Organization is the OpenAI Organization (optional)
	Organization string
}

type OpenAIProvider struct {
	config OpenAIConfig
	client *openai.Client
}

func NewOpenAIProvider(config OpenAIConfig) (*OpenAIProvider, error) {
	if config.APIKey == "" {
		config.APIKey = os.Getenv("OPENAI_API_KEY")
		if config.APIKey == "" {
			return nil, errors.New("OpenAI API key is required")
		}
	}

	clientConfig := openai.DefaultConfig(config.APIKey)
	if config.BaseURL != "" {
		clientConfig.BaseURL = config.BaseURL
	}
	if config.Organization != "" {
		clientConfig.OrgID = config.Organization
	}

	return &OpenAIProvider{
		config: config,
		client: openai.NewClientWithConfig(clientConfig),
	}, nil
}

// NewDefaultOpenAIProvider creates an OpenAI provider using API key from environment variables
func NewDefaultOpenAIProvider() (*OpenAIProvider, error) {
	return NewOpenAIProvider(OpenAIConfig{})
}

func (p *OpenAIProvider) CreateChatCompletion(ctx context.Context, messages []Message, settings Settings) (*Response, error) {
	openaiMessages := convertToOpenAIMessages(messages)

	request := openai.ChatCompletionRequest{
		Model:            getModelName(settings),
		Messages:         openaiMessages,
		Temperature:      float32(settings.Temperature),
		MaxTokens:        settings.MaxTokens,
		TopP:             float32(settings.TopP),
		FrequencyPenalty: float32(settings.FrequencyPenalty),
		PresencePenalty:  float32(settings.PresencePenalty),
		Stop:             settings.StopSequences,
	}

	if tools, ok := settings.Custom["tools"].([]map[string]any); ok && len(tools) > 0 {
		openaiTools := make([]openai.Tool, 0, len(tools))
		for _, toolDef := range tools {
			openaiTool, err := mapToOpenAITool(toolDef)
			if err != nil {
				continue
			}
			openaiTools = append(openaiTools, openaiTool)
		}
		request.Tools = openaiTools
	}

	if toolChoice, ok := settings.Custom["tool_choice"].(string); ok {
		switch toolChoice {
		case "auto":
			request.ToolChoice = "auto"
		case "none":
			request.ToolChoice = "none"
		default:
			if strings.HasPrefix(toolChoice, "force_") {
				toolName := strings.TrimPrefix(toolChoice, "force_")
				request.ToolChoice = map[string]any{
					"type": "function",
					"function": map[string]any{
						"name": toolName,
					},
				}
			}
		}
	}

	if settings.ResponseFormat != "" {
		if settings.ResponseFormat == "json_object" {
			request.ResponseFormat = &openai.ChatCompletionResponseFormat{
				Type: openai.ChatCompletionResponseFormatTypeJSONObject,
			}
		}
	}

	result, err := p.client.CreateChatCompletion(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("OpenAI API call failed: %w", err)
	}

	if len(result.Choices) == 0 {
		return nil, errors.New("no response from OpenAI")
	}

	choice := result.Choices[0]

	toolCalls, err := convertAPIToolCalls(choice.Message.ToolCalls)
	if err != nil {
		return nil, fmt.Errorf("error converting tool calls: %w", err)
	}

	response := &Response{
		Message: Message{
			Role:      choice.Message.Role,
			Content:   choice.Message.Content,
			ToolCalls: toolCalls,
		},
		Usage: Usage{
			PromptTokens:     result.Usage.PromptTokens,
			CompletionTokens: result.Usage.CompletionTokens,
			TotalTokens:      result.Usage.TotalTokens,
		},
	}

	return response, nil
}

// CreateChatCompletionStream creates a streaming chat completion
func (p *OpenAIProvider) CreateChatCompletionStream(ctx context.Context, messages []Message, settings Settings) (Stream, error) {
	openaiMessages := convertToOpenAIMessages(messages)

	request := openai.ChatCompletionRequest{
		Model:            getModelName(settings),
		Messages:         openaiMessages,
		Temperature:      float32(settings.Temperature),
		MaxTokens:        settings.MaxTokens,
		TopP:             float32(settings.TopP),
		FrequencyPenalty: float32(settings.FrequencyPenalty),
		PresencePenalty:  float32(settings.PresencePenalty),
		Stop:             settings.StopSequences,
		Stream:           true,
	}

	if tools, ok := settings.Custom["tools"].([]map[string]any); ok && len(tools) > 0 {
		openaiTools := make([]openai.Tool, 0, len(tools))
		for _, toolDef := range tools {
			openaiTool, err := mapToOpenAITool(toolDef)
			if err != nil {
				continue
			}
			openaiTools = append(openaiTools, openaiTool)
		}
		request.Tools = openaiTools
	}

	if toolChoice, ok := settings.Custom["tool_choice"].(string); ok {
		switch toolChoice {
		case "auto":
			request.ToolChoice = "auto"
		case "none":
			request.ToolChoice = "none"
		default:
			if strings.HasPrefix(toolChoice, "force_") {
				toolName := strings.TrimPrefix(toolChoice, "force_")
				request.ToolChoice = map[string]any{
					"type": "function",
					"function": map[string]any{
						"name": toolName,
					},
				}
			}
		}
	}

	if settings.ResponseFormat != "" {
		if settings.ResponseFormat == "json_object" {
			request.ResponseFormat = &openai.ChatCompletionResponseFormat{
				Type: openai.ChatCompletionResponseFormatTypeJSONObject,
			}
		}
	}

	stream, err := p.client.CreateChatCompletionStream(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("OpenAI API stream call failed: %w", err)
	}

	return &OpenAIStream{
		stream: stream,
	}, nil
}

// OpenAIStream handles OpenAI streaming responses
type OpenAIStream struct {
	stream *openai.ChatCompletionStream
}

// Recv receives the next chunk from the stream
func (s *OpenAIStream) Recv() (*StreamChunk, error) {
	resp, err := s.stream.Recv()
	if err != nil {
		if err == io.EOF {
			return nil, io.EOF
		}
		return nil, fmt.Errorf("failed to receive from stream: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, errors.New("no choices in stream response")
	}

	choice := resp.Choices[0]

	chunk := &StreamChunk{
		Delta: Message{
			Role:    choice.Delta.Role,
			Content: choice.Delta.Content,
		},
		FinishReason: string(choice.FinishReason),
	}

	// Convert tool calls
	if len(choice.Delta.ToolCalls) > 0 {
		toolCalls := make([]ToolCall, len(choice.Delta.ToolCalls))
		for i, tc := range choice.Delta.ToolCalls {
			toolCalls[i] = ToolCall{
				ID:   tc.ID,
				Type: "function", // Currently only function type is supported
				Function: FunctionCall{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			}
		}
		chunk.Delta.ToolCalls = toolCalls
	}

	return chunk, nil
}

func (s *OpenAIStream) Close() error {
	return s.stream.Close()
}

func mapToOpenAITool(toolMap map[string]any) (openai.Tool, error) {
	tool := openai.Tool{}

	toolType, ok := toolMap["type"].(string)
	if !ok {
		return tool, errors.New("invalid tool type")
	}

	tool.Type = openai.ToolTypeFunction

	if toolType == "function" {
		functionMap, ok := toolMap["function"].(map[string]any)
		if !ok {
			return tool, errors.New("invalid function definition")
		}

		function := openai.FunctionDefinition{}

		if name, ok := functionMap["name"].(string); ok {
			function.Name = name
		} else {
			return tool, errors.New("function name is required")
		}

		if description, ok := functionMap["description"].(string); ok {
			function.Description = description
		}

		if parameters, ok := functionMap["parameters"].(map[string]any); ok {
			function.Parameters = parameters
		} else {
			function.Parameters = map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			}
		}

		tool.Function = &function
	}

	return tool, nil
}

// convertToOpenAIMessages converts messages to OpenAI format
func convertToOpenAIMessages(messages []Message) []openai.ChatCompletionMessage {
	result := make([]openai.ChatCompletionMessage, len(messages))
	for i, msg := range messages {
		result[i] = openai.ChatCompletionMessage{
			Role:      msg.Role,
			Content:   msg.Content,
			ToolCalls: convertToOpenAIToolCalls(msg.ToolCalls),
		}
		if msg.Name != "" {
			result[i].Name = msg.Name
		}
		if msg.ToolCallID != "" {
			result[i].ToolCallID = msg.ToolCallID
		}
	}
	return result
}

// convertToOpenAIToolCalls converts tool calls to OpenAI format
func convertToOpenAIToolCalls(toolCalls []ToolCall) []openai.ToolCall {
	result := make([]openai.ToolCall, len(toolCalls))
	for i, tc := range toolCalls {
		result[i] = openai.ToolCall{
			ID:   tc.ID,
			Type: openai.ToolType(tc.Type),
			Function: openai.FunctionCall{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		}
	}
	return result
}

func getModelName(settings Settings) string {
	defaultModel := "gpt-4o"

	if modelName, ok := settings.Custom["model"].(string); ok && modelName != "" {
		return modelName
	}

	return defaultModel
}

func convertAPIToolCalls(apiToolCalls []openai.ToolCall) ([]ToolCall, error) {
	result := make([]ToolCall, 0, len(apiToolCalls))

	for _, tc := range apiToolCalls {
		newTc := ToolCall{
			ID: tc.ID,
		}

		newTc.Type = "function" // Currently only function is supported

		if tc.Function.Name != "" {
			newTc.Function = FunctionCall{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			}
		}

		result = append(result, newTc)
	}

	return result, nil
}
