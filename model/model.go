// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package model

import (
	"context"
)

// Settings represents model settings
type Settings struct {
	// Temperature sets the generation temperature (0.0-2.0)
	Temperature float64

	// MaxTokens sets the maximum number of tokens to generate
	MaxTokens int

	// TopP sets the top P for generation (0.0-1.0)
	TopP float64

	// FrequencyPenalty sets the frequency penalty (-2.0-2.0)
	FrequencyPenalty float64

	// PresencePenalty sets the presence penalty (-2.0-2.0)
	PresencePenalty float64

	// StopSequences sets sequences that stop generation
	StopSequences []string

	ResponseFormat string

	// Seed sets the generation seed
	Seed int

	// Tools sets tool definitions
	Tools []map[string]any

	// Custom holds custom settings
	Custom map[string]any
}

// DefaultSettings returns default model settings
func DefaultSettings() Settings {
	return Settings{
		Temperature:      0.7,
		MaxTokens:        1024,
		TopP:             1.0,
		FrequencyPenalty: 0.0,
		PresencePenalty:  0.0,
		StopSequences:    []string{},
		ResponseFormat:   "",
		Seed:             0,
		Tools:            []map[string]any{},
		Custom:           make(map[string]any),
	}
}

// Message represents a chat message
type Message struct {
	// Role is the role of the message (system, user, assistant, tool)
	Role       string
	Content    string
	ToolCalls  []ToolCall
	ToolCallID string
	Name       string
}

type ToolCall struct {
	ID       string
	Type     string
	Function FunctionCall
}

type FunctionCall struct {
	Name      string
	Arguments string
}

// Provider is the interface for model providers
type Provider interface {
	CreateChatCompletion(ctx context.Context, messages []Message, settings Settings) (*Response, error)
	CreateChatCompletionStream(ctx context.Context, messages []Message, settings Settings) (Stream, error)
}

// Response represents a model response
type Response struct {
	Message Message
	Usage   Usage
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

// Stream is the interface for streaming responses
type Stream interface {
	// Recv receives the next chunk from the stream
	Recv() (*StreamChunk, error)

	// Close closes the stream
	Close() error
}

type StreamChunk struct {
	Delta Message

	FinishReason string
}
