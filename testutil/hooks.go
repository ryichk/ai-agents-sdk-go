// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package testutil

import (
	"context"
	"time"

	"github.com/ryichk/ai-agents-sdk-go/handoff"
	"github.com/ryichk/ai-agents-sdk-go/tool"
)

type AgentInterface interface {
	GetName() string
	GetSystemPrompt(ctx context.Context) (string, error)
}

type AgentHooksInterface interface {
	OnStart(ctx context.Context, agent any) error
	OnEnd(ctx context.Context, agent any, output any) error
	OnHandoff(ctx context.Context, agent any, source any) error
	OnToolStart(ctx context.Context, agent any, t tool.Tool) error
	OnToolEnd(ctx context.Context, agent any, t tool.Tool, result string) error
}

type BaseAgentHooks struct{}

func (h *BaseAgentHooks) OnStart(ctx context.Context, agent any) error {
	return nil
}

func (h *BaseAgentHooks) OnEnd(ctx context.Context, agent any, output any) error {
	return nil
}

func (h *BaseAgentHooks) OnHandoff(ctx context.Context, agent any, source any) error {
	return nil
}

func (h *BaseAgentHooks) OnToolStart(ctx context.Context, agent any, t tool.Tool) error {
	return nil
}

func (h *BaseAgentHooks) OnToolEnd(ctx context.Context, agent any, t tool.Tool, result string) error {
	return nil
}

type TestHooks struct {
	BaseAgentHooks
	StartCount     int
	EndCount       int
	HandoffCount   int
	ToolStartCount int
	ToolEndCount   int
}

func (h *TestHooks) OnStart(ctx context.Context, agent any) error {
	h.StartCount++
	return nil
}

func (h *TestHooks) OnEnd(ctx context.Context, agent any, output any) error {
	h.EndCount++
	return nil
}

func (h *TestHooks) OnHandoff(ctx context.Context, agent any, source any) error {
	h.HandoffCount++
	return nil
}

func (h *TestHooks) OnToolStart(ctx context.Context, agent any, t tool.Tool) error {
	h.ToolStartCount++
	return nil
}

func (h *TestHooks) OnToolEnd(ctx context.Context, agent any, t tool.Tool, result string) error {
	h.ToolEndCount++
	return nil
}

type TestTool struct {
	name        string
	description string
	result      string
}

func NewTestTool(name string, description string, result string) *TestTool {
	return &TestTool{
		name:        name,
		description: description,
		result:      result,
	}
}

func (t *TestTool) Name() string {
	return t.name
}

func (t *TestTool) Description() string {
	return t.description
}

func (t *TestTool) ParamsJSONSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"param": map[string]any{
				"type": "string",
			},
		},
	}
}

func (t *TestTool) Invoke(ctx context.Context, input string) (string, error) {
	return t.result, nil
}

// HandoffInterface is the interface for handoffs
type HandoffInterface interface {
	TargetAgent() any
	Description() string
	ShouldHandoff(ctx context.Context, input string) (bool, error)
	Name() string
	FilterInput(ctx context.Context, inputData *handoff.InputData) (*handoff.InputData, error)
	GetLastHandoffTime() time.Time
	UpdateLastHandoffTime()
	ToolName() string
	ToolDescription() string
	InputJSONSchema() handoff.JSONSchema
	OnHandoff(ctx context.Context, inputData *handoff.InputData, inputJSON string) error
}

// TestHandoff is a handoff for testing
type TestHandoff struct {
	targetAgent     any
	name            string
	description     string
	toolName        string
	toolDescription string
	lastHandoffTime time.Time
}

// NewTestHandoff creates a new test handoff
func NewTestHandoff(name string, targetAgent any) *TestHandoff {
	return &TestHandoff{
		name:            name,
		targetAgent:     targetAgent,
		description:     "Test handoff description",
		toolName:        name,
		toolDescription: "Test handoff tool description",
		lastHandoffTime: time.Time{},
	}
}

func (h *TestHandoff) TargetAgent() any {
	return h.targetAgent
}

func (h *TestHandoff) Description() string {
	return h.description
}

func (h *TestHandoff) ShouldHandoff(ctx context.Context, input string) (bool, error) {
	return true, nil
}

func (h *TestHandoff) Name() string {
	return h.name
}

func (h *TestHandoff) FilterInput(ctx context.Context, inputData *handoff.InputData) (*handoff.InputData, error) {
	return inputData, nil
}

func (h *TestHandoff) GetLastHandoffTime() time.Time {
	return h.lastHandoffTime
}

func (h *TestHandoff) UpdateLastHandoffTime() {
	h.lastHandoffTime = time.Now()
}

func (h *TestHandoff) ToolName() string {
	return h.toolName
}

func (h *TestHandoff) ToolDescription() string {
	return h.toolDescription
}

func (h *TestHandoff) InputJSONSchema() handoff.JSONSchema {
	return handoff.JSONSchema{
		"type": "object",
		"properties": map[string]any{
			"input": map[string]any{
				"type": "string",
			},
		},
	}
}

func (h *TestHandoff) OnHandoff(ctx context.Context, inputData *handoff.InputData, inputJSON string) error {
	return nil
}
