// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/ryichk/ai-agents-sdk-go/agent"
	"github.com/ryichk/ai-agents-sdk-go/handoff"
	"github.com/ryichk/ai-agents-sdk-go/model"
	"github.com/ryichk/ai-agents-sdk-go/runner"
	"github.com/ryichk/ai-agents-sdk-go/tool"
)

type RandomNumberTool struct{}

func (t *RandomNumberTool) Name() string {
	return "random_number"
}

func (t *RandomNumberTool) Description() string {
	return "Generate a random number up to the provided max."
}

func (t *RandomNumberTool) ParamsJSONSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"max": map[string]any{
				"type":        "integer",
				"description": "The maximum value for the random number",
			},
		},
		"required": []string{"max"},
	}
}

func (t *RandomNumberTool) Invoke(ctx context.Context, input string) (string, error) {
	var params struct {
		Max int `json:"max"`
	}
	if err := json.Unmarshal([]byte(input), &params); err != nil {
		return "", err
	}
	result := rand.Intn(params.Max + 1)
	return strconv.Itoa(result), nil
}

type MessageFilterHandoff struct {
	targetAgent  *agent.Agent
	lastHandoff  time.Time
	filterPrefix string
}

func NewMessageFilterHandoff(target *agent.Agent, filterPrefix string) *MessageFilterHandoff {
	return &MessageFilterHandoff{
		targetAgent:  target,
		filterPrefix: filterPrefix,
	}
}

func (h *MessageFilterHandoff) TargetAgent() any {
	return h.targetAgent
}

func (h *MessageFilterHandoff) Description() string {
	return fmt.Sprintf("Handoff to %s with filtered messages", h.targetAgent.Name)
}

func (h *MessageFilterHandoff) ShouldHandoff(ctx context.Context, input string) (bool, error) {
	return true, nil
}

func (h *MessageFilterHandoff) Name() string {
	return "handoff_to_" + h.targetAgent.Name
}

func (h *MessageFilterHandoff) FilterInput(ctx context.Context, inputData *handoff.InputData) (*handoff.InputData, error) {
	// Message filtering process
	for i, msg := range inputData.InputHistory {
		// Filter user messages only
		if role, ok := msg["role"].(string); ok && role == "user" {
			if content, ok := msg["content"].(string); ok {
				// Add filter prefix
				msg["content"] = h.filterPrefix + ": " + content
				inputData.InputHistory[i] = msg
			}
		}
	}

	fmt.Println("Messages filtered:")
	for _, msg := range inputData.InputHistory {
		if role, ok := msg["role"].(string); ok && role == "user" {
			if content, ok := msg["content"].(string); ok {
				fmt.Printf("- %s\n", content)
			}
		}
	}

	return inputData, nil
}

func (h *MessageFilterHandoff) GetLastHandoffTime() time.Time {
	return h.lastHandoff
}

func (h *MessageFilterHandoff) UpdateLastHandoffTime() {
	h.lastHandoff = time.Now()
}

func (h *MessageFilterHandoff) ToolName() string {
	return "handoff_to_" + h.targetAgent.Name
}

func (h *MessageFilterHandoff) ToolDescription() string {
	return fmt.Sprintf("Handoff to %s with filtered messages", h.targetAgent.Name)
}

func (h *MessageFilterHandoff) InputJSONSchema() handoff.JSONSchema {
	return handoff.JSONSchema{
		"type": "object",
		"properties": map[string]any{
			"input": map[string]any{
				"type": "string",
			},
		},
	}
}

func (h *MessageFilterHandoff) OnHandoff(ctx context.Context, inputData *handoff.InputData, inputJSON string) error {
	fmt.Printf("Handoff started: %s -> %s\n", "Source Agent", h.targetAgent.Name)
	return nil
}

type ExampleHooks struct {
	agent.BaseAgentHooks
}

func (h *ExampleHooks) OnStart(ctx context.Context, a *agent.Agent) error {
	fmt.Printf("Agent started: %s\n", a.Name)
	return nil
}

func (h *ExampleHooks) OnEnd(ctx context.Context, a *agent.Agent, output any) error {
	fmt.Printf("Agent ended: %s\n", a.Name)
	return nil
}

func (h *ExampleHooks) OnHandoff(ctx context.Context, a *agent.Agent, source *agent.Agent) error {
	fmt.Printf("Handoff occurred: %s -> %s\n", source.Name, a.Name)
	return nil
}

func (h *ExampleHooks) OnToolStart(ctx context.Context, a *agent.Agent, t tool.Tool) error {
	fmt.Printf("Tool started: %s\n", t.Name())
	return nil
}

func (h *ExampleHooks) OnToolEnd(ctx context.Context, a *agent.Agent, t tool.Tool, result string) error {
	fmt.Printf("Tool ended: %s, result: %s\n", t.Name(), result)
	return nil
}

func main() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("OPENAI_API_KEY environment variable is not set")
	}

	rand.Seed(time.Now().UnixNano())

	provider, err := model.NewDefaultOpenAIProvider()
	if err != nil {
		log.Fatalf("Failed to create OpenAI provider: %v", err)
	}

	hooks := &ExampleHooks{}

	finalAgent := agent.New("Final Agent", "Say 'Success!' and include the value of the random number generated.")
	finalAgent.SetHooks(hooks)

	intermediateAgent := agent.New("Intermediate Agent", "Generate a random number between 1 and 10, then hand off to the final agent.")
	intermediateAgent.AddTool(&RandomNumberTool{})
	intermediateAgent.SetHooks(hooks)

	filterHandoff := NewMessageFilterHandoff(finalAgent, "Random number")
	intermediateAgent.AddHandoff(filterHandoff)

	startAgent := agent.New("Start Agent", "Generate a random number between 1 and 5, then hand off to the intermediate agent.")
	startAgent.AddTool(&RandomNumberTool{})
	startAgent.SetHooks(hooks)

	startHandoff := handoff.NewHandoff(intermediateAgent, "Handoff to the intermediate agent")
	startAgent.AddHandoff(startHandoff)

	ctx := context.Background()

	result, err := runner.RunWithConfig(ctx, startAgent, "Start the process", runner.RunConfig{
		MaxTurns:      10,
		ModelProvider: provider,
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("\n=== Final Result ===")
	if result != nil {
		fmt.Println(result.FinalOutput)
	} else {
		fmt.Println("No result available")
	}

	fmt.Println("Done!")
}
