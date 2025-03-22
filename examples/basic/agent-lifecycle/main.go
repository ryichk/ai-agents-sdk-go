// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/ryichk/ai-agents-sdk-go/agent"
	"github.com/ryichk/ai-agents-sdk-go/handoff"
	"github.com/ryichk/ai-agents-sdk-go/model"
	"github.com/ryichk/ai-agents-sdk-go/runner"
	"github.com/ryichk/ai-agents-sdk-go/tool"
	"github.com/ryichk/ai-agents-sdk-go/tracing"
)

type ExampleHooks struct {
	agent.BaseAgentHooks
	EventCounter int
}

func (h *ExampleHooks) OnStart(ctx context.Context, a *agent.Agent) error {
	h.EventCounter++
	fmt.Printf("### %d: Agent %s started.\n", h.EventCounter, a.Name)
	return nil
}

func (h *ExampleHooks) OnEnd(ctx context.Context, a *agent.Agent, output any) error {
	h.EventCounter++
	fmt.Printf("### %d: Agent %s ended with output %v.\n", h.EventCounter, a.Name, output)
	return nil
}

func (h *ExampleHooks) OnHandoff(ctx context.Context, a *agent.Agent, source *agent.Agent) error {
	h.EventCounter++
	fmt.Printf("### %d: Handoff from %s to %s.\n", h.EventCounter, source.Name, a.Name)
	return nil
}

func (h *ExampleHooks) OnToolStart(ctx context.Context, a *agent.Agent, t tool.Tool) error {
	h.EventCounter++
	fmt.Printf("### %d: Tool %s started.\n", h.EventCounter, t.Name())
	return nil
}

func (h *ExampleHooks) OnToolEnd(ctx context.Context, a *agent.Agent, t tool.Tool, result string) error {
	h.EventCounter++
	fmt.Printf("### %d: Tool %s ended with result %s.\n", h.EventCounter, t.Name(), result)
	return nil
}

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

type MultiplyByTwoTool struct{}

func (t *MultiplyByTwoTool) Name() string {
	return "multiply_by_two"
}

func (t *MultiplyByTwoTool) Description() string {
	return "Return x times two."
}

func (t *MultiplyByTwoTool) ParamsJSONSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"x": map[string]any{
				"type":        "integer",
				"description": "The number to multiply by two",
			},
		},
		"required": []string{"x"},
	}
}

func (t *MultiplyByTwoTool) Invoke(ctx context.Context, input string) (string, error) {
	var params struct {
		X int `json:"x"`
	}
	if err := json.Unmarshal([]byte(input), &params); err != nil {
		return "", err
	}
	result := params.X * 2
	return strconv.Itoa(result), nil
}

type AgentHandoff struct {
	targetAgent *agent.Agent
}

func NewAgentHandoff(target *agent.Agent) *AgentHandoff {
	return &AgentHandoff{targetAgent: target}
}

func (h *AgentHandoff) TargetAgent() any {
	return h.targetAgent
}

func (h *AgentHandoff) Description() string {
	return "Handoff to " + h.targetAgent.Name
}

func (h *AgentHandoff) ShouldHandoff(ctx context.Context, input string) (bool, error) {
	return true, nil
}

func (h *AgentHandoff) Name() string {
	return "handoff_to_" + h.targetAgent.Name
}

func (h *AgentHandoff) FilterInput(ctx context.Context, inputData *handoff.InputData) (*handoff.InputData, error) {
	return inputData, nil
}

func (h *AgentHandoff) GetLastHandoffTime() time.Time {
	return time.Time{}
}

func (h *AgentHandoff) UpdateLastHandoffTime() {}

func (h *AgentHandoff) ToolName() string {
	return "handoff_to_" + h.targetAgent.Name
}

func (h *AgentHandoff) ToolDescription() string {
	return "Handoff to " + h.targetAgent.Name
}

func (h *AgentHandoff) InputJSONSchema() handoff.JSONSchema {
	return handoff.JSONSchema{
		"type": "object",
		"properties": map[string]any{
			"input": map[string]any{
				"type": "string",
			},
		},
	}
}

func (h *AgentHandoff) OnHandoff(ctx context.Context, inputData *handoff.InputData, inputJSON string) error {
	return nil
}

func main() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("OPENAI_API_KEY environment variable is not set")
	}

	err := tracing.EnableTracing(os.Getenv("OPENAI_API_KEY"), "./traces/lifecycle-example", 100, 5*time.Second)
	if err != nil {
		log.Printf("Failed to initialize tracing: %v", err)
	}

	provider, err := model.NewDefaultOpenAIProvider()
	if err != nil {
		log.Fatalf("Failed to create OpenAI provider: %v", err)
	}

	hooks := &ExampleHooks{}

	multiplyAgent := agent.New("Multiply Agent", "Multiply the number by 2 and then return the final result.")
	multiplyAgent.AddTool(&MultiplyByTwoTool{})
	multiplyAgent.SetHooks(hooks)

	startAgent := agent.New("Start Agent", "Generate a random number. If it's even, stop. If it's odd, hand off to the multipler agent.")
	startAgent.AddTool(&RandomNumberTool{})
	startAgent.AddHandoff(NewAgentHandoff(multiplyAgent))
	startAgent.SetHooks(hooks)

	// Get user input
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Enter a max number: ")
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)

	ctx := context.Background()

	_, err = runner.RunWithConfig(ctx, startAgent, fmt.Sprintf("Generate a random number between 0 and %s.", input), runner.RunConfig{
		MaxTurns:      10,
		ModelProvider: provider,
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Done!")
}
