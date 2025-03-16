// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/ryichk/ai-agents-sdk-go/agent"
	"github.com/ryichk/ai-agents-sdk-go/model"
	"github.com/ryichk/ai-agents-sdk-go/runner"
	"github.com/ryichk/ai-agents-sdk-go/tool"
)

func getWeather(city string) string {
	// In actual implementation, call an external API to get weather information
	return fmt.Sprintf("%s's weather is sunny.", city)
}

func getCurrentTime(timezone string) string {
	// In actual implementation, return the current time according to the timezone
	return fmt.Sprintf("%s's current time is 14:30.", timezone)
}

func main() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("OPENAI_API_KEY environment variable is not set")
	}

	provider, err := model.NewDefaultOpenAIProvider()
	if err != nil {
		log.Fatalf("Failed to create OpenAI provider: %v", err)
	}

	agent := agent.New(
		"Information Assistant",
		"You are a helpful information assistant. Use tools to collect information and answer the user's question politely.",
	)

	weatherTool, err := tool.NewFunctionToolFromFunc(getWeather, "Get weather information for the specified city")
	if err != nil {
		log.Fatalf("Failed to create weather tool: %v", err)
	}
	agent.AddTool(weatherTool)

	timeTool, err := tool.NewFunctionToolFromFunc(getCurrentTime, "Get the current time in the specified timezone")
	if err != nil {
		log.Fatalf("Failed to create time tool: %v", err)
	}
	agent.AddTool(timeTool)

	config := runner.DefaultRunConfig()
	config.ModelProvider = provider

	ctx := context.Background()

	result, err := runner.RunWithConfig(ctx, agent, "Tell me the weather and time in Tokyo.", config)
	if err != nil {
		log.Fatalf("Failed to run agent: %v", err)
	}

	fmt.Println("=== Agent's response ===")
	fmt.Println(result.FinalOutput)
	fmt.Println()

	fmt.Println("Token usage:")
	fmt.Printf("   Prompt: %d\n", result.Usage.PromptTokens)
	fmt.Printf("   Completion: %d\n", result.Usage.CompletionTokens)
	fmt.Printf("   Total: %d\n", result.Usage.TotalTokens)
}
