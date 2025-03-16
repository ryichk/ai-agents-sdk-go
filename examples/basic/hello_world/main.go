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
)

func main() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("OPENAI_API_KEY environment variable is not set")
	}

	provider, err := model.NewDefaultOpenAIProvider()
	if err != nil {
		log.Fatalf("Failed to create OpenAI provider: %v", err)
	}

	agent := agent.New("Hello World Agent", "You are a helpful assistant. Answer the user's question briefly.")

	ctx := context.Background()

	result, err := runner.RunWithConfig(ctx, agent, "What is recursion? Explain it briefly.", runner.RunConfig{
		ModelProvider: provider,
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("=== Agent's response ===")
	if result != nil {
		fmt.Println(result.FinalOutput)
	} else {
		fmt.Println("No response")
	}

	fmt.Println("Token usage:")
	fmt.Printf("   Prompt: %d\n", result.Usage.PromptTokens)
	fmt.Printf("   Completion: %d\n", result.Usage.CompletionTokens)
	fmt.Printf("   Total: %d\n", result.Usage.TotalTokens)
}
