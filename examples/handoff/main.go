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
	"github.com/ryichk/ai-agents-sdk-go/handoff"
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

	japaneseAgent := agent.New(
		"日本語エージェント",
		"あなたは日本語のみで話すアシスタントです。質問に丁寧に回答してください。",
	)

	englishAgent := agent.New(
		"English Agent",
		"You are an assistant that only speaks English. Answer questions politely.",
	)

	spanishAgent := agent.New(
		"Spanish Agent",
		"Eres un asistente que solo habla español. Responde a las preguntas educadamente.",
	)

	triageAgent := agent.New(
		"Language Triage Agent",
		"You are an agent that determines the user's input language and hands off to the appropriate language agent."+
			"If the user's input is in Japanese, hand off to the Japanese agent. If it's in English, hand off to the English agent. If it's in Spanish, hand off to the Spanish agent.",
	)

	triageAgent.AddHandoffs(
		handoff.NewHandoff(japaneseAgent, "Process Japanese queries"),
		handoff.NewHandoff(englishAgent, "Process English queries"),
		handoff.NewHandoff(spanishAgent, "Process Spanish queries"),
	)

	config := runner.DefaultRunConfig()
	config.ModelProvider = provider

	queries := []struct {
		lang  string
		query string
	}{
		{"日本語", "こんにちは、今日の天気はどうですか？"},
		{"English", "Hello, how are you today?"},
		{"Spanish", "Hola, ¿cómo estás hoy?"},
	}

	ctx := context.Background()
	for _, q := range queries {
		fmt.Printf("\n--- Processing %s query ---\n", q.lang)
		fmt.Printf("Input: %s\n", q.query)

		result, err := runner.RunWithConfig(ctx, triageAgent, q.query, config)
		if err != nil {
			log.Printf("Failed to run agent: %v", err)
			continue
		}

		fmt.Println("Output:")
		fmt.Println(result.FinalOutput)

		fmt.Println("Token usage:")
		fmt.Printf("   Total: %d\n", result.Usage.TotalTokens)
	}
}
