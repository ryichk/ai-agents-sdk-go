// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/ryichk/ai-agents-sdk-go/agent"
	"github.com/ryichk/ai-agents-sdk-go/model"
	"github.com/ryichk/ai-agents-sdk-go/runner"
	"github.com/ryichk/ai-agents-sdk-go/tool"
)

/*
This example shows the agents-as-tools pattern. The orchestrator agent receives a user message and
then picks which agents to call as tools. In this case, it picks from a set of translation
agents (Spanish, French, Italian, and Japanese).

After the translations are completed, a synthesizer agent inspects, corrects if needed,
and produces a final concatenated response.
*/

// CustomRunnerImpl is a custom implementation of the RunnerIF interface
type CustomRunnerImpl struct {
	Config runner.RunConfig
}

// Run runs the agent with the custom configuration
func (r *CustomRunnerImpl) Run(ctx context.Context, agentIF any, input string) (any, error) {
	// Check if the agent is of the correct type
	a, ok := agentIF.(*agent.Agent)
	if !ok {
		return nil, fmt.Errorf("agent must be of type *agent.Agent")
	}

	// Run with the custom configuration
	result, err := runner.RunWithConfig(ctx, a, input, r.Config)
	if err != nil {
		return nil, err
	}

	// Wrap the result to implement the RunResult interface
	return &runner.GetResult{Result: result}, nil
}

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("OPENAI_API_KEY environment variable is required")
		os.Exit(1)
	}

	config := model.OpenAIConfig{
		APIKey: apiKey,
	}
	provider, err := model.NewOpenAIProvider(config)
	if err != nil {
		fmt.Printf("Error creating OpenAI provider: %v\n", err)
		os.Exit(1)
	}
	runner.DefaultProvider = provider

	// Create a run config
	runConfig := runner.DefaultRunConfig()
	runConfig.ModelProvider = provider

	// Create a custom runner implementation with the config
	customRunner := &CustomRunnerImpl{
		Config: runConfig,
	}

	// Create translation agents
	spanishAgent := agent.New(
		"spanish_agent",
		"You translate the user's message to Spanish. Respond only with the Spanish translation, nothing else.",
	)
	spanishAgent.HandoffDescription = "An English to Spanish translator"

	frenchAgent := agent.New(
		"french_agent",
		"You translate the user's message to French. Respond only with the French translation, nothing else.",
	)
	frenchAgent.HandoffDescription = "An English to French translator"

	italianAgent := agent.New(
		"italian_agent",
		"You translate the user's message to Italian. Respond only with the Italian translation, nothing else.",
	)
	italianAgent.HandoffDescription = "An English to Italian translator"

	japaneseAgent := agent.New(
		"japanese_agent",
		"You translate the user's message to Japanese. Respond only with the Japanese translation, nothing else.",
	)
	japaneseAgent.HandoffDescription = "An English to Japanese translator"

	// Create agent tools using the custom runner
	spanishTool, err := spanishAgent.AsTool(customRunner, tool.AgentToolOption{
		Name:        "translate_to_spanish",
		Description: "Translate the user's message to Spanish",
	})
	if err != nil {
		fmt.Printf("Error creating Spanish tool: %v\n", err)
		os.Exit(1)
	}

	frenchTool, err := frenchAgent.AsTool(customRunner, tool.AgentToolOption{
		Name:        "translate_to_french",
		Description: "Translate the user's message to French",
	})
	if err != nil {
		fmt.Printf("Error creating French tool: %v\n", err)
		os.Exit(1)
	}

	italianTool, err := italianAgent.AsTool(customRunner, tool.AgentToolOption{
		Name:        "translate_to_italian",
		Description: "Translate the user's message to Italian",
	})
	if err != nil {
		fmt.Printf("Error creating Italian tool: %v\n", err)
		os.Exit(1)
	}

	japaneseTool, err := japaneseAgent.AsTool(customRunner, tool.AgentToolOption{
		Name:        "translate_to_japanese",
		Description: "Translate the user's message to Japanese",
	})
	if err != nil {
		fmt.Printf("Error creating Japanese tool: %v\n", err)
		os.Exit(1)
	}

	// Create orchestrator agent
	orchestratorAgent := agent.New(
		"orchestrator_agent",
		`You are a translation agent. You use the tools given to you to translate.
If asked for multiple translations, you call the relevant tools in order.
You never translate on your own, you always use the provided tools.
Always prefix each translation with the language name, like "Spanish: [translation]".`,
	)

	// Add translation tools to the orchestrator
	orchestratorAgent.AddTool(spanishTool)
	orchestratorAgent.AddTool(frenchTool)
	orchestratorAgent.AddTool(italianTool)
	orchestratorAgent.AddTool(japaneseTool)

	// Create synthesizer agent
	synthesizerAgent := agent.New(
		"synthesizer_agent",
		`You inspect translations, correct them if needed, and produce a final concatenated response.
Present all translations in a clear, organized format.
Add a brief evaluation of the quality and accuracy of each translation.`,
	)

	// Get user input
	fmt.Println("Hi! What would you like translated, and to which languages?")
	fmt.Print("> ")
	reader := bufio.NewReader(os.Stdin)
	userInput, _ := reader.ReadString('\n')
	userInput = strings.TrimSpace(userInput)

	// Run orchestrator agent with user input
	fmt.Println("\nProcessing your request...")
	orchestratorResult, err := runner.RunWithConfig(context.Background(), orchestratorAgent, userInput, runConfig)
	if err != nil {
		fmt.Printf("Error running orchestrator agent: %v\n", err)
		os.Exit(1)
	}

	// Print translation steps
	fmt.Println("\nTranslation steps:")
	fmt.Println(orchestratorResult.FinalOutput)

	// Pass the orchestrator result to the synthesizer
	synthesizerResult, err := runner.RunWithConfig(context.Background(), synthesizerAgent, orchestratorResult.FinalOutput, runConfig)
	if err != nil {
		fmt.Printf("Error running synthesizer agent: %v\n", err)
		os.Exit(1)
	}

	// Print the final synthesized result
	fmt.Println("\nFinal response:")
	fmt.Println(synthesizerResult.FinalOutput)
}
