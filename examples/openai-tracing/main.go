package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/ryichk/ai-agents-sdk-go/agent"
	"github.com/ryichk/ai-agents-sdk-go/model"
	"github.com/ryichk/ai-agents-sdk-go/runner"
	"github.com/ryichk/ai-agents-sdk-go/tool"
	"github.com/ryichk/ai-agents-sdk-go/tracing"
)

// TracingHooks implements agent.Hooks for tracing agent execution
type TracingHooks struct{}

// OnStart is called when an agent starts execution
func (h *TracingHooks) OnStart(ctx context.Context, a *agent.Agent) error {
	fmt.Printf("Agent %s started\n", a.Name)
	return nil
}

// OnEnd is called when an agent ends execution
func (h *TracingHooks) OnEnd(ctx context.Context, a *agent.Agent, result any) error {
	fmt.Printf("Agent %s ended with result: %v\n", a.Name, result)
	return nil
}

// OnToolStart is called when a tool starts execution
func (h *TracingHooks) OnToolStart(ctx context.Context, a *agent.Agent, t tool.Tool) error {
	fmt.Printf("Tool %s started\n", t.Name())
	return nil
}

// OnToolEnd is called when a tool ends execution
func (h *TracingHooks) OnToolEnd(ctx context.Context, a *agent.Agent, t tool.Tool, result string) error {
	fmt.Printf("Tool %s ended with result: %s\n", t.Name(), result)
	return nil
}

// OnHandoff is called when a handoff occurs
func (h *TracingHooks) OnHandoff(ctx context.Context, targetAgent *agent.Agent, sourceAgent *agent.Agent) error {
	fmt.Printf("Handoff from %s to %s\n", sourceAgent.Name, targetAgent.Name)
	return nil
}

// getWeatherCallback is a simple function that returns weather information
func getWeatherCallback(city string) string {
	return fmt.Sprintf("The weather in %s is sunny and 25°C.", city)
}

func main() {
	// Ensure API key is set
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is not set")
	}

	// Create trace directory if it doesn't exist
	if err := os.MkdirAll("./traces", 0755); err != nil {
		log.Fatalf("Failed to create trace directory: %v", err)
	}

	// Initialize tracing
	exporter, err := tracing.NewOpenAIExporter(tracing.OpenAIExporterOptions{
		APIKey:    apiKey,
		BackupDir: "./traces", // Backup directory for failed API calls
	})
	if err != nil {
		log.Fatalf("Failed to create OpenAI exporter: %v", err)
	}

	// Create batch processor
	processor := tracing.NewBatchSpanProcessor(exporter,
		tracing.WithBatchSize(10),
		tracing.WithExportInterval(5*time.Second))

	// Initialize tracer
	tracer := tracing.NewStandardTracer(processor)
	tracing.SetTracer(tracer)

	// Clean up on exit
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		processor.Shutdown(ctx)
	}()

	// Create a new provider
	provider, err := model.NewOpenAIProvider(model.OpenAIConfig{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("Failed to create OpenAI provider: %v", err)
	}

	// Create a weather tool
	weatherTool, err := tool.NewFunctionTool(
		getWeatherCallback,
		tool.FunctionToolOption{
			NameOverride:        "get_weather",
			DescriptionOverride: "Get the current weather for a city",
		},
	)
	if err != nil {
		log.Fatalf("Failed to create weather tool: %v", err)
	}

	// Create a weather agent
	weatherAgent := agent.New(
		"Weather Assistant",
		"You are a helpful assistant that provides weather information. Use the get_weather tool to check the weather for a city.",
	)
	weatherAgent.AddTool(weatherTool)
	weatherAgent.SetHooks(&TracingHooks{})

	// Run the agent
	ctx := context.Background()
	result, err := runner.RunWithConfig(ctx, weatherAgent, "What's the weather in Tokyo?", runner.RunConfig{
		ModelProvider: provider,
	})
	if err != nil {
		log.Fatalf("Failed to run agent: %v", err)
	}

	// Print the result
	fmt.Printf("\nFinal output: %s\n", result.FinalOutput)
}
