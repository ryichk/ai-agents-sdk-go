# AI Agents SDK for Go

A lightweight yet powerful framework for building multi-agent workflows in Go.

This project is a Go language implementation inspired by [OpenAI's Agents SDK for Python](https://github.com/openai/openai-agents-python), which is licensed under the MIT License.

### Core concepts:

1. **Agents**: LLMs configured with instructions, tools, guardrails, and handoffs
2. **Handoffs**: Allow agents to transfer control to other agents for specific tasks
3. **Guardrails**: Configurable safety checks for input and output validation
4. **Tracing**: Built-in tracking of agent runs, allowing you to view, debug and optimize your workflows

Explore the [examples](examples) directory to see the SDK in action.

This SDK is compatible with any model providers that support the OpenAI Chat Completions API format.

## Get started

1. Set up your Go environment (Go 1.24+ required)

2. Install ai-agents-sdk-go

```bash
go get github.com/ryichk/ai-agents-sdk-go
```

## Hello world example

```go
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
	// Create a new agent
	myAgent := agent.New(
		"Assistant",
		"You are a helpful assistant",
	)

	// Create a new provider
	provider, err := model.NewDefaultOpenAIProvider()
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	// Run the agent
	result, err := runner.RunWithConfig(ctx, myAgent, "Write a haiku about recursion in programming.", runner.RunConfig{
		ModelProvider: provider,
	})
	if err != nil {
		log.Fatal(err)
	}

	// Print the result
	fmt.Println(result.FinalOutput)
	// Expected output:
	// Code within the code,
	// Functions calling themselves,
	// Infinite loop's dance.
}
```

(_Ensure you set the `OPENAI_API_KEY` environment variable_)

## Handoffs example

```go
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
	// Create a new provider
	provider, err := model.NewDefaultOpenAIProvider()
	if err != nil {
		log.Fatal(err)
	}

	// Create agents
	spanishAgent := agent.New(
		"Spanish agent",
		"You only speak Spanish.",
	)

	englishAgent := agent.New(
		"English agent",
		"You only speak English",
	)

	// Create triage agent with handoffs
	triageAgent := agent.New(
		"Triage agent",
		"Handoff to the appropriate agent based on the language of the request.",
	)

	// Add handoffs
	spanishHandoff := handoff.NewHandoff(spanishAgent, "Handoff to Spanish agent")
	englishHandoff := handoff.NewHandoff(englishAgent, "Handoff to English agent") 
	triageAgent.AddHandoff(spanishHandoff)
	triageAgent.AddHandoff(englishHandoff)

	// Create context
	ctx := context.Background()

	// Run the agent
	result, err := runner.RunWithConfig(ctx, triageAgent, "Hola, ¿cómo estás?", runner.RunConfig{
		ModelProvider: provider,
	})
	if err != nil {
		log.Fatal(err)
	}

	// Print the result
	fmt.Println(result.FinalOutput)
	// Expected output:
	// ¡Hola! Estoy bien, gracias por preguntar. ¿Y tú, cómo estás?
}
```

## Functions example

```go
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

// Define a function tool
func getWeather(city string) string {
	return fmt.Sprintf("The weather in %s is sunny.", city)
}

func main() {
	// Create a new provider
	provider, err := model.NewDefaultOpenAIProvider()
	if err != nil {
		log.Fatal(err)
	}

	// Create a new agent with a tool
	weatherAgent := agent.New(
		"Weather agent",
		"You are a helpful agent.",
	)

	// Register the function as a tool
	weatherTool, err := tool.NewFunctionTool(getWeather, "Gets weather information for a city")
	if err != nil {
		log.Fatal(err)
	}
	weatherAgent.AddTool(weatherTool)

	// Run the agent
	ctx := context.Background()
	result, err := runner.RunWithConfig(ctx, weatherAgent, "What's the weather in Tokyo?", runner.RunConfig{
		ModelProvider: provider,
	})
	if err != nil {
		log.Fatal(err)
	}

	// Print the result
	fmt.Println(result.FinalOutput)
	// Expected output:
	// The weather in Tokyo is sunny.
}
```

## The agent loop

When you call `runner.Run()`, we run a loop until we get a final output.

1. We call the LLM, using the model and settings on the agent, and the message history.
2. The LLM returns a response, which may include tool calls.
3. If the response has a final output (see below for more on this), we return it and end the loop.
4. If the response has a handoff, we set the agent to the new agent and go back to step 1.
5. We process the tool calls (if any) and append the tool responses messages. Then we go to step 1.

There is a `maxTurns` parameter that you can use to limit the number of times the loop executes.

### Final output

Final output is the last thing the agent produces in the loop.

1. If you set an `outputType` on the agent, the final output is when the LLM returns something of that type. We use structured outputs for this.
2. If there's no `outputType` (i.e. plain text responses), then the first LLM response without any tool calls or handoffs is considered as the final output.

## Tracing

The Agents SDK automatically traces your agent runs, making it easy to track and debug the behavior of your agents. Tracing is extensible by design, supporting custom spans and a wide variety of external destinations.

### OpenAI Tracing Service

You can send your traces to OpenAI's tracing service for visualization and analysis:

```go
// Create OpenAI exporter
exporter, err := tracing.NewOpenAIExporter(tracing.OpenAIExporterOptions{
    APIKey:    os.Getenv("OPENAI_API_KEY"),
    BackupDir: "./trace_backups", // Optional backup directory for failed API calls
})
if err != nil {
    log.Fatalf("Failed to create OpenAI exporter: %v", err)
}

// Create batch processor
processor := tracing.NewBatchSpanProcessor(exporter, 
    tracing.WithBatchSize(100),
    tracing.WithScheduledDelay(5 * time.Second))

// Register processor
tracing.RegisterSpanProcessor(processor)

// Initialize tracer
tracer, err := tracing.NewStandardTracer("")
if err != nil {
    log.Fatalf("Failed to initialize tracer: %v", err)
}
tracing.SetGlobalTracer(tracer)

// Clean up on exit
defer func() {
    ctx, cancel := context.WithTimeout(context.Background(), 5 * time.Second)
    defer cancel()
    processor.Shutdown(ctx)
}()
```

See the [examples/openai_tracing](examples/openai_tracing) directory for a complete example.

## Development

1. Clone the repository
```bash
git clone https://github.com/ryichk/ai-agents-sdk-go.git
cd ai-agents-sdk-go
```

2. Install dependencies
```bash
go mod tidy
```

3. Install development tools
```bash
make install-tools
```

4. Run tests
```bash
make test
```

### Development tools

This project includes a `Makefile` with common development tasks:

| Command | Description |
|---------|-------------|
| `make build` | Build the project |
| `make test`  | Run all tests |
| `make lint`  | Run golangci-lint |
| `make vet`   | Run go vet |
| `make fmt`   | Format code with go fmt |
| `make tidy`  | Run go mod tidy |
| `make cover` | Generate test coverage report |
| `make check` | Run fmt, vet, lint and tests |
| `make clean` | Clean build artifacts |
| `make bench` | Run benchmarks |

#### Linting

The project uses [golangci-lint](https://golangci-lint.run/) for code quality checks. Configuration is in `.golangci.yml`.

To run the linter:
```bash
make lint
```

## Acknowledgements

This project is inspired by [OpenAI's Agents SDK for Python](https://github.com/openai/openai-agents-python), which is licensed under the MIT License.

We'd like to acknowledge the excellent work of the open-source community, especially:

- [OpenAI Go](https://github.com/openai/openai-go) for OpenAI API bindings
- [Go language](https://golang.org) and its ecosystem

We're committed to continuing to build the Agents SDK as an open source framework so others in the community can expand on our approach.
