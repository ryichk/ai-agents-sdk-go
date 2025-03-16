# OpenAI Tracing for Go Agents SDK

This package provides tracing functionality for the Go Agents SDK. Using OpenAI's tracing API, you can track agent execution, which helps with debugging and performance analysis.

## Features

- Integration with OpenAI's tracing API
- Creation and management of spans
- Addition of events and attributes
- Efficient transmission of trace data through batch processing
- Easy configuration using environment variables

## Usage

### Initialization using Environment Variables

The easiest way is to configure tracing using environment variables:

```go
import "github.com/ryichk/ai-agents-sdk-go/tracing"

func main() {
    // Initialize tracing from environment variables
    if err := tracing.InitFromEnv(); err != nil {
        log.Fatalf("Failed to initialize tracing: %v", err)
    }
    defer tracing.ShutdownTracing(context.Background())

    // Subsequent code...
}
```

The following environment variables are available:

- `OPENAI_TRACING_ENABLED`: Enable tracing ("true" or "1")
- `OPENAI_TRACE_LEVEL`: Trace level ("none", "basic", "detailed", "all")
- `OPENAI_TRACE_EXPORT_ENABLED`: Enable export to OpenAI
- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_TRACE_ENDPOINT`: Trace endpoint (default: "https://api.openai.com/v1/traces/ingest")
- `OPENAI_TRACE_BACKUP_DIR`: Backup directory
- `OPENAI_TRACE_BATCH_SIZE`: Batch size
- `OPENAI_TRACE_EXPORT_INTERVAL`: Export interval (seconds)

### Manual Configuration

For more detailed configuration, you can set up manually:

```go
import "github.com/ryichk/ai-agents-sdk-go/tracing"

func main() {
    config := &tracing.TracingConfig{
        Enabled:    true,
        TraceLevel: tracing.TraceLevelDetailed,
        OpenAI: &tracing.OpenAITracingConfig{
            Enabled:   true,
            APIKey:    "your-api-key",
            BackupDir: "/path/to/backup",
        },
        BatchSize:      200,
        ExportInterval: 10 * time.Second,
    }

    if err := tracing.InitTracing(config); err != nil {
        log.Fatalf("Failed to initialize tracing: %v", err)
    }
    defer tracing.ShutdownTracing(context.Background())

    // Subsequent code...
}
```

### Creating and Using Spans

After initializing tracing, you can create and use spans:

```go
// Create a span
span, ctx := tracing.StartSpan(ctx, "operation_name", map[string]any{
    "key1": "value1",
    "key2": 42,
})

// Execute processing
// ...

// Add an event
span.AddEvent("something_happened", map[string]any{
    "details": "more information",
})

// Set an attribute
span.SetAttribute("result", "success")

// End the span
span.End()
```

### Integration with Agent Hooks

Tracing can be integrated with agent lifecycle hooks:

```go
// Implement tracing hooks
type TracingHooks struct{}

func (h *TracingHooks) OnStart(ctx context.Context, agent *agent.Agent, msg model.Message) context.Context {
    span, newCtx := tracing.StartSpan(ctx, "agent_run", map[string]any{
        "span_type": string(tracing.SpanTypeAgent),
        "agent_id":  agent.ID,
        "message":   msg.Content,
    })
    return newCtx
}

func (h *TracingHooks) OnEnd(ctx context.Context, agent *agent.Agent, result *runner.Result) {
    if span := tracing.GetActiveSpan(ctx); span != nil {
        span.SetAttribute("result", result.FinalMessage.Content)
        span.End()
    }
}

// Implement other hook methods similarly...

// Use hooks when running an agent
config := runner.RunConfig{
    Hooks: &TracingHooks{},
}
result, err := runner.RunWithConfig(ctx, agent, message, config)
```

## Creating Custom Exporters

You can also create your own exporters:

```go
type CustomExporter struct {
    // Fields...
}

func (e *CustomExporter) ExportSpan(ctx context.Context, span *tracing.StandardSpan) error {
    // Implementation to export a span
    return nil
}

func (e *CustomExporter) ExportSpans(ctx context.Context, spans []*tracing.StandardSpan) error {
    // Implementation to export multiple spans
    return nil
}

func (e *CustomExporter) Shutdown(ctx context.Context) error {
    // Shutdown processing
    return nil
}

// Use the exporter
exporter := &CustomExporter{}
processor := tracing.NewBatchSpanProcessor(exporter)
tracer := tracing.NewStandardTracer(processor)
tracing.SetTracer(tracer)
```

## Trace Visualization

You can visualize traces using OpenAI's tracing dashboard. For more details, please refer to OpenAI's documentation.
