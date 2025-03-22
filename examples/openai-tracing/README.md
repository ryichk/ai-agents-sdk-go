# OpenAI Tracing Example

This sample demonstrates how to trace agent execution using the OpenAI Tracing API.

## Overview

This sample implements the following features:

1. Initialization and configuration of OpenAI tracing
2. Tracing using custom agent hooks
3. Tracing tool calls
4. Sending trace data to OpenAI

## How to Run

1. Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=your_api_key_here
```

2. Run the sample:

```bash
go run main.go
```

## Code Explanation

### Initializing Tracing

```go
tracingConfig := &tracing.TracingConfig{
    Enabled:    true,
    TraceLevel: tracing.TraceLevelDetailed,
    OpenAI: &tracing.OpenAITracingConfig{
        Enabled:   true,
        APIKey:    apiKey,
        BackupDir: "./trace_backups",
    },
    BatchSize:      50,
    ExportInterval: 5 * time.Second,
}

if err := tracing.InitTracing(tracingConfig); err != nil {
    log.Fatalf("Failed to initialize tracing: %v", err)
}
```

### Custom Tracing Hooks

```go
type TracingHooks struct {
    agent.BaseAgentHooks
}

func (h *TracingHooks) OnStart(ctx context.Context, a *agent.Agent) error {
    _, ctx = tracing.StartSpan(ctx, "agent_run", map[string]any{
        "span_type":  string(tracing.SpanTypeAgent),
        "agent_name": a.Name,
    })
    return nil
}
```

### Visualizing Traces

Using the OpenAI Tracing API allows you to visualize the execution flow of agents. Trace data can be viewed in the OpenAI dashboard.

## Notes

- A valid OpenAI API key is required to run this sample.
- Trace data is sent to OpenAI servers. Be careful when tracing data that contains sensitive information.
- If you specify a backup directory, local copies of trace data will be saved.
