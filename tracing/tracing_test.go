// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package tracing

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNoopTracer(t *testing.T) {
	tracer := &NoopTracer{}

	span, ctx := tracer.StartSpan(context.Background(), "test_span", map[string]any{
		"test_attr": "test_value",
	})

	// Add events and attributes to the span
	span.AddEvent("test_event", map[string]any{
		"event_attr": "event_value",
	})
	span.SetAttribute("attr_key", "attr_value")

	// End the span (verify no error occurs)
	span.End()

	// Shutdown the tracer
	err := tracer.Close(ctx)
	assert.NoError(t, err)
}

// TestStandardSpanCreation tests span creation and attribute setting
func TestStandardSpanCreation(t *testing.T) {
	tracer := NewStandardTracer()

	span, ctx := tracer.StartSpan(context.Background(), "test_span", map[string]any{
		"initial_attr": "initial_value",
	})

	// Type assertion
	stdSpan, ok := span.(*StandardSpan)
	require.True(t, ok, "span should be of type *StandardSpan")

	// Basic validation of span context
	assert.Equal(t, "test_span", stdSpan.ctx.Name)
	assert.NotEmpty(t, stdSpan.ctx.TraceID)
	assert.NotEmpty(t, stdSpan.ctx.SpanID)
	assert.Equal(t, "initial_value", stdSpan.ctx.Attributes["initial_attr"])

	retrievedSpan := SpanFromContext(ctx)
	assert.Equal(t, span, retrievedSpan)

	span.AddEvent("test_event", map[string]any{
		"event_data": 123,
	})

	span.SetAttribute("string_attr", "string_value")
	span.SetAttribute("int_attr", 42)
	span.SetAttribute("bool_attr", true)

	span.SetAttributes(map[string]any{
		"batch_attr1": "value1",
		"batch_attr2": "value2",
	})

	span.End()

	// Check the state of the ended span
	assert.True(t, stdSpan.completed)
	assert.NotZero(t, stdSpan.ctx.EndTime)

	err := tracer.Close(context.Background())
	assert.NoError(t, err)
}

func TestSpanHierarchy(t *testing.T) {
	tracer := NewStandardTracer()

	parentSpan, ctx := tracer.StartSpan(context.Background(), "parent_span", nil)
	parentStdSpan := parentSpan.(*StandardSpan)

	childSpan, childCtx := tracer.StartSpan(ctx, "child_span", nil)
	childStdSpan := childSpan.(*StandardSpan)

	grandchildSpan, _ := tracer.StartSpan(childCtx, "grandchild_span", nil)
	grandchildStdSpan := grandchildSpan.(*StandardSpan)

	// Verify parent-child relationships
	assert.Equal(t, parentStdSpan.ctx.SpanID, childStdSpan.ctx.ParentSpanID)
	assert.Equal(t, childStdSpan.ctx.SpanID, grandchildStdSpan.ctx.ParentSpanID)
	assert.Equal(t, parentStdSpan.ctx.TraceID, childStdSpan.ctx.TraceID)
	assert.Equal(t, childStdSpan.ctx.TraceID, grandchildStdSpan.ctx.TraceID)

	grandchildSpan.End()
	childSpan.End()
	parentSpan.End()

	err := tracer.Close(context.Background())
	assert.NoError(t, err)
}

type MockExporter struct {
	mu             sync.Mutex
	exportedSpans  []*StandardSpan
	ShutdownCalled bool
}

func (e *MockExporter) ExportSpan(ctx context.Context, span *StandardSpan) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.exportedSpans = append(e.exportedSpans, span)
	return nil
}

func (e *MockExporter) ExportSpans(ctx context.Context, spans []*StandardSpan) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.exportedSpans = append(e.exportedSpans, spans...)
	return nil
}

// GetExportedSpans returns a copy of the exported spans in a thread-safe manner
func (e *MockExporter) GetExportedSpans() []*StandardSpan {
	e.mu.Lock()
	defer e.mu.Unlock()
	result := make([]*StandardSpan, len(e.exportedSpans))
	copy(result, e.exportedSpans)
	return result
}

// GetExportedSpansCount returns the count of exported spans in a thread-safe manner
func (e *MockExporter) GetExportedSpansCount() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.exportedSpans)
}

func (e *MockExporter) Shutdown(ctx context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.ShutdownCalled = true
	return nil
}

func TestBatchProcessor(t *testing.T) {
	mockExporter := &MockExporter{
		exportedSpans: make([]*StandardSpan, 0),
	}

	// Create a batch processor (with small batch size and short interval)
	processor := NewBatchSpanProcessor(mockExporter,
		WithBatchSize(2),
		WithExportInterval(100*time.Millisecond),
	)

	tracer := NewStandardTracer(processor)

	// Create multiple spans and end them
	span1, _ := tracer.StartSpan(context.Background(), "span1", nil)
	span1.End()

	span2, _ := tracer.StartSpan(context.Background(), "span2", nil)
	span2.End()

	// Wait a bit for processing and then force flush to ensure export
	time.Sleep(50 * time.Millisecond)
	processor.ForceFlush()
	assert.Equal(t, 2, mockExporter.GetExportedSpansCount(), "Expected 2 spans to be exported")

	span3, _ := tracer.StartSpan(context.Background(), "span3", nil)
	span3.End()

	// Wait for export interval and force flush again
	time.Sleep(200 * time.Millisecond)
	processor.ForceFlush()
	assert.Equal(t, 3, mockExporter.GetExportedSpansCount(), "Expected 3 spans to be exported")

	// Shutdown processor and tracer
	err := processor.Shutdown(context.Background())
	assert.NoError(t, err)
	assert.True(t, mockExporter.ShutdownCalled)

	err = tracer.Close(context.Background())
	assert.NoError(t, err)
}

// TestGlobalTracer tests setting and getting the global tracer
func TestGlobalTracer(t *testing.T) {
	// Initial state should be NoopTracer
	tracer := GetTracer()
	_, ok := tracer.(*NoopTracer)
	assert.True(t, ok, "Default global tracer should be NoopTracer")

	// Set a custom tracer
	customTracer := NewStandardTracer()
	SetTracer(customTracer)

	// Verify the set tracer can be retrieved
	retrieved := GetTracer()
	assert.Equal(t, customTracer, retrieved)

	// Test adding a global span processor
	mockExporter := &MockExporter{
		exportedSpans: make([]*StandardSpan, 0),
	}
	processor := NewBatchSpanProcessor(mockExporter, WithBatchSize(1))
	AddGlobalProcessor(processor)

	// Create a span using the global processor
	span, _ := customTracer.StartSpan(context.Background(), "global_processed", nil)
	span.End()

	// Verify the span was exported (need to wait a bit)
	time.Sleep(200 * time.Millisecond)
	assert.GreaterOrEqual(t, mockExporter.GetExportedSpansCount(), 1, "Expected at least 1 span to be exported")

	// Cleanup
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	err := processor.Shutdown(ctx)
	assert.NoError(t, err)
	SetTracer(&NoopTracer{})

	// Shutdown global tracer
	err = ShutdownTracing(ctx)
	assert.NoError(t, err)
}

// TestEnvironmentVariables tests configuration from environment variables
func TestEnvironmentVariables(t *testing.T) {
	os.Setenv("OPENAI_TRACING_ENABLED", "true")
	os.Setenv("OPENAI_TRACE_LEVEL", "basic")
	os.Setenv("OPENAI_TRACE_BATCH_SIZE", "50")
	os.Setenv("OPENAI_TRACE_EXPORT_INTERVAL", "2")

	// Load configuration from environment variables
	config := DefaultConfig()
	assert.False(t, config.Enabled)

	// Initialize from environment variables (in this case, to avoid actually calling the API)
	os.Setenv("OPENAI_TRACE_EXPORT_ENABLED", "false")
	err := InitFromEnv()
	assert.NoError(t, err)

	// Cleanup
	err = ShutdownTracing(context.Background())
	assert.NoError(t, err)

	// Reset environment variables
	os.Unsetenv("OPENAI_TRACING_ENABLED")
	os.Unsetenv("OPENAI_TRACE_LEVEL")
	os.Unsetenv("OPENAI_TRACE_BATCH_SIZE")
	os.Unsetenv("OPENAI_TRACE_EXPORT_INTERVAL")
	os.Unsetenv("OPENAI_TRACE_EXPORT_ENABLED")
}

// TestOpenAIExporter tests the OpenAI exporter (mocked)
func TestOpenAIExporter(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		os.Setenv("OPENAI_API_KEY", "test_api_key") // Temporarily set for testing
		defer os.Unsetenv("OPENAI_API_KEY")
		apiKey = "test_api_key"
	}

	// Test with API key - NewOpenAIExporter should return nil error
	exporter, err := NewOpenAIExporter(OpenAIExporterOptions{
		APIKey: apiKey,
	})
	assert.NoError(t, err)
	assert.NotNil(t, exporter)

	err = exporter.Shutdown(context.Background())
	assert.NoError(t, err)
}

func TestSpanTypeConversion(t *testing.T) {
	tracer := NewStandardTracer()

	// Test with each span type
	spanTypes := []OpenAISpanType{
		SpanTypeAgent,
		SpanTypeTool,
		SpanTypeModel,
		SpanTypeHandoff,
		SpanTypeRunner,
	}

	for _, spanType := range spanTypes {
		span, _ := tracer.StartSpan(context.Background(), "span_"+string(spanType), map[string]any{
			"span_type": string(spanType),
		})

		stdSpan := span.(*StandardSpan)
		assert.Equal(t, string(spanType), stdSpan.ctx.Attributes["span_type"])
		span.End()
	}

	err := tracer.Close(context.Background())
	assert.NoError(t, err)
}

// TestTracingIntegration tests integration of tracing components
func TestTracingIntegration(t *testing.T) {
	// Save the original global tracer to restore it later
	originalTracer := GetTracer()

	// Create a mock exporter and processor
	mockExporter := &MockExporter{
		exportedSpans: make([]*StandardSpan, 0),
	}

	processor := NewBatchSpanProcessor(mockExporter, WithBatchSize(1))
	tracer := NewStandardTracer(processor)

	// Set up cleanup to properly shutdown resources at the end of the test
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
		defer cancel()

		err := processor.Shutdown(ctx)
		if err != nil {
			t.Logf("Error shutting down processor: %v", err)
		}

		// Restore the original tracer
		SetTracer(originalTracer)
	})

	// Set the tracer for this test
	SetTracer(tracer)

	// Setup test scenario
	ctx := context.Background()
	agentCtx, agentSpan := TraceAgentRun(ctx, string(SpanTypeAgent), "test-agent", "Test input")

	// Run an agent step
	stepCtx, stepSpan := TraceAgentStep(agentCtx, 1, []Message{
		{"role": "user", "content": "hello"},
	})
	stepSpan.End()

	// Trace a tool call
	toolCtx, toolSpan := TraceToolCall(stepCtx, "test-tool", "call-123", `{"param":"value"}`)
	toolSpan.SetAttribute("custom_attr", "custom_value")
	TraceToolCallEnd(toolCtx, "Tool result", nil)

	// Trace a handoff
	handoffCtx, handoffSpan := TraceHandoff(stepCtx, "agent-1", "agent-2", "Handoff reason")
	TraceHandoffEnd(handoffCtx, true, nil)
	// Make sure the handoff span is used
	handoffSpan.SetAttribute("test_attribute", "test_value")

	// End the agent run
	agentSpan.SetAttribute("agent_id", "test-agent")
	agentSpan.SetAttribute("final_result", "success")
	AgentRunEndEvent(agentCtx, "Final output", 100*time.Millisecond, true, nil)

	// Wait for spans to be processed - increase the timeout for race detection
	time.Sleep(300 * time.Millisecond)

	// Force flush any pending spans
	processor.ForceFlush()

	// Get a thread-safe copy of exported spans
	exportedSpans := mockExporter.GetExportedSpans()

	// Verify exported spans
	toolSpanExported := false
	handoffSpanExported := false
	agentSpanExported := false
	var exportedAgentSpan *StandardSpan

	for _, span := range exportedSpans {
		switch span.ctx.Name {
		case "tool_call":
			toolSpanExported = true
			assert.Equal(t, "test-tool", span.ctx.Attributes["tool_name"])
			assert.Equal(t, "custom_value", span.ctx.Attributes["custom_attr"])
		case "handoff":
			handoffSpanExported = true
			assert.Equal(t, "agent-1", span.ctx.Attributes["from_agent"])
			assert.Equal(t, "agent-2", span.ctx.Attributes["to_agent"])
		case "agent_run":
			agentSpanExported = true
			exportedAgentSpan = span
		}
	}

	assert.True(t, toolSpanExported, "Tool span should be exported")
	assert.True(t, handoffSpanExported, "Handoff span should be exported")
	assert.True(t, agentSpanExported, "Agent span should be exported")

	// Verify span hierarchy relationships
	if agentSpanExported {
		assert.Equal(t, "test-agent", exportedAgentSpan.ctx.Attributes["agent_id"])
		assert.Equal(t, "success", exportedAgentSpan.ctx.Attributes["final_result"])
	}
}

// Message is the type for tracing messages
type Message map[string]any

// The following are tracing-related functions used in runner.go

// TraceAgentRun starts tracing an agent execution
func TraceAgentRun(ctx context.Context, spanType string, agentName string, input string) (context.Context, Span) {
	span, newCtx := StartSpan(ctx, "agent_run", map[string]any{
		"span_type":  spanType,
		"agent_name": agentName,
		"input":      input,
	})
	return newCtx, span
}

// TraceAgentStep starts tracing an agent step
func TraceAgentStep(ctx context.Context, step int, messages []Message) (context.Context, Span) {
	messagesJSON, _ := json.Marshal(messages)
	span, newCtx := StartSpan(ctx, fmt.Sprintf("agent_step_%d", step), map[string]any{
		"span_type": string(SpanTypeAgent),
		"step":      step,
		"messages":  string(messagesJSON),
	})
	return newCtx, span
}

// AgentRunEndEvent records the end event of an agent execution
func AgentRunEndEvent(ctx context.Context, output string, duration time.Duration, success bool, err error) {
	span := GetActiveSpan(ctx)
	if span != nil {
		span.SetAttribute("output", output)
		span.SetAttribute("duration_ms", duration.Milliseconds())
		span.SetAttribute("success", success)
		if err != nil {
			span.SetAttribute("error", err.Error())
		}
		span.End()
	}
}

// TraceToolCall starts tracing a tool call
func TraceToolCall(ctx context.Context, toolName string, toolCallID string, arguments string) (context.Context, Span) {
	span, newCtx := StartSpan(ctx, "tool_call", map[string]any{
		"span_type":    string(SpanTypeTool),
		"tool_name":    toolName,
		"tool_call_id": toolCallID,
		"arguments":    arguments,
	})
	return newCtx, span
}

// TraceToolCallEnd records the end of a tool call
func TraceToolCallEnd(ctx context.Context, result string, err error) {
	span := GetActiveSpan(ctx)
	if span != nil {
		span.SetAttribute("result", result)
		if err != nil {
			span.SetAttribute("error", err.Error())
		}
		span.End()
	}
}

// TraceHandoff starts tracing a handoff
func TraceHandoff(ctx context.Context, fromAgent string, toAgent string, reason string) (context.Context, Span) {
	span, newCtx := StartSpan(ctx, "handoff", map[string]any{
		"span_type":  string(SpanTypeHandoff),
		"from_agent": fromAgent,
		"to_agent":   toAgent,
		"reason":     reason,
	})
	return newCtx, span
}

// TraceHandoffEnd records the end of a handoff
func TraceHandoffEnd(ctx context.Context, success bool, err error) {
	span := GetActiveSpan(ctx)
	if span != nil {
		span.SetAttribute("success", success)
		if err != nil {
			span.SetAttribute("error", err.Error())
		}
		span.End()
	}
}
