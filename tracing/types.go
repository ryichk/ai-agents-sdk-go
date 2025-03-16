// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package tracing

import (
	"context"
	"time"
)

// TraceLevel represents the level of tracing to capture
type TraceLevel int

const (
	// TraceLevelNone means no tracing
	TraceLevelNone TraceLevel = iota

	// TraceLevelBasic captures basic spans
	TraceLevelBasic

	// TraceLevelDetailed captures detailed spans
	TraceLevelDetailed

	// TraceLevelAll captures all spans
	TraceLevelAll
)

// OpenAISpanType represents the type of span for OpenAI's tracing API
type OpenAISpanType string

const (
	// SpanTypeAgent represents an agent span
	SpanTypeAgent OpenAISpanType = "agent"

	// SpanTypeTool represents a tool span
	SpanTypeTool OpenAISpanType = "tool"

	// SpanTypeModel represents a model span
	SpanTypeModel OpenAISpanType = "model"

	// SpanTypeHandoff represents a handoff span
	SpanTypeHandoff OpenAISpanType = "handoff"

	// SpanTypeRunner represents a runner span
	SpanTypeRunner OpenAISpanType = "runner"
)

// SpanContext contains the context of a span
type SpanContext struct {
	// TraceID is the ID of the trace
	TraceID string

	// SpanID is the ID of the span
	SpanID string

	// ParentSpanID is the ID of the parent span
	ParentSpanID string

	// Name is the name of the span
	Name string

	// StartTime is when the span started
	StartTime time.Time

	// EndTime is when the span ended
	EndTime time.Time

	// Attributes are the span attributes
	Attributes map[string]any
}

// SpanEvent represents an event in a span
type SpanEvent struct {
	// Name is the name of the event
	Name string

	// Timestamp is when the event occurred
	Timestamp time.Time

	// Attributes are the event attributes
	Attributes map[string]any
}

// Span represents a span in a trace
type Span interface {
	// Start starts the span and returns the span
	Start() Span

	// End ends the span
	End()

	// AddEvent adds an event to the span
	AddEvent(name string, attributes map[string]any)

	// SetAttribute sets an attribute on the span
	SetAttribute(key string, value any)

	// SetAttributes sets multiple attributes on the span
	SetAttributes(attributes map[string]any)

	// Context returns the span context
	Context() *SpanContext
}

// Tracer is an interface for creating spans
type Tracer interface {
	// StartSpan starts a span with the given name and parents
	StartSpan(ctx context.Context, name string, attributes map[string]any) (Span, context.Context)

	// Close cleans up the tracer
	Close(ctx context.Context) error
}

// SpanProcessor handles span processing
type SpanProcessor interface {
	// OnStart is called when a span starts
	OnStart(span *StandardSpan)

	// OnEnd is called when a span ends
	OnEnd(span *StandardSpan)

	// ForceFlush forces the processor to process all spans
	ForceFlush()

	// Shutdown shuts down the processor
	Shutdown(ctx context.Context) error
}
