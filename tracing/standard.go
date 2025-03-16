// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package tracing

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
)

// StandardTracer is the standard implementation of a Tracer
type StandardTracer struct {
	processors []SpanProcessor
	mu         sync.Mutex
}

// StandardSpan is the standard implementation of a Span
type StandardSpan struct {
	tracer     *StandardTracer
	ctx        *SpanContext
	events     []SpanEvent
	attributes map[string]any
	mu         sync.Mutex
	completed  bool
}

// NewStandardTracer creates a new StandardTracer
func NewStandardTracer(processors ...SpanProcessor) *StandardTracer {
	return &StandardTracer{
		processors: processors,
	}
}

// AddProcessor adds a processor to the tracer
func (t *StandardTracer) AddProcessor(processor SpanProcessor) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.processors = append(t.processors, processor)
}

// StartSpan starts a new span
func (t *StandardTracer) StartSpan(ctx context.Context, name string, attributes map[string]any) (Span, context.Context) {
	// Get parent span from context if available
	var parentSpanID string
	if parentSpan := SpanFromContext(ctx); parentSpan != nil {
		parentSpanID = parentSpan.Context().SpanID
	}

	// Create new span context
	spanContext := &SpanContext{
		TraceID:      getTraceIDFromContext(ctx),
		SpanID:       uuid.New().String(),
		ParentSpanID: parentSpanID,
		Name:         name,
		StartTime:    time.Now().UTC(),
		Attributes:   make(map[string]any),
	}

	// Create span
	span := &StandardSpan{
		tracer:     t,
		ctx:        spanContext,
		events:     make([]SpanEvent, 0),
		attributes: make(map[string]any),
	}

	// Add attributes
	if attributes != nil {
		// コンテキストの属性にも直接設定
		for k, v := range attributes {
			spanContext.Attributes[k] = v
		}
		span.SetAttributes(attributes)
	}

	// Notify processors
	for _, processor := range t.processors {
		processor.OnStart(span)
	}

	// Add span to context
	newCtx := ContextWithSpan(ctx, span)

	return span, newCtx
}

// Close cleans up the tracer
func (t *StandardTracer) Close(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Shutdown all processors
	var firstErr error
	for _, processor := range t.processors {
		if err := processor.Shutdown(ctx); err != nil {
			if firstErr == nil {
				firstErr = err
			}
			// Log error but continue shutting down other processors
			fmt.Printf("Error shutting down processor: %v\n", err)
		}
	}

	return firstErr
}

// getTraceIDFromContext gets the trace ID from the context or creates a new one
func getTraceIDFromContext(ctx context.Context) string {
	// Try to get trace ID from parent span
	if parentSpan := SpanFromContext(ctx); parentSpan != nil {
		return parentSpan.Context().TraceID
	}

	// Create new trace ID
	return uuid.New().String()
}

// Start starts the span
func (s *StandardSpan) Start() Span {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ctx.StartTime = time.Now().UTC()
	return s
}

// End ends the span
func (s *StandardSpan) End() {
	s.mu.Lock()

	if s.completed {
		s.mu.Unlock()
		return
	}

	// Set end time if not already set
	if s.ctx.EndTime.IsZero() {
		s.ctx.EndTime = time.Now().UTC()
	}

	// Apply attributes
	for k, v := range s.attributes {
		s.ctx.Attributes[k] = v
	}

	s.completed = true
	s.mu.Unlock()

	// Notify processors
	for _, processor := range s.tracer.processors {
		processor.OnEnd(s)
	}

	// Notify global processors
	NotifySpanEnded(s)
}

// AddEvent adds an event to the span
func (s *StandardSpan) AddEvent(name string, attributes map[string]any) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.completed {
		return
	}

	event := SpanEvent{
		Name:       name,
		Timestamp:  time.Now().UTC(),
		Attributes: attributes,
	}
	s.events = append(s.events, event)
}

// SetAttribute sets an attribute on the span
func (s *StandardSpan) SetAttribute(key string, value any) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.completed {
		return
	}

	s.attributes[key] = value
}

// SetAttributes sets multiple attributes on the span
func (s *StandardSpan) SetAttributes(attributes map[string]any) {
	if attributes == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.completed {
		return
	}

	for k, v := range attributes {
		s.attributes[k] = v
	}
}

// Context returns the span context
func (s *StandardSpan) Context() *SpanContext {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.ctx
}
