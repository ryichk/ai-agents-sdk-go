// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package tracing

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"sync"
	"time"
)

// Define keys for span context
type contextKey string

const (
	// spanKey is the key for the span in the context
	spanKey contextKey = "openai-trace-span"
)

var (
	// globalTracer is the global tracer instance
	globalTracer Tracer = &NoopTracer{}

	// globalProcessors is the list of global span processors
	spanProcessors []SpanProcessor

	// globalMutex is a mutex for concurrent access to global state
	globalMutex sync.RWMutex
)

// NoopTracer is a tracer that does nothing
type NoopTracer struct{}

// StartSpan starts a noop span
func (t *NoopTracer) StartSpan(ctx context.Context, name string, attributes map[string]any) (Span, context.Context) {
	span := &NoopSpan{}
	return span, ctx
}

// Close does nothing for NoopTracer
func (t *NoopTracer) Close(ctx context.Context) error {
	return nil
}

// NoopSpan is a span that does nothing
type NoopSpan struct{}

// Start does nothing for NoopSpan
func (s *NoopSpan) Start() Span {
	return s
}

// End does nothing for NoopSpan
func (s *NoopSpan) End() {}

// AddEvent does nothing for NoopSpan
func (s *NoopSpan) AddEvent(name string, attributes map[string]any) {}

// SetAttribute does nothing for NoopSpan
func (s *NoopSpan) SetAttribute(key string, value any) {}

// SetAttributes does nothing for NoopSpan
func (s *NoopSpan) SetAttributes(attributes map[string]any) {}

// Context returns a dummy context for NoopSpan
func (s *NoopSpan) Context() *SpanContext {
	return &SpanContext{
		TraceID:    "",
		SpanID:     "",
		StartTime:  time.Now(),
		Attributes: make(map[string]any),
	}
}

// GetTracer returns the global tracer
func GetTracer() Tracer {
	globalMutex.RLock()
	defer globalMutex.RUnlock()
	return globalTracer
}

// SetTracer sets the global tracer
func SetTracer(tracer Tracer) {
	globalMutex.Lock()
	defer globalMutex.Unlock()
	globalTracer = tracer
}

// AddGlobalProcessor adds a span processor to the global list
func AddGlobalProcessor(processor SpanProcessor) {
	globalMutex.Lock()
	defer globalMutex.Unlock()
	spanProcessors = append(spanProcessors, processor)
}

// NotifySpanEnded notifies all global processors that a span has ended
func NotifySpanEnded(span *StandardSpan) {
	globalMutex.RLock()
	defer globalMutex.RUnlock()
	for _, processor := range spanProcessors {
		processor.OnEnd(span)
	}
}

// ContextWithSpan adds a span to the context
func ContextWithSpan(ctx context.Context, span Span) context.Context {
	return context.WithValue(ctx, spanKey, span)
}

// SpanFromContext gets a span from the context
func SpanFromContext(ctx context.Context) Span {
	val := ctx.Value(spanKey)
	if val == nil {
		return nil
	}
	if span, ok := val.(Span); ok {
		return span
	}
	return nil
}

// StartSpan starts a span with the global tracer
func StartSpan(ctx context.Context, name string, attributes map[string]any) (Span, context.Context) {
	return GetTracer().StartSpan(ctx, name, attributes)
}

// GetActiveSpan gets the active span from the context
func GetActiveSpan(ctx context.Context) Span {
	return SpanFromContext(ctx)
}

// Config contains configuration for tracing
type Config struct {
	// Enabled indicates whether tracing is enabled
	Enabled bool

	// TraceLevel is the level of tracing to capture
	TraceLevel TraceLevel

	// OpenAI contains configuration for OpenAI tracing
	OpenAI *OpenAITracingConfig

	// BatchSize is the batch size for the processor
	BatchSize int

	// ExportInterval is the export interval for the processor
	ExportInterval time.Duration
}

// OpenAITracingConfig contains configuration for OpenAI tracing
type OpenAITracingConfig struct {
	// Enabled indicates whether OpenAI tracing is enabled
	Enabled bool

	// APIKey is the OpenAI API key
	APIKey string

	// Endpoint is the OpenAI trace API endpoint
	Endpoint string

	// BackupDir is the directory to save traces if API requests fail
	BackupDir string
}

// DefaultConfig returns the default tracing configuration
func DefaultConfig() *Config {
	return &Config{
		Enabled:        false,
		TraceLevel:     TraceLevelBasic,
		BatchSize:      100,
		ExportInterval: 5 * time.Second,
	}
}

// InitTracing initializes tracing
func InitTracing(config *Config) error {
	if config == nil {
		config = DefaultConfig()
	}

	// If tracing is not enabled, use noop tracer
	if !config.Enabled {
		SetTracer(&NoopTracer{})
		return nil
	}

	processors := make([]SpanProcessor, 0)

	// Add OpenAI processor if enabled
	if config.OpenAI != nil && config.OpenAI.Enabled {
		exporter, err := NewOpenAIExporter(OpenAIExporterOptions{
			APIKey:    config.OpenAI.APIKey,
			Endpoint:  config.OpenAI.Endpoint,
			BackupDir: config.OpenAI.BackupDir,
		})
		if err != nil {
			return fmt.Errorf("failed to create OpenAI exporter: %w", err)
		}

		processor := NewBatchSpanProcessor(exporter,
			WithBatchSize(config.BatchSize),
			WithExportInterval(config.ExportInterval),
		)
		processors = append(processors, processor)
	}

	// Create tracer with processors
	tracer := NewStandardTracer(processors...)
	SetTracer(tracer)

	return nil
}

// ShutdownTracing shuts down tracing
func ShutdownTracing(ctx context.Context) error {
	globalMutex.Lock()
	defer globalMutex.Unlock()

	// Get current tracer
	tracer := globalTracer

	// Reset global tracer to noop
	globalTracer = &NoopTracer{}

	// Clear processors
	for _, processor := range spanProcessors {
		_ = processor.Shutdown(ctx)
	}
	spanProcessors = nil

	// Shutdown tracer
	return tracer.Close(ctx)
}

// InitFromEnv initializes tracing from environment variables
func InitFromEnv() error {
	config := DefaultConfig()

	// Check if tracing is enabled
	if enabled, _ := strconv.ParseBool(os.Getenv("OPENAI_TRACING_ENABLED")); enabled {
		config.Enabled = true

		// Parse trace level
		levelStr := os.Getenv("OPENAI_TRACE_LEVEL")
		switch levelStr {
		case "none":
			config.TraceLevel = TraceLevelNone
		case "basic":
			config.TraceLevel = TraceLevelBasic
		case "detailed":
			config.TraceLevel = TraceLevelDetailed
		case "all":
			config.TraceLevel = TraceLevelAll
		}

		// Check if OpenAI tracing is enabled
		if enabled, _ := strconv.ParseBool(os.Getenv("OPENAI_TRACE_EXPORT_ENABLED")); enabled {
			config.OpenAI = &OpenAITracingConfig{
				Enabled:   true,
				APIKey:    os.Getenv("OPENAI_API_KEY"),
				Endpoint:  os.Getenv("OPENAI_TRACE_ENDPOINT"),
				BackupDir: os.Getenv("OPENAI_TRACE_BACKUP_DIR"),
			}
		}

		// Parse batch size
		if batchSizeStr := os.Getenv("OPENAI_TRACE_BATCH_SIZE"); batchSizeStr != "" {
			if batchSize, err := strconv.Atoi(batchSizeStr); err == nil && batchSize > 0 {
				config.BatchSize = batchSize
			}
		}

		// Parse export interval
		if intervalStr := os.Getenv("OPENAI_TRACE_EXPORT_INTERVAL"); intervalStr != "" {
			if interval, err := strconv.Atoi(intervalStr); err == nil && interval > 0 {
				config.ExportInterval = time.Duration(interval) * time.Second
			}
		}
	}

	return InitTracing(config)
}

// EnableTracing initializes tracing with local backup
func EnableTracing(apiKey string, backupDir string, batchSize int, exportInterval time.Duration) error {
	// Try to create OpenAI exporter if API key is provided
	if apiKey == "" {
		return fmt.Errorf("OpenAI API key is required")
	}

	// Use "./traces" as the default backup dir if not specified
	if backupDir == "" {
		backupDir = "./traces"
	}

	openAIExporter, err := NewOpenAIExporter(OpenAIExporterOptions{
		APIKey:    apiKey,
		BackupDir: backupDir,
	})
	if err != nil {
		return fmt.Errorf("failed to create OpenAI exporter: %w", err)
	}

	// Create batch processor
	processor := NewBatchSpanProcessor(openAIExporter,
		WithBatchSize(batchSize),
		WithExportInterval(exportInterval),
		WithBackupDir(backupDir),
	)

	// Register processor
	globalMutex.Lock()
	defer globalMutex.Unlock()

	spanProcessors = append(spanProcessors, processor)

	// Initialize tracer
	tracer := NewStandardTracer(processor)

	// Set as global tracer
	globalTracer = tracer

	return nil
}

// MultiExporter sends spans to multiple exporters
type MultiExporter struct {
	exporters []SpanExporter
}

// ExportSpan exports a span to all configured exporters
func (e *MultiExporter) ExportSpan(ctx context.Context, span *StandardSpan) error {
	var firstError error

	for _, exporter := range e.exporters {
		if err := exporter.ExportSpan(ctx, span); err != nil {
			if firstError == nil {
				firstError = err
			}
			// Log but continue to other exporters
			fmt.Printf("[ERROR] Failed to export span with exporter %T: %v\n", exporter, err)
		}
	}

	return firstError
}

// ExportSpans exports spans to all configured exporters
func (e *MultiExporter) ExportSpans(ctx context.Context, spans []*StandardSpan) error {
	var firstError error

	for _, exporter := range e.exporters {
		if err := exporter.ExportSpans(ctx, spans); err != nil {
			if firstError == nil {
				firstError = err
			}
			// Log but continue to other exporters
			fmt.Printf("[ERROR] Failed to export spans with exporter %T: %v\n", exporter, err)
		}
	}

	return firstError
}

// Shutdown closes all exporters
func (e *MultiExporter) Shutdown(ctx context.Context) error {
	var firstError error

	for _, exporter := range e.exporters {
		if err := exporter.Shutdown(ctx); err != nil {
			if firstError == nil {
				firstError = err
			}
			// Log but continue shutting down other exporters
			fmt.Printf("[ERROR] Failed to shutdown exporter %T: %v\n", exporter, err)
		}
	}

	return firstError
}
