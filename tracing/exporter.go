// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package tracing

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	// DefaultOpenAITraceEndpoint is the default OpenAI trace ingest endpoint
	DefaultOpenAITraceEndpoint = "https://api.openai.com/v1/traces/ingest"

	// DefaultTimeout is the default timeout for API requests
	DefaultTimeout = 10 * time.Second

	// MaxRetries is the maximum number of retries for failed requests
	MaxRetries = 3
)

// SpanExporter is an interface for exporting traces to external systems
type SpanExporter interface {
	// ExportSpan exports a single span
	ExportSpan(ctx context.Context, span *StandardSpan) error

	// ExportSpans exports multiple spans in a batch
	ExportSpans(ctx context.Context, spans []*StandardSpan) error

	// Shutdown gracefully shuts down the exporter
	Shutdown(ctx context.Context) error
}

// OpenAIExporterOptions configures the OpenAI trace exporter
type OpenAIExporterOptions struct {
	// APIKey is the OpenAI API key (optional, falls back to OPENAI_API_KEY env var)
	APIKey string

	// Endpoint is the OpenAI trace API endpoint (optional, defaults to DefaultOpenAITraceEndpoint)
	Endpoint string

	// Timeout is the timeout for API requests (optional, defaults to DefaultTimeout)
	Timeout time.Duration

	// MaxRetries is the maximum number of retries for failed requests (optional, defaults to MaxRetries)
	MaxRetries int

	// BackupDir is the directory to save traces if API requests fail (optional)
	BackupDir string
}

// OpenAIExporter exports traces to OpenAI's tracing service
type OpenAIExporter struct {
	options OpenAIExporterOptions
	client  *http.Client
	mu      sync.Mutex
}

// OpenAISpanData is the structure of a span for OpenAI's tracing API
type OpenAISpanData struct {
	// Common fields
	Object    string         `json:"object"`
	ID        string         `json:"id"`
	TraceID   string         `json:"trace_id"`
	ParentID  string         `json:"parent_id,omitempty"`
	StartTime string         `json:"started_at"` // ISO 8601 format
	EndTime   string         `json:"ended_at"`   // ISO 8601 format
	SpanData  map[string]any `json:"span_data"`
	Error     map[string]any `json:"error,omitempty"`
}

// NewOpenAIExporter creates a new OpenAI trace exporter
func NewOpenAIExporter(options OpenAIExporterOptions) (*OpenAIExporter, error) {
	if options.APIKey == "" {
		// Try to get API key from environment
		options.APIKey = os.Getenv("OPENAI_API_KEY")
		if options.APIKey == "" {
			return nil, fmt.Errorf("OpenAI API key is required")
		}
	}

	if options.Endpoint == "" {
		options.Endpoint = DefaultOpenAITraceEndpoint
	}

	if options.Timeout == 0 {
		options.Timeout = DefaultTimeout
	}

	if options.MaxRetries == 0 {
		options.MaxRetries = MaxRetries
	}

	// Create backup directory if specified
	if options.BackupDir != "" {
		if err := os.MkdirAll(options.BackupDir, 0755); err != nil {
			return nil, fmt.Errorf("failed to create backup directory: %w", err)
		}
	}

	return &OpenAIExporter{
		options: options,
		client: &http.Client{
			Timeout: options.Timeout,
		},
	}, nil
}

// ExportSpan exports a single span to OpenAI's tracing service
func (e *OpenAIExporter) ExportSpan(ctx context.Context, span *StandardSpan) error {
	return e.ExportSpans(ctx, []*StandardSpan{span})
}

// ExportSpans exports multiple spans to OpenAI
func (e *OpenAIExporter) ExportSpans(ctx context.Context, spans []*StandardSpan) error {
	if len(spans) == 0 {
		return nil
	}

	// Convert spans to OpenAI format
	openAISpans := make([]OpenAISpanData, 0, len(spans))
	for _, span := range spans {
		openAISpan := e.convertToOpenAISpan(span)
		openAISpans = append(openAISpans, openAISpan)
	}

	// If backup directory is specified, always save a local copy first
	if e.options.BackupDir != "" {
		if err := e.saveToBackup(spans); err != nil {
			fmt.Printf("[WARN] Failed to save spans to backup: %v\n", err)
		}
	}

	// Create request body with data as array
	requestBody := map[string]any{
		"data": openAISpans,
	}

	// Marshal request body
	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return fmt.Errorf("failed to marshal traces: %w", err)
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, "POST", e.options.Endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.options.APIKey)
	req.Header.Set("OpenAI-Beta", "traces=v1")

	// Send request
	resp, err := e.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send traces: %w", err)
	}
	defer resp.Body.Close()

	// Check response - Accept 200 OK and 204 No Content as success
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API returned error %d: %s", resp.StatusCode, string(body))
	}

	// Log success message
	if resp.StatusCode == http.StatusNoContent {
		fmt.Println("[INFO] Successfully sent traces (204 No Content)")
	} else {
		fmt.Println("[INFO] Successfully sent traces (200 OK)")
	}

	return nil
}

// saveToBackup saves traces to a backup file if sending fails
func (e *OpenAIExporter) saveToBackup(spans []*StandardSpan) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Create backup filename with timestamp
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("trace_backup_%s_%s.json", spans[0].Context().TraceID, timestamp)
	filepath := fmt.Sprintf("%s/%s", e.options.BackupDir, filename)

	// Convert spans to OpenAI format
	openAISpans := make([]OpenAISpanData, 0, len(spans))
	for _, span := range spans {
		openAISpan := e.convertToOpenAISpan(span)
		openAISpans = append(openAISpans, openAISpan)
	}

	// Marshal to JSON
	jsonData, err := json.MarshalIndent(openAISpans, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal spans for backup: %w", err)
	}

	// Write to file
	err = os.WriteFile(filepath, jsonData, 0644)
	if err != nil {
		return fmt.Errorf("failed to write backup file: %w", err)
	}

	return nil
}

// convertToOpenAISpan converts a StandardSpan to OpenAI's span format
func (e *OpenAIExporter) convertToOpenAISpan(span *StandardSpan) OpenAISpanData {
	ctx := span.Context()

	// Format timestamps as ISO 8601
	startTime := ctx.StartTime.Format(time.RFC3339Nano)
	endTime := ctx.EndTime.Format(time.RFC3339Nano)

	// Get span type
	spanType := e.getSpanType(ctx)

	// Process span data based on type
	spanData := e.createSpanDataByType(ctx, spanType)

	// Format IDs to ensure they have proper prefixes
	spanID := e.formatSpanID(ctx.SpanID)
	traceID := e.formatTraceID(ctx.TraceID)
	parentID := e.formatParentID(ctx.ParentSpanID)

	// Create the span object following the Python spans.py export() format
	openAISpan := OpenAISpanData{
		Object:    "trace.span",
		ID:        spanID,
		TraceID:   traceID,
		ParentID:  parentID,
		StartTime: startTime,
		EndTime:   endTime,
		SpanData:  spanData,
	}

	// Add error information if present
	openAISpan.Error = e.getSpanError(ctx)

	return openAISpan
}

// getSpanType extracts and returns the span type from context attributes
func (e *OpenAIExporter) getSpanType(ctx *SpanContext) string {
	spanType := "agent" // Default
	if typeAttr, ok := ctx.Attributes["span_type"]; ok {
		if typeStr, ok := typeAttr.(string); ok {
			spanType = typeStr
		}
	}
	return spanType
}

// formatSpanID ensures the span ID has the required prefix
func (e *OpenAIExporter) formatSpanID(spanID string) string {
	if !strings.HasPrefix(spanID, "span_") {
		return "span_" + spanID
	}
	return spanID
}

// formatTraceID ensures the trace ID has the required prefix
func (e *OpenAIExporter) formatTraceID(traceID string) string {
	if !strings.HasPrefix(traceID, "trace_") {
		return "trace_" + traceID
	}
	return traceID
}

// formatParentID ensures the parent ID has the required prefix if it exists
func (e *OpenAIExporter) formatParentID(parentID string) string {
	if parentID != "" && !strings.HasPrefix(parentID, "span_") {
		return "span_" + parentID
	}
	return parentID
}

// getSpanError extracts error information from span context
func (e *OpenAIExporter) getSpanError(ctx *SpanContext) map[string]any {
	if errorAttr, ok := ctx.Attributes["error"]; ok {
		if errorStr, ok := errorAttr.(string); ok && errorStr != "" {
			return map[string]any{
				"message": errorStr,
			}
		} else if errorMap, ok := errorAttr.(map[string]any); ok {
			return errorMap
		}
	}
	return nil
}

// createSpanDataByType builds span data based on the span type
func (e *OpenAIExporter) createSpanDataByType(ctx *SpanContext, spanType string) map[string]any {
	spanData := make(map[string]any)
	spanData["type"] = spanType

	switch spanType {
	case "agent":
		e.processAgentSpan(ctx, spanData)
	case "function":
		e.processFunctionSpan(ctx, spanData)
	case "generation":
		e.processGenerationSpan(ctx, spanData)
	case "response":
		e.processResponseSpan(ctx, spanData)
	case "handoff":
		e.processHandoffSpan(ctx, spanData)
	case "custom":
		e.processCustomSpan(ctx, spanData)
	case "guardrail":
		e.processGuardrailSpan(ctx, spanData)
	case "tool":
		e.processToolSpan(ctx, spanData)
	default:
		// For unknown types, just include the name
		spanData["name"] = ctx.Name
	}

	return spanData
}

// processAgentSpan populates span data for agent spans
func (e *OpenAIExporter) processAgentSpan(ctx *SpanContext, spanData map[string]any) {
	spanData["name"] = ctx.Name
	e.addAttributeIfExists(ctx, spanData, "handoffs")
	e.addAttributeIfExists(ctx, spanData, "tools")
	e.addAttributeIfExists(ctx, spanData, "output_type")
}

// processFunctionSpan populates span data for function spans
func (e *OpenAIExporter) processFunctionSpan(ctx *SpanContext, spanData map[string]any) {
	spanData["name"] = ctx.Name
	e.addAttributeIfExists(ctx, spanData, "input")
	e.addAttributeIfExists(ctx, spanData, "output")
}

// processGenerationSpan populates span data for generation spans
func (e *OpenAIExporter) processGenerationSpan(ctx *SpanContext, spanData map[string]any) {
	e.addAttributeIfExists(ctx, spanData, "input")
	e.addAttributeIfExists(ctx, spanData, "output")
	e.addAttributeIfExists(ctx, spanData, "model")
	e.addAttributeIfExists(ctx, spanData, "model_config")
	e.addAttributeIfExists(ctx, spanData, "usage")
}

// processResponseSpan populates span data for response spans
func (e *OpenAIExporter) processResponseSpan(ctx *SpanContext, spanData map[string]any) {
	e.addAttributeIfExists(ctx, spanData, "response_id")
}

// processHandoffSpan populates span data for handoff spans
func (e *OpenAIExporter) processHandoffSpan(ctx *SpanContext, spanData map[string]any) {
	e.addAttributeIfExists(ctx, spanData, "from_agent")
	e.addAttributeIfExists(ctx, spanData, "to_agent")
}

// processCustomSpan populates span data for custom spans
func (e *OpenAIExporter) processCustomSpan(ctx *SpanContext, spanData map[string]any) {
	spanData["name"] = ctx.Name
	e.addAttributeIfExists(ctx, spanData, "data")
}

// processGuardrailSpan populates span data for guardrail spans
func (e *OpenAIExporter) processGuardrailSpan(ctx *SpanContext, spanData map[string]any) {
	spanData["name"] = ctx.Name
	e.addAttributeIfExists(ctx, spanData, "triggered")
}

// processToolSpan populates span data for tool spans
func (e *OpenAIExporter) processToolSpan(ctx *SpanContext, spanData map[string]any) {
	spanData["name"] = ctx.Name

	// Override name with tool_name if available
	if toolName, ok := ctx.Attributes["tool_name"]; ok {
		spanData["name"] = toolName
	}

	// Map arguments to input and result to output
	if arguments, ok := ctx.Attributes["arguments"]; ok {
		spanData["input"] = arguments
	}
	if result, ok := ctx.Attributes["result"]; ok {
		spanData["output"] = result
	}
}

// addAttributeIfExists adds an attribute to span data if it exists in the context
func (e *OpenAIExporter) addAttributeIfExists(ctx *SpanContext, spanData map[string]any, key string) {
	if value, ok := ctx.Attributes[key]; ok {
		spanData[key] = value
	}
}

// Shutdown gracefully shuts down the exporter
func (e *OpenAIExporter) Shutdown(ctx context.Context) error {
	// Nothing to do here for now
	return nil
}
