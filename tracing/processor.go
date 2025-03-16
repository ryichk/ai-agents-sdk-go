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
	"time"
)

type LogLevel int

const (
	LogLevelDebug LogLevel = iota
	LogLevelInfo
	LogLevelWarn
	LogLevelError
	LogLevelNone
)

// Internal global tracing configuration
var tracingConfig = struct {
	LogLevel LogLevel
}{
	LogLevel: LogLevelInfo, // Default is Info
}

// SetLogLevel sets the global log level
func SetLogLevel(level LogLevel) {
	tracingConfig.LogLevel = level
}

// Logger provides a simple logging interface
type Logger interface {
	Debug(format string, args ...any)
	Info(format string, args ...any)
	Warn(format string, args ...any)
	Error(format string, args ...any)
}

// defaultLogger is the default logger implementation
type defaultLogger struct{}

func (l *defaultLogger) Debug(format string, args ...any) {
	if tracingConfig.LogLevel <= LogLevelDebug {
		fmt.Printf("[DEBUG] "+format+"\n", args...)
	}
}

func (l *defaultLogger) Info(format string, args ...any) {
	if tracingConfig.LogLevel <= LogLevelInfo {
		fmt.Printf("[INFO] "+format+"\n", args...)
	}
}

func (l *defaultLogger) Warn(format string, args ...any) {
	if tracingConfig.LogLevel <= LogLevelWarn {
		fmt.Printf("[WARN] "+format+"\n", args...)
	}
}

func (l *defaultLogger) Error(format string, args ...any) {
	if tracingConfig.LogLevel <= LogLevelError {
		fmt.Printf("[ERROR] "+format+"\n", args...)
	}
}

// logger is the global logger instance
var logger Logger = &defaultLogger{}

// SetLogger sets a custom logger
func SetLogger(l Logger) {
	if l != nil {
		logger = l
	}
}

// BatchSpanProcessorOptions defines configuration options for the batch span processor
type BatchSpanProcessorOptions struct {
	// MaxQueueSize is the maximum size of the queue
	MaxQueueSize int
	// MaxBatchSize is the maximum number of spans to process at once
	MaxBatchSize int
	// ExportInterval is the interval for exporting
	ExportInterval time.Duration
	// BackupDir is the path to the backup directory
	BackupDir string
}

// BatchSpanProcessor is a processor that accumulates spans and processes them in batches
type BatchSpanProcessor struct {
	exporter       SpanExporter
	spans          []*StandardSpan
	maxBatchSize   int
	exportInterval time.Duration
	mu             sync.Mutex
	wg             sync.WaitGroup
	options        *BatchSpanProcessorOptions
	done           chan struct{}
	queue          chan *StandardSpan
	exportTrigger  chan struct{}
}

func NewBatchSpanProcessor(exporter SpanExporter, option ...BatchProcessorOption) *BatchSpanProcessor {
	opts := &BatchSpanProcessorOptions{
		MaxQueueSize:   1000,
		MaxBatchSize:   100,
		ExportInterval: 5 * time.Second,
		BackupDir:      "",
	}

	// Apply options
	for _, o := range option {
		o(opts)
	}

	processor := &BatchSpanProcessor{
		exporter:       exporter,
		spans:          make([]*StandardSpan, 0, opts.MaxBatchSize),
		maxBatchSize:   opts.MaxBatchSize,
		exportInterval: opts.ExportInterval,
		done:           make(chan struct{}),
		options:        opts,
		queue:          make(chan *StandardSpan, opts.MaxQueueSize),
		exportTrigger:  make(chan struct{}, 1),
	}

	// Start export processing in the background
	go processor.processLoop()

	return processor
}

// processLoop runs a loop that exports spans periodically
func (p *BatchSpanProcessor) processLoop() {
	p.wg.Add(1)
	defer p.wg.Done()

	ticker := time.NewTicker(p.exportInterval)
	defer ticker.Stop()

	for {
		select {
		case <-p.done:
			// When receiving a termination signal, export the remaining spans and exit
			p.exportBatch()
			return
		case <-ticker.C:
			// Export spans periodically
			p.exportBatch()
		case <-p.exportTrigger:
			// When receiving a trigger, export spans
			p.exportBatch()
		case span := <-p.queue:
			// Get spans from the queue and add them
			p.mu.Lock()
			p.spans = append(p.spans, span)
			if len(p.spans) >= p.maxBatchSize {
				// When reaching the batch size, export immediately
				spansToExport := p.spans
				p.spans = make([]*StandardSpan, 0, p.maxBatchSize)
				p.mu.Unlock()
				p.export(spansToExport)
			} else {
				p.mu.Unlock()
			}
		}
	}
}

// exportBatch exports the current batch
func (p *BatchSpanProcessor) exportBatch() {
	p.mu.Lock()
	if len(p.spans) == 0 {
		p.mu.Unlock()
		return
	}
	spansToExport := p.spans
	p.spans = make([]*StandardSpan, 0, p.maxBatchSize)
	p.mu.Unlock()

	// Export spans
	p.export(spansToExport)
}

// ProcessSpan processes a span by adding it to the batch queue
func (p *BatchSpanProcessor) ProcessSpan(span *StandardSpan) {
	select {
	case p.queue <- span:
		// Successfully added to queue
	default:
		// Queue is full, log and potentially save to backup
		logger.Warn("Span queue is full, dropping span")
		if p.options.BackupDir != "" {
			if err := p.saveSpanToBackup(span); err != nil {
				logger.Error("Failed to save span to backup: %v", err)
			}
		}
	}

	// If the queue size exceeds the threshold, signal for export
	if len(p.queue) >= p.options.MaxBatchSize {
		select {
		case p.exportTrigger <- struct{}{}:
			// Successfully triggered export
		default:
			// Export already triggered, do nothing
		}
	}
}

// saveSpanToBackup saves a single span to a backup file
func (p *BatchSpanProcessor) saveSpanToBackup(span *StandardSpan) error {
	if p.options.BackupDir == "" {
		return nil
	}

	// Ensure the backup directory exists
	if err := os.MkdirAll(p.options.BackupDir, 0750); err != nil {
		return fmt.Errorf("failed to create backup directory: %w", err)
	}

	// Create backup filename with timestamp
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("span_backup_%s_%s.json", span.Context().TraceID, timestamp)
	filepath := fmt.Sprintf("%s/%s", p.options.BackupDir, filename)

	// Convert the span to JSON
	spanJSON, err := json.MarshalIndent(span.Context(), "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal span for backup: %w", err)
	}

	// Write to file
	if err := os.WriteFile(filepath, spanJSON, 0600); err != nil {
		return fmt.Errorf("failed to write backup file: %w", err)
	}

	logger.Info("Saved span to backup file: %s", filepath)
	return nil
}

// export exports a batch of spans
func (p *BatchSpanProcessor) export(batch []*StandardSpan) {
	if len(batch) == 0 {
		return
	}

	logger.Debug("Exporting %d spans", len(batch))

	// Try exporting via the exporter
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	err := p.exporter.ExportSpans(ctx, batch)
	if err != nil {
		logger.Error("Failed to export spans: %v", err)

		// If export fails and backup directory is specified, save to backup
		if p.options.BackupDir != "" {
			batchBackupFile := fmt.Sprintf("%s/batch_backup_%s.json",
				p.options.BackupDir,
				time.Now().Format("20060102_150405"),
			)

			// Convert all spans to JSON
			spansJSON := make([]map[string]any, 0, len(batch))
			for _, span := range batch {
				spanMap := map[string]any{
					"trace_id":   span.Context().TraceID,
					"span_id":    span.Context().SpanID,
					"parent_id":  span.Context().ParentSpanID,
					"name":       span.Context().Name,
					"attributes": span.Context().Attributes,
					"start_time": span.Context().StartTime.Format(time.RFC3339Nano),
					"end_time":   span.Context().EndTime.Format(time.RFC3339Nano),
				}
				spansJSON = append(spansJSON, spanMap)
			}

			// Marshal to JSON
			jsonData, err := json.MarshalIndent(spansJSON, "", "  ")
			if err != nil {
				logger.Error("Failed to marshal batch spans for backup: %v", err)
				return
			}

			// Ensure the backup directory exists
			if err := os.MkdirAll(p.options.BackupDir, 0750); err != nil {
				logger.Error("Failed to create backup directory: %v", err)
				return
			}

			// Write to file
			if err := os.WriteFile(batchBackupFile, jsonData, 0600); err != nil {
				logger.Error("Failed to write batch backup file: %v", err)
			} else {
				logger.Info("Saved batch to backup file: %s", batchBackupFile)
			}
		}
	} else {
		logger.Info("Successfully exported %d spans", len(batch))
	}
}

// NotifySpanEnded processes a span that has ended
func (p *BatchSpanProcessor) NotifySpanEnded(span *StandardSpan) {
	if span != nil {
		p.ProcessSpan(span)
	}
}

// BatchProcessorOption is a function that sets options for the batch processor
type BatchProcessorOption func(*BatchSpanProcessorOptions)

// WithMaxQueueSize sets the maximum queue size
func WithMaxQueueSize(size int) BatchProcessorOption {
	return func(o *BatchSpanProcessorOptions) {
		if size > 0 {
			o.MaxQueueSize = size
		}
	}
}

// WithBatchSize sets the batch size
func WithBatchSize(size int) BatchProcessorOption {
	return func(o *BatchSpanProcessorOptions) {
		if size > 0 {
			o.MaxBatchSize = size
		}
	}
}

// WithExportInterval sets the export interval
func WithExportInterval(interval time.Duration) BatchProcessorOption {
	return func(o *BatchSpanProcessorOptions) {
		if interval > 0 {
			o.ExportInterval = interval
		}
	}
}

// WithBackupDir sets the backup directory
func WithBackupDir(dir string) BatchProcessorOption {
	return func(o *BatchSpanProcessorOptions) {
		o.BackupDir = dir
	}
}

// Shutdown stops the processor
func (p *BatchSpanProcessor) Shutdown(ctx context.Context) error {
	p.mu.Lock()
	alreadyClosed := false

	select {
	case <-p.done:
		// If already closed, record but continue processing
		alreadyClosed = true
	default:
		close(p.done)
	}
	p.mu.Unlock()

	// Even if already closed, execute exporter shutdown
	if alreadyClosed {
		// Only perform exporter shutdown
		return p.exporter.Shutdown(ctx)
	}

	// Channel for waiting for termination
	doneCh := make(chan struct{})

	go func() {
		p.wg.Wait()
		close(doneCh)
	}()

	// Wait for context timeout
	var err error
	select {
	case <-ctx.Done():
		err = ctx.Err()
	case <-doneCh:
		// Do nothing
	}

	// Shutdown exporter
	exporterErr := p.exporter.Shutdown(ctx)
	if err == nil {
		err = exporterErr
	}

	return err
}

// OnStart is called when a span starts
func (p *BatchSpanProcessor) OnStart(span *StandardSpan) {
	// Nothing special to do when a span starts
}

// OnEnd is called when a span ends
func (p *BatchSpanProcessor) OnEnd(span *StandardSpan) {
	// Add the span to the queue for processing
	p.ProcessSpan(span)
}

// ForceFlush forces immediate export of the current batch
func (p *BatchSpanProcessor) ForceFlush() {
	p.exportBatch()
}
