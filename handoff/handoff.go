// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package handoff

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/abadojack/whatlanggo"
)

// InputData represents the data being passed during a handoff
type InputData struct {
	// Complete conversation history
	InputHistory []map[string]any
	// Items before the handoff
	PreHandoffItems []map[string]any
	// New items since the handoff
	NewItems []map[string]any
	// Metadata about the handoff
	Metadata map[string]any
}

// Result represents the result of a handoff decision
type Result struct {
	// Whether the handoff should occur
	ShouldHandoff bool
	// Reason for the handoff decision
	Reason string
	// Any additional context or metadata
	Context map[string]any
}

// Callback is a function that is called when a handoff is invoked
type Callback func(ctx context.Context, inputData *InputData, inputJSON string) error

// Registry manages handoff registrations
type Registry struct {
	mutex                  sync.Mutex
	recentHandoffs         map[string]time.Time
	minTimeBetweenHandoffs time.Duration
}

// Options represents handoff options
type Options struct {
	Name            string
	Description     string
	ToolName        string
	ToolDescription string
	InputSchema     JSONSchema
	Callback        Callback
	InputJSONSchema JSONSchema
	OnHandoff       Callback
	InputFilter     func(ctx context.Context, inputData *InputData) (*InputData, error)
}

// JSONSchema represents a JSON schema for validating handoff inputs
type JSONSchema map[string]any

// Handoff represents a handoff to another agent
type Handoff interface {
	// TargetAgent returns the target agent of the handoff
	TargetAgent() any

	// Description returns the description of the handoff
	Description() string

	// ShouldHandoff determines whether to handoff
	ShouldHandoff(ctx context.Context, input string) (bool, error)

	// Name returns the name of the handoff
	Name() string

	// FilterInput filters the input being passed to the target agent
	FilterInput(ctx context.Context, inputData *InputData) (*InputData, error)

	// GetLastHandoffTime returns the time of the last handoff from this handoff
	GetLastHandoffTime() time.Time

	// UpdateLastHandoffTime updates the time of the last handoff
	UpdateLastHandoffTime()

	// ToolName returns the name of the tool that represents the handoff
	ToolName() string

	// ToolDescription returns the description of the tool that represents the handoff
	ToolDescription() string

	// InputJSONSchema returns the JSON schema for the handoff input
	InputJSONSchema() JSONSchema

	// OnHandoff executes the callback when a handoff is invoked
	OnHandoff(ctx context.Context, inputData *InputData, inputJSON string) error
}

// BaseHandoff is the base implementation of a handoff
type BaseHandoff struct {
	targetAgent     any
	description     string
	name            string
	lastHandoffTime time.Time
	toolName        string
	toolDescription string
	inputJSONSchema JSONSchema
	onHandoffCB     Callback
}

// DefaultToolName generates a default tool name for an agent
func DefaultToolName(agentName string) string {
	// Convert spaces to underscores and make lowercase
	name := strings.ToLower(strings.ReplaceAll(agentName, " ", "_"))
	return fmt.Sprintf("transfer_to_%s", name)
}

// DefaultToolDescription generates a default tool description for an agent
func DefaultToolDescription(agentName, handoffDescription string) string {
	desc := fmt.Sprintf("Handoff to the %s agent to handle the request.", agentName)
	if handoffDescription != "" {
		desc += " " + handoffDescription
	}
	return desc
}

// TargetAgent returns the target agent of the handoff
func (h *BaseHandoff) TargetAgent() any {
	return h.targetAgent
}

// Description returns the description of the handoff
func (h *BaseHandoff) Description() string {
	return h.description
}

// Name returns the name of the handoff
func (h *BaseHandoff) Name() string {
	return h.name
}

// FilterInput provides a default implementation for filtering input
func (h *BaseHandoff) FilterInput(ctx context.Context, inputData *InputData) (*InputData, error) {
	return inputData, nil
}

// GetLastHandoffTime returns the last time this handoff was invoked
func (h *BaseHandoff) GetLastHandoffTime() time.Time {
	return h.lastHandoffTime
}

// UpdateLastHandoffTime updates the last handoff time to now
func (h *BaseHandoff) UpdateLastHandoffTime() {
	h.lastHandoffTime = time.Now()
}

// ToolName returns the name of the tool that represents the handoff
func (h *BaseHandoff) ToolName() string {
	if h.toolName != "" {
		return h.toolName
	}

	// If targetAgent has a Name() method, try to use it for the default name
	if agent, ok := h.targetAgent.(interface{ Name() string }); ok {
		return DefaultToolName(agent.Name())
	}

	return DefaultToolName(h.name)
}

// ToolDescription returns the description of the tool that represents the handoff
func (h *BaseHandoff) ToolDescription() string {
	if h.toolDescription != "" {
		return h.toolDescription
	}

	// If targetAgent has a Name() method, try to use it for the default description
	if agent, ok := h.targetAgent.(interface{ Name() string }); ok {
		return DefaultToolDescription(agent.Name(), h.description)
	}

	return DefaultToolDescription(h.name, h.description)
}

// InputJSONSchema returns the JSON schema for the handoff input
func (h *BaseHandoff) InputJSONSchema() JSONSchema {
	if h.inputJSONSchema != nil {
		return h.inputJSONSchema
	}
	return JSONSchema{}
}

// OnHandoff executes the callback when a handoff is invoked
func (h *BaseHandoff) OnHandoff(ctx context.Context, inputData *InputData, inputJSON string) error {
	if h.onHandoffCB != nil {
		return h.onHandoffCB(ctx, inputData, inputJSON)
	}
	return nil
}

// SimpleHandoff is a simple handoff that always returns true for ShouldHandoff
type SimpleHandoff struct {
	BaseHandoff
}

// ShouldHandoff for SimpleHandoff always returns true (delegating to the LLM)
func (h *SimpleHandoff) ShouldHandoff(ctx context.Context, input string) (bool, error) {
	return true, nil
}

// FunctionHandoff uses a function to determine whether to handoff
type FunctionHandoff struct {
	BaseHandoff
	handoffFunc func(ctx context.Context, input string) (bool, error)
}

// ShouldHandoff calls the handoff function
func (h *FunctionHandoff) ShouldHandoff(ctx context.Context, input string) (bool, error) {
	return h.handoffFunc(ctx, input)
}

// PatternHandoff handles handoffs based on regex pattern matching
type PatternHandoff struct {
	BaseHandoff
	pattern *regexp.Regexp
}

// ShouldHandoff checks if the input matches the pattern
func (h *PatternHandoff) ShouldHandoff(ctx context.Context, input string) (bool, error) {
	return h.pattern.MatchString(input), nil
}

// KeywordHandoff handles handoffs based on keyword presence
type KeywordHandoff struct {
	BaseHandoff
	keywords []string
}

// ShouldHandoff checks if any keyword is in the input
func (h *KeywordHandoff) ShouldHandoff(ctx context.Context, input string) (bool, error) {
	lowerInput := strings.ToLower(input)
	for _, keyword := range h.keywords {
		if strings.Contains(lowerInput, strings.ToLower(keyword)) {
			return true, nil
		}
	}
	return false, nil
}

// LanguageHandoff defines a handoff that occurs when the input is detected to be in a specified language
type LanguageHandoff struct {
	targetAgent any
	description string
	langCode    string
	targetLang  whatlanggo.Lang
}

// NewLanguageHandoff creates a handoff that triggers when the input text is detected to be in the specified language
func NewLanguageHandoff(targetAgent any, description, langCode string) *LanguageHandoff {
	var targetLang whatlanggo.Lang
	switch langCode {
	case "es":
		targetLang = whatlanggo.Spa
	case "fr":
		targetLang = whatlanggo.Fra
	case "de":
		targetLang = whatlanggo.Deu
	case "it":
		targetLang = whatlanggo.Ita
	case "ja":
		targetLang = whatlanggo.Jpn
	default:
		targetLang = whatlanggo.Eng
	}

	return &LanguageHandoff{
		targetAgent: targetAgent,
		description: description,
		langCode:    langCode,
		targetLang:  targetLang,
	}
}

// Name returns the name of the handoff
func (h *LanguageHandoff) Name() string {
	return "language_handoff"
}

// Description returns the description of the handoff
func (h *LanguageHandoff) Description() string {
	return h.description
}

// TargetAgent returns the target agent for this handoff
func (h *LanguageHandoff) TargetAgent() any {
	return h.targetAgent
}

// ShouldHandoff determines if a handoff should occur based on language detection
func (h *LanguageHandoff) ShouldHandoff(ctx context.Context, input string) (bool, error) {
	if input == "" {
		return false, nil
	}

	// Simple check based on special characters and common words first
	// For Spanish detection
	if h.langCode == "es" {
		// Check for Spanish special characters
		if strings.Contains(input, "¿") || strings.Contains(input, "¡") ||
			strings.Contains(input, "ñ") || strings.Contains(input, "ó") ||
			strings.Contains(input, "á") || strings.Contains(input, "é") ||
			strings.Contains(input, "í") || strings.Contains(input, "ú") {
			return true, nil
		}

		// Check for common Spanish words
		spanishWords := []string{"hola", "como", "estas", "gracias", "buenos", "dias", "adios", "por favor", "ayuda"}
		for _, word := range spanishWords {
			if strings.Contains(strings.ToLower(input), word) {
				return true, nil
			}
		}
	}

	// Fall back to the language detection library
	info := whatlanggo.Detect(input)
	return info.Lang == h.targetLang && info.Confidence > 0.5, nil
}

// FilterInput filters the input before handing off
func (h *LanguageHandoff) FilterInput(ctx context.Context, inputData *InputData) (*InputData, error) {
	// Just pass through the input data by default
	return inputData, nil
}

// ToolName returns the name of the tool that represents the handoff
func (h *LanguageHandoff) ToolName() string {
	if agent, ok := h.targetAgent.(interface{ Name() string }); ok {
		langName := ""
		switch h.langCode {
		case "es":
			langName = "Spanish"
		case "fr":
			langName = "French"
		case "de":
			langName = "German"
		case "it":
			langName = "Italian"
		case "ja":
			langName = "Japanese"
		default:
			langName = h.langCode
		}
		return DefaultToolName(fmt.Sprintf("%s_%s", langName, agent.Name()))
	}
	return DefaultToolName(fmt.Sprintf("language_%s", h.langCode))
}

// ToolDescription returns the description of the tool that represents the handoff
func (h *LanguageHandoff) ToolDescription() string {
	langName := ""
	switch h.langCode {
	case "es":
		langName = "Spanish"
	case "fr":
		langName = "French"
	case "de":
		langName = "German"
	case "it":
		langName = "Italian"
	case "ja":
		langName = "Japanese"
	default:
		langName = h.langCode
	}

	if agent, ok := h.targetAgent.(interface{ Name() string }); ok {
		return fmt.Sprintf("Handoff to the %s agent to handle %s language requests.", agent.Name(), langName)
	}
	return fmt.Sprintf("Handoff to handle %s language requests.", langName)
}

// InputJSONSchema returns an empty JSON schema
func (h *LanguageHandoff) InputJSONSchema() JSONSchema {
	return JSONSchema{}
}

// OnHandoff does nothing for LanguageHandoff
func (h *LanguageHandoff) OnHandoff(ctx context.Context, inputData *InputData, inputJSON string) error {
	return nil
}

// GetLastHandoffTime returns zero time for LanguageHandoff
func (h *LanguageHandoff) GetLastHandoffTime() time.Time {
	return time.Time{}
}

// UpdateLastHandoffTime does nothing for LanguageHandoff
func (h *LanguageHandoff) UpdateLastHandoffTime() {
	// No-op
}

// FilteredHandoff wraps another handoff and applies input filtering
type FilteredHandoff struct {
	BaseHandoff
	baseHandoff Handoff
	filterFunc  func(ctx context.Context, inputData *InputData) (*InputData, error)
}

// ShouldHandoff delegates to the base handoff
func (h *FilteredHandoff) ShouldHandoff(ctx context.Context, input string) (bool, error) {
	return h.baseHandoff.ShouldHandoff(ctx, input)
}

// FilterInput applies the filter function
func (h *FilteredHandoff) FilterInput(ctx context.Context, inputData *InputData) (*InputData, error) {
	return h.filterFunc(ctx, inputData)
}

// ToolName delegates to the base handoff
func (h *FilteredHandoff) ToolName() string {
	return h.baseHandoff.ToolName()
}

// ToolDescription delegates to the base handoff
func (h *FilteredHandoff) ToolDescription() string {
	return h.baseHandoff.ToolDescription()
}

// InputJSONSchema delegates to the base handoff
func (h *FilteredHandoff) InputJSONSchema() JSONSchema {
	return h.baseHandoff.InputJSONSchema()
}

// OnHandoff delegates to the base handoff
func (h *FilteredHandoff) OnHandoff(ctx context.Context, inputData *InputData, inputJSON string) error {
	return h.baseHandoff.OnHandoff(ctx, inputData, inputJSON)
}

// ValidateJSON validates JSON data against a schema
func ValidateJSON(jsonData string, schema JSONSchema) (map[string]any, error) {
	if len(schema) == 0 {
		return nil, nil
	}

	var data map[string]any
	if err := json.Unmarshal([]byte(jsonData), &data); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	return validateJSONData(data, schema)
}

// validateJSONData validates the parsed JSON data against the schema
func validateJSONData(data map[string]any, schema JSONSchema) (map[string]any, error) {
	if err := validateRequiredFields(data, schema); err != nil {
		return nil, err
	}

	if err := validateFieldTypes(data, schema); err != nil {
		return nil, err
	}

	return data, nil
}

// validateRequiredFields checks if all required fields are present
func validateRequiredFields(data map[string]any, schema JSONSchema) error {
	if required, ok := schema["required"].([]string); ok {
		for _, field := range required {
			if _, exists := data[field]; !exists {
				return fmt.Errorf("missing required field: %s", field)
			}
		}
	}
	return nil
}

// validateFieldTypes validates the types of fields according to the schema
func validateFieldTypes(data map[string]any, schema JSONSchema) error {
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		return nil
	}

	for field, value := range data {
		if err := validateFieldType(field, value, properties); err != nil {
			return err
		}
	}
	return nil
}

// validateFieldType validates a single field's type
func validateFieldType(field string, value any, properties map[string]any) error {
	fieldSchema, ok := properties[field].(map[string]any)
	if !ok {
		return nil
	}

	expectedType, ok := fieldSchema["type"].(string)
	if !ok {
		return nil
	}

	if !isValidType(value, expectedType) {
		return fmt.Errorf("invalid type for field %s: expected %s", field, expectedType)
	}

	return nil
}

// isValidType checks if a value matches the expected type
func isValidType(value any, expectedType string) bool {
	switch expectedType {
	case "string":
		_, ok := value.(string)
		return ok
	case "number", "integer":
		switch value.(type) {
		case float64, int, int64:
			return true
		default:
			return false
		}
	case "boolean":
		_, ok := value.(bool)
		return ok
	case "array":
		_, ok := value.([]any)
		return ok
	case "object":
		_, ok := value.(map[string]any)
		return ok
	default:
		return true
	}
}

// NewHandoffRegistry creates a new handoff registry
func NewHandoffRegistry(minTimeBetween time.Duration) *Registry {
	return &Registry{
		recentHandoffs:         make(map[string]time.Time),
		minTimeBetweenHandoffs: minTimeBetween,
	}
}

// CanHandoff checks if a handoff can occur based on timing and loop detection
func (r *Registry) CanHandoff(from string, to string) (bool, string) {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	now := time.Now()

	// Check for recent handoff in this direction
	key := fmt.Sprintf("%s->%s", from, to)
	if lastTime, exists := r.recentHandoffs[key]; exists {
		if now.Sub(lastTime) < r.minTimeBetweenHandoffs {
			return false, fmt.Sprintf("too soon to handoff from %s to %s again", from, to)
		}
	}

	// Check for potential loop
	reverseKey := fmt.Sprintf("%s->%s", to, from)
	if lastTime, exists := r.recentHandoffs[reverseKey]; exists {
		if now.Sub(lastTime) < r.minTimeBetweenHandoffs*2 {
			return false, fmt.Sprintf("potential handoff loop detected between %s and %s", from, to)
		}
	}

	// Record this handoff
	r.recentHandoffs[key] = now
	return true, ""
}

// NewHandoffWithOptions creates a handoff with customizable options
func NewHandoffWithOptions(targetAgent any, description string, options Options) Handoff {
	return &SimpleHandoff{
		BaseHandoff: BaseHandoff{
			targetAgent:     targetAgent,
			description:     description,
			name:            "simple_handoff",
			toolName:        options.ToolName,
			toolDescription: options.ToolDescription,
			inputJSONSchema: options.InputJSONSchema,
			onHandoffCB:     options.OnHandoff,
		},
	}
}

// NewHandoff creates a new simple handoff
func NewHandoff(targetAgent any, description string) Handoff {
	return &SimpleHandoff{
		BaseHandoff: BaseHandoff{
			targetAgent: targetAgent,
			description: description,
			name:        "simple_handoff",
		},
	}
}

// NewFunctionHandoff creates a new function-based handoff
func NewFunctionHandoff(targetAgent any, description string, handoffFunc func(ctx context.Context, input string) (bool, error)) Handoff {
	return &FunctionHandoff{
		BaseHandoff: BaseHandoff{
			targetAgent: targetAgent,
			description: description,
			name:        "function_handoff",
		},
		handoffFunc: handoffFunc,
	}
}

// NewFunctionHandoffWithOptions creates a new function-based handoff with options
func NewFunctionHandoffWithOptions(targetAgent any, description string, handoffFunc func(ctx context.Context, input string) (bool, error), options Options) Handoff {
	return &FunctionHandoff{
		BaseHandoff: BaseHandoff{
			targetAgent:     targetAgent,
			description:     description,
			name:            "function_handoff",
			toolName:        options.ToolName,
			toolDescription: options.ToolDescription,
			inputJSONSchema: options.InputJSONSchema,
			onHandoffCB:     options.OnHandoff,
		},
		handoffFunc: handoffFunc,
	}
}

// NewPatternHandoff creates a new pattern-based handoff
func NewPatternHandoff(targetAgent any, description string, pattern string) (Handoff, error) {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, fmt.Errorf("invalid pattern: %w", err)
	}

	return &PatternHandoff{
		BaseHandoff: BaseHandoff{
			targetAgent: targetAgent,
			description: description,
			name:        "pattern_handoff",
		},
		pattern: re,
	}, nil
}

// NewPatternHandoffWithOptions creates a new pattern-based handoff with options
func NewPatternHandoffWithOptions(targetAgent any, description string, pattern string, options Options) (Handoff, error) {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, fmt.Errorf("invalid pattern: %w", err)
	}

	return &PatternHandoff{
		BaseHandoff: BaseHandoff{
			targetAgent:     targetAgent,
			description:     description,
			name:            "pattern_handoff",
			toolName:        options.ToolName,
			toolDescription: options.ToolDescription,
			inputJSONSchema: options.InputJSONSchema,
			onHandoffCB:     options.OnHandoff,
		},
		pattern: re,
	}, nil
}

// NewKeywordHandoff creates a new keyword-based handoff
func NewKeywordHandoff(targetAgent any, description string, keywords []string) Handoff {
	return &KeywordHandoff{
		BaseHandoff: BaseHandoff{
			targetAgent: targetAgent,
			description: description,
			name:        "keyword_handoff",
		},
		keywords: keywords,
	}
}

// NewKeywordHandoffWithOptions creates a new keyword-based handoff with options
func NewKeywordHandoffWithOptions(targetAgent any, description string, keywords []string, options Options) Handoff {
	return &KeywordHandoff{
		BaseHandoff: BaseHandoff{
			targetAgent:     targetAgent,
			description:     description,
			name:            "keyword_handoff",
			toolName:        options.ToolName,
			toolDescription: options.ToolDescription,
			inputJSONSchema: options.InputJSONSchema,
			onHandoffCB:     options.OnHandoff,
		},
		keywords: keywords,
	}
}

// NewFilteredHandoff creates a handoff with input filtering
func NewFilteredHandoff(baseHandoff Handoff, filterFunc func(ctx context.Context, inputData *InputData) (*InputData, error)) Handoff {
	return &FilteredHandoff{
		BaseHandoff: BaseHandoff{
			targetAgent: baseHandoff.TargetAgent(),
			description: baseHandoff.Description(),
			name:        "filtered_" + baseHandoff.Name(),
		},
		baseHandoff: baseHandoff,
		filterFunc:  filterFunc,
	}
}

// CreateJSONSchema creates a JSON schema for a struct type or an interface with required fields
func CreateJSONSchema(structType any, required []string) JSONSchema {
	schema := JSONSchema{
		"type":       "object",
		"properties": map[string]any{},
	}

	if len(required) > 0 {
		schema["required"] = required
	}

	t := reflect.TypeOf(structType)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	if t.Kind() != reflect.Struct {
		return schema
	}

	properties := schema["properties"].(map[string]any)

	for i := range make([]int, t.NumField()) {
		field := t.Field(i)
		jsonTag := field.Tag.Get("json")
		if jsonTag == "" || jsonTag == "-" {
			continue
		}

		// Split the json tag to get the field name
		parts := strings.Split(jsonTag, ",")
		fieldName := parts[0]

		fieldType := field.Type
		propSchema := map[string]any{}

		switch fieldType.Kind() {
		case reflect.Bool:
			propSchema["type"] = "boolean"
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			propSchema["type"] = "integer"
		case reflect.Float32, reflect.Float64:
			propSchema["type"] = "number"
		case reflect.String:
			propSchema["type"] = "string"
		case reflect.Slice, reflect.Array:
			propSchema["type"] = "array"
			if fieldType.Elem().Kind() == reflect.Struct {
				propSchema["items"] = CreateJSONSchema(reflect.New(fieldType.Elem()).Interface(), nil)
			} else {
				propSchema["items"] = map[string]any{
					"type": getJSONType(fieldType.Elem().Kind()),
				}
			}
		case reflect.Map:
			propSchema["type"] = "object"
		case reflect.Struct:
			propSchema["type"] = "object"
			propSchema["properties"] = CreateJSONSchema(reflect.New(fieldType).Interface(), nil)["properties"]
		}

		properties[fieldName] = propSchema
	}

	return schema
}

// getJSONType returns the JSON schema type for a Go reflect.Kind
func getJSONType(kind reflect.Kind) string {
	switch kind {
	case reflect.Bool:
		return "boolean"
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return "integer"
	case reflect.Float32, reflect.Float64:
		return "number"
	case reflect.String:
		return "string"
	case reflect.Slice, reflect.Array:
		return "array"
	case reflect.Map, reflect.Struct:
		return "object"
	default:
		return "string"
	}
}

// InputFilter is a function that filters the input data during a handoff
type InputFilter func(ctx context.Context, inputData *InputData) (*InputData, error)
