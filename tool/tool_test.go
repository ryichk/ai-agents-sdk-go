// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package tool

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func add(a, b int) int {
	return a + b
}

func divide(a, b float64) (float64, error) {
	if b == 0 {
		return 0, errors.New("cannot divide by zero")
	}
	return a / b, nil
}

func processWithContext(ctx context.Context, data string) string {
	return "Processed: " + data
}

// Simple tool implementation for testing
type SimpleAddTool struct{}

func (t *SimpleAddTool) Name() string {
	return "add"
}

func (t *SimpleAddTool) Description() string {
	return "Adds two numbers together"
}

func (t *SimpleAddTool) ParamsJSONSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"a": map[string]any{
				"type":        "integer",
				"description": "First number",
			},
			"b": map[string]any{
				"type":        "integer",
				"description": "Second number",
			},
		},
		"required": []string{"a", "b"},
	}
}

func (t *SimpleAddTool) Invoke(ctx context.Context, paramsJSON string) (string, error) {
	var params struct {
		A json.Number `json:"a"`
		B json.Number `json:"b"`
	}

	d := json.NewDecoder(strings.NewReader(paramsJSON))
	d.UseNumber()
	if err := d.Decode(&params); err != nil {
		return "", fmt.Errorf("failed to parse parameters: %w", err)
	}

	// Convert json.Number to integers
	a, err := params.A.Int64()
	if err != nil {
		return "", fmt.Errorf("failed to convert a to integer: %w", err)
	}

	b, err := params.B.Int64()
	if err != nil {
		return "", fmt.Errorf("failed to convert b to integer: %w", err)
	}

	result := a + b
	return strconv.FormatInt(result, 10), nil
}

// Divide tool implementation
type DivideTool struct{}

func (t *DivideTool) Name() string {
	return "divide"
}

func (t *DivideTool) Description() string {
	return "Divides two numbers"
}

func (t *DivideTool) ParamsJSONSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"a": map[string]any{
				"type":        "number",
				"description": "Dividend",
			},
			"b": map[string]any{
				"type":        "number",
				"description": "Divisor",
			},
		},
		"required": []string{"a", "b"},
	}
}

func (t *DivideTool) Invoke(ctx context.Context, paramsJSON string) (string, error) {
	var params struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}

	if err := json.Unmarshal([]byte(paramsJSON), &params); err != nil {
		return "", fmt.Errorf("failed to parse parameters: %w", err)
	}

	if params.B == 0 {
		return "", errors.New("cannot divide by zero")
	}

	result := params.A / params.B
	return fmt.Sprintf("%f", result), nil
}

func TestNewFunctionTool(t *testing.T) {
	// Create a basic function tool
	tool, err := NewFunctionTool(add)
	assert.NoError(t, err, "Tool creation should not return an error")
	assert.NotNil(t, tool, "Tool should not be nil")

	// Check default name
	assert.Equal(t, "add", tool.Name(), "Default name should be the function name")

	// Check default description
	assert.Equal(t, "No description provided", tool.Description(), "Default description is incorrect")

	// Create a tool with custom name and description
	customName := "addition"
	customDescription := "Adds two numbers together"
	tool, err = NewFunctionTool(add, FunctionToolOption{
		NameOverride:        customName,
		DescriptionOverride: customDescription,
	})
	assert.NoError(t, err, "Tool creation should not return an error")
	assert.Equal(t, customName, tool.Name(), "Custom name is not set correctly")
	assert.Equal(t, customDescription, tool.Description(), "Custom description is not set correctly")

	// Check error for non-function value
	_, err = NewFunctionTool("not a function")
	assert.Error(t, err, "Non-function value should return an error")
}

func TestFunctionToolParamsSchema(t *testing.T) {
	// Schema for basic function
	tool, _ := NewFunctionTool(add)
	schema := tool.ParamsJSONSchema()

	// Check basic schema structure
	assert.Equal(t, "object", schema["type"], "Schema type should be object")
	properties, ok := schema["properties"].(map[string]any)
	assert.True(t, ok, "Properties map should exist")

	// Check parameter count
	assert.Len(t, properties, 2, "Add should have 2 parameters")

	// Check required parameters
	required, ok := schema["required"].([]string)
	assert.True(t, ok, "Required list should exist")
	assert.Len(t, required, 2, "There should be 2 required parameters")

	// Schema for function with context
	ctxTool, _ := NewFunctionTool(processWithContext)
	ctxSchema := ctxTool.ParamsJSONSchema()
	ctxProperties, _ := ctxSchema["properties"].(map[string]any)

	// Check that context is excluded from schema
	assert.Len(t, ctxProperties, 1, "There should be 1 parameter excluding context")
}

// Test functions for simple tools
func TestSimpleAddTool(t *testing.T) {
	// Create simple test tool
	addTool := &SimpleAddTool{}

	// Check basic properties
	assert.Equal(t, "add", addTool.Name(), "Tool name is incorrect")
	assert.Equal(t, "Adds two numbers together", addTool.Description(), "Tool description is incorrect")

	// Check schema
	schema := addTool.ParamsJSONSchema()
	assert.Equal(t, "object", schema["type"], "Schema type should be object")

	// Invoke the tool
	paramsJSON := `{"a": 5, "b": 7}`
	result, err := addTool.Invoke(context.Background(), paramsJSON)
	assert.NoError(t, err, "Invoke should not return an error")
	assert.Equal(t, "12", result, "5 + 7 should be 12")

	// Test invalid JSON
	_, err = addTool.Invoke(context.Background(), "not a json")
	assert.Error(t, err, "Invalid JSON should return an error")

	// Test missing parameter
	_, err = addTool.Invoke(context.Background(), `{"a": 5}`)
	assert.Error(t, err, "Missing parameter should return an error")
}

// Test functions supporting floating point values
func TestDivideTool(t *testing.T) {
	// Test with FunctionTool
	divideTool, err := NewFunctionTool(divide)
	assert.NoError(t, err, "Tool creation should not return an error")

	// Check divide parameter names
	divideSchema := divideTool.ParamsJSONSchema()
	divideRequired := divideSchema["required"].([]string)
	t.Logf("Divide parameter names: %v", divideRequired)

	// Use custom tool
	divideCustomTool := &DivideTool{}

	// Invoke custom tool
	paramsJSON := `{"a": 10, "b": 2}`
	result, err := divideCustomTool.Invoke(context.Background(), paramsJSON)
	assert.NoError(t, err, "Invoke should not return an error")
	assert.Equal(t, "5.000000", result, "10 / 2 should be 5.000000")

	// Test division by zero
	paramsJSON = `{"a": 10, "b": 0}`
	_, err = divideCustomTool.Invoke(context.Background(), paramsJSON)
	assert.Error(t, err, "Division by zero should return an error")
	assert.Contains(t, err.Error(), "cannot divide by zero", "Error message is incorrect")
}

// Custom tool implementation test
type CustomTool struct {
	name        string
	description string
}

func (t *CustomTool) Name() string {
	return t.name
}

func (t *CustomTool) Description() string {
	return t.description
}

func (t *CustomTool) ParamsJSONSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"message": map[string]any{
				"type":        "string",
				"description": "Message to display",
			},
		},
		"required": []string{"message"},
	}
}

func (t *CustomTool) Invoke(ctx context.Context, paramsJSON string) (string, error) {
	var params struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal([]byte(paramsJSON), &params); err != nil {
		return "", err
	}
	return "Custom tool: " + params.Message, nil
}

func TestCustomTool(t *testing.T) {
	customTool := &CustomTool{
		name:        "custom_tool",
		description: "Test custom tool implementation",
	}

	// Check basic properties
	assert.Equal(t, "custom_tool", customTool.Name(), "Tool name is incorrect")
	assert.Equal(t, "Test custom tool implementation", customTool.Description(), "Tool description is incorrect")

	// Check schema
	schema := customTool.ParamsJSONSchema()
	assert.Equal(t, "object", schema["type"], "Schema type should be object")

	// Invoke the tool
	paramsJSON := `{"message": "hello"}`
	result, err := customTool.Invoke(context.Background(), paramsJSON)
	assert.NoError(t, err, "Invoke should not return an error")
	assert.Equal(t, "Custom tool: hello", result, "Tool output is incorrect")
}
