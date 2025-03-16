// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package handoff

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// Mock agent for testing
type MockAgent struct {
	AgentName    string
	Instructions string
}

// Name returns the name of the agent
func (a *MockAgent) Name() string {
	return a.AgentName
}

// Function to create a new mock agent
func newMockAgent(name, instructions string) *MockAgent {
	return &MockAgent{
		AgentName:    name,
		Instructions: instructions,
	}
}

// TestData is a struct for testing JSON schema functionality
type TestData struct {
	Name    string  `json:"name"`
	Age     int     `json:"age"`
	IsValid bool    `json:"is_valid"`
	Score   float64 `json:"score"`
}

// HandoffInputTestData represents input data for handoff testing
type HandoffInputTestData struct {
	Reason   string `json:"reason"`
	Priority int    `json:"priority"`
}

func TestNewHandoff(t *testing.T) {
	// Create a mock agent
	targetAgent := newMockAgent("Target Agent", "This is a target agent")

	// Create a handoff
	description := "Test Handoff"
	handoff := NewHandoff(targetAgent, description)

	// Check basic properties of the handoff
	assert.Equal(t, targetAgent, handoff.TargetAgent(), "Target agent of handoff is incorrect")
	assert.Equal(t, description, handoff.Description(), "Handoff description is incorrect")
	assert.Equal(t, "simple_handoff", handoff.Name(), "Handoff name is incorrect")

	// For SimpleHandoff, ShouldHandoff always returns true
	shouldHandoff, err := handoff.ShouldHandoff(context.Background(), "Test input")
	assert.NoError(t, err, "ShouldHandoff should not return an error")
	assert.True(t, shouldHandoff, "ShouldHandoff for SimpleHandoff should return true")
}

func TestNewFunctionHandoff(t *testing.T) {
	// Create a mock agent
	targetAgent := newMockAgent("Target Agent", "This is a target agent")

	// Define a handoff function
	var handoffCalled bool
	handoffFunc := func(ctx context.Context, input string) (bool, error) {
		handoffCalled = true
		return input == "handoff", nil
	}

	// Create a handoff
	description := "Function Handoff"
	handoff := NewFunctionHandoff(targetAgent, description, handoffFunc)

	// Check basic properties of the handoff
	assert.Equal(t, targetAgent, handoff.TargetAgent(), "Target agent of handoff is incorrect")
	assert.Equal(t, description, handoff.Description(), "Handoff description is incorrect")
	assert.Equal(t, "function_handoff", handoff.Name(), "Handoff name is incorrect")

	// Test the handoff function - case where handoff should not occur
	shouldHandoff, err := handoff.ShouldHandoff(context.Background(), "no handoff")
	assert.NoError(t, err, "ShouldHandoff should not return an error")
	assert.True(t, handoffCalled, "Handoff function should be called")
	assert.False(t, shouldHandoff, "Should return false based on input")

	// Reset
	handoffCalled = false

	// Test the handoff function - case where handoff should occur
	shouldHandoff, err = handoff.ShouldHandoff(context.Background(), "handoff")
	assert.NoError(t, err, "ShouldHandoff should not return an error")
	assert.True(t, handoffCalled, "Handoff function should be called")
	assert.True(t, shouldHandoff, "Should return true based on input")
}

func TestNewPatternHandoff(t *testing.T) {
	// Create a mock agent
	targetAgent := newMockAgent("Email Agent", "This is an email agent")

	// Create a pattern handoff for email-related queries
	description := "Email Pattern Handoff"
	handoff, err := NewPatternHandoff(targetAgent, description, `(?i)email|mail|inbox`)
	assert.NoError(t, err, "NewPatternHandoff should not return an error")

	// Check basic properties of the handoff
	assert.Equal(t, targetAgent, handoff.TargetAgent(), "Target agent of handoff is incorrect")
	assert.Equal(t, description, handoff.Description(), "Handoff description is incorrect")
	assert.Equal(t, "pattern_handoff", handoff.Name(), "Handoff name is incorrect")

	// Test with non-matching input
	shouldHandoff, err := handoff.ShouldHandoff(context.Background(), "What is the weather today?")
	assert.NoError(t, err, "ShouldHandoff should not return an error")
	assert.False(t, shouldHandoff, "Should not handoff for non-matching input")

	// Test with matching input
	shouldHandoff, err = handoff.ShouldHandoff(context.Background(), "Check my email")
	assert.NoError(t, err, "ShouldHandoff should not return an error")
	assert.True(t, shouldHandoff, "Should handoff for matching input")

	// Test with matching input in different case
	shouldHandoff, err = handoff.ShouldHandoff(context.Background(), "How can I send an EMAIL?")
	assert.NoError(t, err, "ShouldHandoff should not return an error")
	assert.True(t, shouldHandoff, "Should handoff for matching input regardless of case")
}

func TestNewKeywordHandoff(t *testing.T) {
	// Create a mock agent
	targetAgent := newMockAgent("Support Agent", "This is a support agent")

	// Create a keyword handoff for support-related queries
	description := "Support Keyword Handoff"
	keywords := []string{"help", "support", "issue", "problem"}
	handoff := NewKeywordHandoff(targetAgent, description, keywords)

	// Check basic properties of the handoff
	assert.Equal(t, targetAgent, handoff.TargetAgent(), "Target agent of handoff is incorrect")
	assert.Equal(t, description, handoff.Description(), "Handoff description is incorrect")
	assert.Equal(t, "keyword_handoff", handoff.Name(), "Handoff name is incorrect")

	// Test with non-matching input
	shouldHandoff, err := handoff.ShouldHandoff(context.Background(), "What time is it?")
	assert.NoError(t, err, "ShouldHandoff should not return an error")
	assert.False(t, shouldHandoff, "Should not handoff for non-matching input")

	// Test with matching input
	shouldHandoff, err = handoff.ShouldHandoff(context.Background(), "I need help with my order")
	assert.NoError(t, err, "ShouldHandoff should not return an error")
	assert.True(t, shouldHandoff, "Should handoff for input containing 'help'")

	// Test with matching input in different case
	shouldHandoff, err = handoff.ShouldHandoff(context.Background(), "I have a PROBLEM with my account")
	assert.NoError(t, err, "ShouldHandoff should not return an error")
	assert.True(t, shouldHandoff, "Should handoff for input containing 'PROBLEM'")
}

func TestNewLanguageHandoff(t *testing.T) {
	// Create mock agents
	spanishAgent := newMockAgent("Spanish Agent", "This is a Spanish agent")
	frenchAgent := newMockAgent("French Agent", "This is a French agent")

	// Create language handoffs
	spanishHandoff := NewLanguageHandoff(spanishAgent, "Spanish Language Handoff", "es")
	frenchHandoff := NewLanguageHandoff(frenchAgent, "French Language Handoff", "fr")

	// Check basic properties of the Spanish handoff
	assert.Equal(t, spanishAgent, spanishHandoff.TargetAgent(), "Target agent of handoff is incorrect")
	assert.Equal(t, "Spanish Language Handoff", spanishHandoff.Description(), "Handoff description is incorrect")
	assert.Equal(t, "language_handoff", spanishHandoff.Name(), "Handoff name is incorrect")

	// Test Spanish detection
	shouldHandoff, err := spanishHandoff.ShouldHandoff(context.Background(), "Hola, ¿cómo estás? Necesito ayuda.")
	assert.NoError(t, err, "ShouldHandoff should not return an error")
	assert.True(t, shouldHandoff, "Should handoff for Spanish input")

	// Test Spanish non-detection for English
	shouldHandoff, err = spanishHandoff.ShouldHandoff(context.Background(), "Hello, how are you? I need help.")
	assert.NoError(t, err, "ShouldHandoff should not return an error")
	assert.False(t, shouldHandoff, "Should not handoff for English input to Spanish agent")

	// Test French detection
	shouldHandoff, err = frenchHandoff.ShouldHandoff(context.Background(), "Bonjour, comment allez-vous? J'ai besoin d'aide.")
	assert.NoError(t, err, "ShouldHandoff should not return an error")
	assert.True(t, shouldHandoff, "Should handoff for French input")
}

func TestNewFilteredHandoff(t *testing.T) {
	// Create a mock agent
	targetAgent := newMockAgent("Support Agent", "This is a support agent")

	// Create a base handoff
	baseHandoff := NewHandoff(targetAgent, "Base Handoff")

	// Define a filter function that keeps only user messages
	filterFunc := func(ctx context.Context, inputData *InputData) (*InputData, error) {
		filteredHistory := make([]map[string]any, 0)
		for _, item := range inputData.InputHistory {
			if role, ok := item["role"].(string); ok && role == "user" {
				filteredHistory = append(filteredHistory, item)
			}
		}
		inputData.InputHistory = filteredHistory
		return inputData, nil
	}

	// Create a filtered handoff
	filteredHandoff := NewFilteredHandoff(baseHandoff, filterFunc)

	// Check basic properties
	assert.Equal(t, targetAgent, filteredHandoff.TargetAgent(), "Target agent of handoff is incorrect")
	assert.Equal(t, baseHandoff.Description(), filteredHandoff.Description(), "Handoff description is incorrect")
	assert.Equal(t, "filtered_simple_handoff", filteredHandoff.Name(), "Handoff name is incorrect")

	// Test filter function
	inputData := &InputData{
		InputHistory: []map[string]any{
			{"role": "user", "content": "Hello"},
			{"role": "assistant", "content": "Hi"},
			{"role": "user", "content": "Help me"},
		},
	}

	filteredData, err := filteredHandoff.FilterInput(context.Background(), inputData)
	assert.NoError(t, err, "FilterInput should not return an error")
	assert.Len(t, filteredData.InputHistory, 2, "Should filter out assistant messages")

	// ShouldHandoff should delegate to base handoff
	shouldHandoff, err := filteredHandoff.ShouldHandoff(context.Background(), "test")
	assert.NoError(t, err, "ShouldHandoff should not return an error")
	assert.True(t, shouldHandoff, "Should delegate to base handoff, which always returns true")
}

func TestHandoffRegistry(t *testing.T) {
	// Create a registry with a 5-second minimum time between handoffs
	registry := NewHandoffRegistry(5 * time.Second)

	// First handoff should be allowed
	allowed, reason := registry.CanHandoff("agent1", "agent2")
	assert.True(t, allowed, "First handoff should be allowed")
	assert.Empty(t, reason, "No reason should be given for allowing handoff")

	// Immediate repeat of the same handoff should be blocked
	allowed, reason = registry.CanHandoff("agent1", "agent2")
	assert.False(t, allowed, "Immediate repeat handoff should be blocked")
	assert.Contains(t, reason, "too soon", "Reason should indicate time constraint")

	// Check for loop detection
	allowed, reason = registry.CanHandoff("agent2", "agent1")
	assert.False(t, allowed, "Potential loop should be detected")
	assert.Contains(t, reason, "loop detected", "Reason should indicate loop detection")
}

func TestHandoffInterface(t *testing.T) {
	// Create a mock agent
	targetAgent := newMockAgent("Target Agent", "This is a target agent")

	// Create a SimpleHandoff
	simpleHandoff := NewHandoff(targetAgent, "Simple Handoff")

	// Check Handoff as an interface
	var handoff Handoff = simpleHandoff
	assert.NotNil(t, handoff, "Handoff interface should not be nil")
	assert.Equal(t, targetAgent, handoff.TargetAgent(), "TargetAgent should return the correct agent")
	assert.Equal(t, "Simple Handoff", handoff.Description(), "Description should return the correct description")

	// Create a FunctionHandoff
	functionHandoff := NewFunctionHandoff(targetAgent, "Function Handoff", func(ctx context.Context, input string) (bool, error) {
		return true, nil
	})

	// Check Handoff as an interface
	handoff = functionHandoff
	assert.NotNil(t, handoff, "Handoff interface should not be nil")
	assert.Equal(t, targetAgent, handoff.TargetAgent(), "TargetAgent should return the correct agent")
	assert.Equal(t, "Function Handoff", handoff.Description(), "Description should return the correct description")

	// Create a PatternHandoff
	patternHandoff, _ := NewPatternHandoff(targetAgent, "Pattern Handoff", "test")

	// Check Handoff as an interface
	handoff = patternHandoff
	assert.NotNil(t, handoff, "Handoff interface should not be nil")
	assert.Equal(t, targetAgent, handoff.TargetAgent(), "TargetAgent should return the correct agent")
	assert.Equal(t, "Pattern Handoff", handoff.Description(), "Description should return the correct description")
}

// TestToolNameAndDescription tests the custom tool name and description features
func TestToolNameAndDescription(t *testing.T) {
	// Create a mock agent
	targetAgent := newMockAgent("Support Agent", "This is a support agent")

	// Create a handoff with options
	customToolName := "custom_tool_name"
	customToolDesc := "Custom tool description"
	handoff := NewHandoffWithOptions(targetAgent, "Test Handoff", Options{
		ToolName:        customToolName,
		ToolDescription: customToolDesc,
	})

	// Check tool name and description
	assert.Equal(t, customToolName, handoff.ToolName(), "Custom tool name should be used")
	assert.Equal(t, customToolDesc, handoff.ToolDescription(), "Custom tool description should be used")

	// Test default tool name and description
	simpleHandoff := NewHandoff(targetAgent, "Simple Handoff")
	assert.Equal(t, "transfer_to_support_agent", simpleHandoff.ToolName(), "Default tool name should be generated correctly")
	assert.Contains(t, simpleHandoff.ToolDescription(), "Support Agent", "Default tool description should contain agent name")
}

// TestHandoffCallback tests the handoff callback functionality
func TestHandoffCallback(t *testing.T) {
	// Create a mock agent
	targetAgent := newMockAgent("Support Agent", "This is a support agent")

	// Define variables to track callback execution
	var callbackCalled bool
	var receivedInputJSON string

	// Create a callback function
	callback := func(ctx context.Context, inputData *InputData, inputJSON string) error {
		callbackCalled = true
		receivedInputJSON = inputJSON
		return nil
	}

	// Create a handoff with callback
	handoff := NewHandoffWithOptions(targetAgent, "Test Handoff", Options{
		OnHandoff: callback,
	})

	// Test the callback
	inputData := &InputData{
		InputHistory: []map[string]any{
			{"role": "user", "content": "Help me"},
		},
	}
	testJSON := `{"reason": "Test callback"}`

	err := handoff.OnHandoff(context.Background(), inputData, testJSON)
	assert.NoError(t, err, "OnHandoff should not return an error")
	assert.True(t, callbackCalled, "Callback function should be called")
	assert.Equal(t, testJSON, receivedInputJSON, "Callback should receive the correct JSON input")
}

// TestJSONSchema tests JSON schema creation and validation
func TestJSONSchema(t *testing.T) {
	// Create a test struct
	testStruct := TestData{
		Name:    "Test",
		Age:     30,
		IsValid: true,
		Score:   95.5,
	}

	// Create a JSON schema from the struct
	schema := CreateJSONSchema(testStruct, []string{"name", "age"})

	// Verify schema properties
	assert.Equal(t, "object", schema["type"], "Schema type should be object")
	assert.Contains(t, schema, "properties", "Schema should have properties")
	assert.Contains(t, schema, "required", "Schema should have required fields")

	properties := schema["properties"].(map[string]any)
	assert.Contains(t, properties, "name", "Properties should contain name field")
	assert.Contains(t, properties, "age", "Properties should contain age field")
	assert.Contains(t, properties, "is_valid", "Properties should contain is_valid field")
	assert.Contains(t, properties, "score", "Properties should contain score field")

	// Test valid JSON validation
	validJSON := `{"name": "Test", "age": 25, "is_valid": true, "score": 90.5}`
	data, err := ValidateJSON(validJSON, schema)
	assert.NoError(t, err, "Valid JSON should validate without errors")
	assert.Equal(t, "Test", data["name"], "Name should be correctly parsed")
	assert.Equal(t, float64(25), data["age"], "Age should be correctly parsed")

	// Test invalid JSON validation (missing required field)
	invalidJSON := `{"name": "Test", "is_valid": true}`
	_, err = ValidateJSON(invalidJSON, schema)
	assert.Error(t, err, "Invalid JSON should return an error")
	assert.Contains(t, err.Error(), "missing required field", "Error should indicate missing field")
}

// TestHandoffWithJSONInput tests a handoff with JSON input schema
func TestHandoffWithJSONInput(t *testing.T) {
	// Create a mock agent
	targetAgent := newMockAgent("Support Agent", "This is a support agent")

	// Create a JSON schema
	schema := JSONSchema{
		"type": "object",
		"properties": map[string]any{
			"reason": map[string]any{
				"type": "string",
			},
			"priority": map[string]any{
				"type": "integer",
			},
		},
		"required": []string{"reason"},
	}

	// Create a handoff with the schema
	handoff := NewHandoffWithOptions(targetAgent, "Test Handoff", Options{
		InputJSONSchema: schema,
	})

	// Verify schema
	handoffSchema := handoff.InputJSONSchema()
	assert.Equal(t, schema, handoffSchema, "Input JSON schema should match")

	// Track callback execution
	var receivedJSON string
	callback := func(ctx context.Context, inputData *InputData, inputJSON string) error {
		receivedJSON = inputJSON

		// Validate the JSON against our schema
		_, err := ValidateJSON(inputJSON, schema)
		return err
	}

	// Update the handoff with the callback
	handoffWithCallback := NewHandoffWithOptions(targetAgent, "Test Handoff", Options{
		InputJSONSchema: schema,
		OnHandoff:       callback,
	})

	// Test with valid JSON
	validJSON := `{"reason": "Customer complaint", "priority": 1}`
	inputData := &InputData{}
	err := handoffWithCallback.OnHandoff(context.Background(), inputData, validJSON)
	assert.NoError(t, err, "Valid JSON should not cause an error")
	assert.Equal(t, validJSON, receivedJSON, "Callback should receive the correct JSON")

	// Test with invalid JSON (missing required field)
	invalidJSON := `{"priority": 1}`
	err = handoffWithCallback.OnHandoff(context.Background(), inputData, invalidJSON)
	assert.Error(t, err, "Invalid JSON should cause an error")
	assert.Contains(t, err.Error(), "missing required field", "Error should indicate missing field")
}

// TestHandoffWithStructJSONSchema tests a handoff with JSON schema derived from a struct
func TestHandoffWithStructJSONSchema(t *testing.T) {
	// Create a mock agent
	targetAgent := newMockAgent("Support Agent", "This is a support agent")

	// Create a JSON schema from a struct
	schema := CreateJSONSchema(&HandoffInputTestData{}, []string{"reason"})

	// Create a handoff with the schema
	handoff := NewHandoffWithOptions(targetAgent, "Test Handoff", Options{
		InputJSONSchema: schema,
	})

	// Verify schema
	handoffSchema := handoff.InputJSONSchema()
	assert.Equal(t, schema, handoffSchema, "Input JSON schema should match")

	// Test JSON decoding into the struct
	testJSON := `{"reason": "Test reason", "priority": 2}`

	// Parse the JSON
	var data HandoffInputTestData
	err := json.Unmarshal([]byte(testJSON), &data)
	assert.NoError(t, err, "JSON unmarshaling should succeed")
	assert.Equal(t, "Test reason", data.Reason, "Reason should be correctly parsed")
	assert.Equal(t, 2, data.Priority, "Priority should be correctly parsed")

	// Test validation
	_, err = ValidateJSON(testJSON, schema)
	assert.NoError(t, err, "JSON validation should succeed")

	// Test with invalid JSON
	invalidJSON := `{"priority": 2}`
	_, err = ValidateJSON(invalidJSON, schema)
	assert.Error(t, err, "Invalid JSON should cause an error")
}
