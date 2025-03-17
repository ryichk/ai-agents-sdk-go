// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package agent

import (
	"context"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCloneAgent(t *testing.T) {
	originalAgent := New("Original Agent", "Original instructions")
	originalAgent.SetModel("gpt-4o")
	originalAgent.HandoffDescription = "Original handoff description"

	clonedAgent := originalAgent.Clone()

	assert.Equal(t, originalAgent.Name, clonedAgent.Name, "Name should be copied")
	assert.Equal(t, originalAgent.Instructions, clonedAgent.Instructions, "Instructions should be copied")
	assert.Equal(t, originalAgent.Model, clonedAgent.Model, "Model should be copied")
	assert.Equal(t, originalAgent.HandoffDescription, clonedAgent.HandoffDescription, "Handoff description should be copied")

	clonedAgent.SetModel("gpt-4-turbo")
	clonedAgent.HandoffDescription = "New handoff description"

	assert.Equal(t, "gpt-4o", originalAgent.Model, "Original agent's model should not be changed")
	assert.Equal(t, "Original handoff description", originalAgent.HandoffDescription, "Original agent's handoff description should not be changed")

	assert.Equal(t, "gpt-4-turbo", clonedAgent.Model, "Cloned agent's model should be changed")
	assert.Equal(t, "New handoff description", clonedAgent.HandoffDescription, "Cloned agent's handoff description should be changed")
}

func TestDynamicInstructions(t *testing.T) {
	staticAgent := New("Static Agent", "Static instructions")
	assert.Equal(t, "Static instructions", staticAgent.Instructions, "Static instructions should match")

	dynamicAgent := New("Dynamic Agent", "")
	dynamicAgent.SetDynamicInstructions(func(ctx context.Context) string {
		return "Dynamic instructions"
	})

	ctx := context.Background()

	instructions, err := dynamicAgent.GetInstructions(ctx)
	assert.NoError(t, err, "Getting dynamic instructions should not error")
	assert.Equal(t, "Dynamic instructions", instructions, "Dynamic instructions should match")

	asyncAgent := New("Async Agent", "")
	asyncAgent.SetAsyncDynamicInstructions(func(ctx context.Context) (string, error) {
		return "Async dynamic instructions", nil
	})

	asyncInstructions, err := asyncAgent.GetInstructions(ctx)
	assert.NoError(t, err, "Getting async dynamic instructions should not error")
	assert.Equal(t, "Async dynamic instructions", asyncInstructions, "Async dynamic instructions should match")
}

type TestOutputStruct struct {
	Name    string `json:"name"`
	Message string `json:"message"`
	Count   int    `json:"count"`
}

func TestOutputSchema(t *testing.T) {
	plainAgent := New("Plain Agent", "Plain text agent")
	assert.Nil(t, plainAgent.OutputType, "Plain agent should not have output type")

	structAgent := New("Struct Agent", "Structured output agent")
	structAgent.SetOutputType(reflect.TypeOf(TestOutputStruct{}))

	assert.NotNil(t, structAgent.OutputType, "Struct agent should have output type")
	assert.Equal(t, "TestOutputStruct", structAgent.OutputType.Name(), "Output type name should match")

	nameField, ok := structAgent.OutputType.FieldByName("Name")
	assert.True(t, ok, "Name field should exist")
	assert.Equal(t, "string", nameField.Type.Name(), "Name field should be string")

	messageField, ok := structAgent.OutputType.FieldByName("Message")
	assert.True(t, ok, "Message field should exist")
	assert.Equal(t, "string", messageField.Type.Name(), "Message field should be string")

	countField, ok := structAgent.OutputType.FieldByName("Count")
	assert.True(t, ok, "Count field should exist")
	assert.Equal(t, "int", countField.Type.Name(), "Count field should be int")

	assert.Equal(t, `json:"name"`, string(nameField.Tag), "Name field should have correct JSON tag")
	assert.Equal(t, `json:"message"`, string(messageField.Tag), "Message field should have correct JSON tag")
	assert.Equal(t, `json:"count"`, string(countField.Tag), "Count field should have correct JSON tag")
}

func TestCloneWithOptions(t *testing.T) {
	agent := New("Original", "Original instructions")
	agent.SetModel("gpt-3.5-turbo")

	// Clone with options
	cloned := agent.Clone(
		WithName("Cloned"),
		WithInstructions("New instructions"),
		WithModel("gpt-4"),
	)

	// Verify original agent is unchanged
	assert.Equal(t, "Original", agent.Name)
	assert.Equal(t, "Original instructions", agent.Instructions)
	assert.Equal(t, "gpt-3.5-turbo", agent.Model)

	// Verify cloned agent has new values
	assert.Equal(t, "Cloned", cloned.Name)
	assert.Equal(t, "New instructions", cloned.Instructions)
	assert.Equal(t, "gpt-4", cloned.Model)
}
