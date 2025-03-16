// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package guardrail

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestInputGuardrail(t *testing.T) {
	// Create input guardrail that allows input
	allowName := "Allow Guardrail"
	allowDescription := "Guardrail that always allows input"
	allowFunc := func(ctx context.Context, input string) (InputGuardrailResult, error) {
		return InputGuardrailResult{
			Allowed: true,
			Message: "",
		}, nil
	}

	allowGuardrail := NewInputGuardrail(allowName, allowDescription, allowFunc)

	// Check basic properties
	assert.Equal(t, allowName, allowGuardrail.Name(), "Guardrail name is incorrect")
	assert.Equal(t, allowDescription, allowGuardrail.Description(), "Guardrail description is incorrect")

	// Test the check function
	result, err := allowGuardrail.Check(context.Background(), "Test input")
	assert.NoError(t, err, "No error should occur")
	assert.True(t, result.Allowed, "Input should be allowed")
	assert.Empty(t, result.Message, "Message should be empty")

	// Create input guardrail that denies input
	denyName := "Deny Guardrail"
	denyDescription := "Guardrail that always denies input"
	denyMessage := "This input is not allowed"
	denyFunc := func(ctx context.Context, input string) (InputGuardrailResult, error) {
		return InputGuardrailResult{
			Allowed: false,
			Message: denyMessage,
		}, nil
	}

	denyGuardrail := NewInputGuardrail(denyName, denyDescription, denyFunc)

	// Check basic properties
	assert.Equal(t, denyName, denyGuardrail.Name(), "Guardrail name is incorrect")
	assert.Equal(t, denyDescription, denyGuardrail.Description(), "Guardrail description is incorrect")

	// Test the check function
	result, err = denyGuardrail.Check(context.Background(), "Test input")
	assert.NoError(t, err, "No error should occur")
	assert.False(t, result.Allowed, "Input should be denied")
	assert.Equal(t, denyMessage, result.Message, "Message is incorrect")

	// Create input guardrail that returns an error
	errorName := "Error Guardrail"
	errorDescription := "Guardrail that always returns an error"
	expectedError := errors.New("Test error")
	errorFunc := func(ctx context.Context, input string) (InputGuardrailResult, error) {
		return InputGuardrailResult{}, expectedError
	}

	errorGuardrail := NewInputGuardrail(errorName, errorDescription, errorFunc)

	// Test the check function
	_, err = errorGuardrail.Check(context.Background(), "Test input")
	assert.Error(t, err, "Error should occur")
	assert.Equal(t, expectedError, err, "Returned error does not match")
}

func TestOutputGuardrail(t *testing.T) {
	// Create output guardrail that allows output
	allowName := "Allow Guardrail"
	allowDescription := "Guardrail that always allows output"
	allowFunc := func(ctx context.Context, output string) (OutputGuardrailResult, error) {
		return OutputGuardrailResult{
			Allowed:        true,
			Message:        "",
			ModifiedOutput: "",
		}, nil
	}

	allowGuardrail := NewOutputGuardrail(allowName, allowDescription, allowFunc)

	// Check basic properties
	assert.Equal(t, allowName, allowGuardrail.Name(), "Guardrail name is incorrect")
	assert.Equal(t, allowDescription, allowGuardrail.Description(), "Guardrail description is incorrect")

	// Test the check function
	result, err := allowGuardrail.Check(context.Background(), "Test output")
	assert.NoError(t, err, "No error should occur")
	assert.True(t, result.Allowed, "Output should be allowed")
	assert.Empty(t, result.Message, "Message should be empty")
	assert.Empty(t, result.ModifiedOutput, "Modified output should be empty")

	// Create output guardrail that denies output
	denyName := "Deny Guardrail"
	denyDescription := "Guardrail that always denies output"
	denyMessage := "This output is not allowed"
	denyFunc := func(ctx context.Context, output string) (OutputGuardrailResult, error) {
		return OutputGuardrailResult{
			Allowed:        false,
			Message:        denyMessage,
			ModifiedOutput: "",
		}, nil
	}

	denyGuardrail := NewOutputGuardrail(denyName, denyDescription, denyFunc)

	// Check basic properties
	assert.Equal(t, denyName, denyGuardrail.Name(), "Guardrail name is incorrect")
	assert.Equal(t, denyDescription, denyGuardrail.Description(), "Guardrail description is incorrect")

	// Test the check function
	result, err = denyGuardrail.Check(context.Background(), "Test output")
	assert.NoError(t, err, "No error should occur")
	assert.False(t, result.Allowed, "Output should be denied")
	assert.Equal(t, denyMessage, result.Message, "Message is incorrect")
	assert.Empty(t, result.ModifiedOutput, "Modified output should be empty")

	// Create guardrail that modifies output
	modifyName := "Modify Guardrail"
	modifyDescription := "Guardrail that modifies output"
	modifyMessage := "Output has been modified"
	originalOutput := "Original output"
	modifiedOutput := "Modified output"
	modifyFunc := func(ctx context.Context, output string) (OutputGuardrailResult, error) {
		return OutputGuardrailResult{
			Allowed:        true,
			Message:        modifyMessage,
			ModifiedOutput: modifiedOutput,
		}, nil
	}

	modifyGuardrail := NewOutputGuardrail(modifyName, modifyDescription, modifyFunc)

	// Test the check function
	result, err = modifyGuardrail.Check(context.Background(), originalOutput)
	assert.NoError(t, err, "No error should occur")
	assert.True(t, result.Allowed, "Output should be allowed")
	assert.Equal(t, modifyMessage, result.Message, "Message is incorrect")
	assert.Equal(t, modifiedOutput, result.ModifiedOutput, "Modified output is incorrect")

	// Create output guardrail that returns an error
	errorName := "Error Guardrail"
	errorDescription := "Guardrail that always returns an error"
	expectedError := errors.New("Test error")
	errorFunc := func(ctx context.Context, output string) (OutputGuardrailResult, error) {
		return OutputGuardrailResult{}, expectedError
	}

	errorGuardrail := NewOutputGuardrail(errorName, errorDescription, errorFunc)

	// Test the check function
	_, err = errorGuardrail.Check(context.Background(), "Test output")
	assert.Error(t, err, "Error should occur")
	assert.Equal(t, expectedError, err, "Returned error does not match")
}
