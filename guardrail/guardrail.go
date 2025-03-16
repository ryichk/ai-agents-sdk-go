// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package guardrail

import (
	"context"
)

type InputGuardrailResult struct {
	// Allowed indicates whether the input is allowed
	Allowed bool

	// Message is the message when the guardrail is tripped
	Message string
}

type OutputGuardrailResult struct {
	// Allowed indicates whether the output is allowed
	Allowed bool

	// Message is the message when the guardrail is tripped
	Message string

	// ModifiedOutput is the modified output (if the guardrail modifies the output)
	ModifiedOutput string
}

type InputGuardrail interface {
	Name() string
	Description() string
	Check(ctx context.Context, input string) (InputGuardrailResult, error)
}

type OutputGuardrail interface {
	Name() string
	Description() string
	Check(ctx context.Context, output string) (OutputGuardrailResult, error)
}

type FunctionInputGuardrail struct {
	name        string
	description string
	checkFunc   func(ctx context.Context, input string) (InputGuardrailResult, error)
}

func (g *FunctionInputGuardrail) Name() string {
	return g.name
}

func (g *FunctionInputGuardrail) Description() string {
	return g.description
}

func (g *FunctionInputGuardrail) Check(ctx context.Context, input string) (InputGuardrailResult, error) {
	return g.checkFunc(ctx, input)
}

type FunctionOutputGuardrail struct {
	name        string
	description string
	checkFunc   func(ctx context.Context, output string) (OutputGuardrailResult, error)
}

func (g *FunctionOutputGuardrail) Name() string {
	return g.name
}

func (g *FunctionOutputGuardrail) Description() string {
	return g.description
}

func (g *FunctionOutputGuardrail) Check(ctx context.Context, output string) (OutputGuardrailResult, error) {
	return g.checkFunc(ctx, output)
}

func NewInputGuardrail(name string, description string, checkFunc func(ctx context.Context, input string) (InputGuardrailResult, error)) InputGuardrail {
	return &FunctionInputGuardrail{
		name:        name,
		description: description,
		checkFunc:   checkFunc,
	}
}

func NewOutputGuardrail(name string, description string, checkFunc func(ctx context.Context, output string) (OutputGuardrailResult, error)) OutputGuardrail {
	return &FunctionOutputGuardrail{
		name:        name,
		description: description,
		checkFunc:   checkFunc,
	}
}
