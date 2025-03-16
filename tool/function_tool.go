// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package tool

// NewFunctionToolFromFunc wraps a Go function to be used as an agent tool.
// In its simplest form, you just pass the function you want to wrap.
//
// Example:
// ```go
//
//	func getWeather(city string) string {
//		return fmt.Sprintf("The weather in %s is sunny.", city)
//	}
//
// weatherTool := NewFunctionToolFromFunc(getWeather, "Gets weather information for a city")
// agent.AddTool(weatherTool)
// ```
func NewFunctionToolFromFunc(fn any, description string) (Tool, error) {
	return NewFunctionTool(fn, FunctionToolOption{
		DescriptionOverride: description,
	})
}

// NewFunctionToolWithName wraps a function and sets a custom name and description.
func NewFunctionToolWithName(fn any, name string, description string) (Tool, error) {
	return NewFunctionTool(fn, FunctionToolOption{
		NameOverride:        name,
		DescriptionOverride: description,
	})
}
