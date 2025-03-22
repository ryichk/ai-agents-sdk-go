// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package tool

// NewFunctionToolFromFunc wraps a Go function to be used as an agent tool.
//
// This is a simplified version of NewFunctionTool that takes just a function and a description.
// It automatically extracts the function name for use as the tool name.
//
// Example usage:
//
//	func getWeather(city string) (string, error) {
//		return "Weather information for " + city, nil
//	}
//
//	weatherTool, err := NewFunctionToolFromFunc(getWeather, "Gets weather information for a city")
//
// Args:
//   - fn: The Go function to wrap as a tool
//   - description: A description of what the tool does
//
// Returns:
//   - A Tool that wraps the provided function
//   - An error if the function cannot be converted to a tool
func NewFunctionToolFromFunc(fn any, description string) (Tool, error) {
	return NewFunctionTool(fn, FunctionToolOption{
		DescriptionOverride: description,
	})
}

// NewFunctionToolWithName wraps a function and sets a custom name and description.
//
// This function allows creating a tool with both a custom name and description,
// which is useful when the function name doesn't match the desired tool name.
//
// Example usage:
//
//	weatherTool, err := NewFunctionToolWithName(
//		getWeatherData,
//		"get_weather",
//		"Gets current weather information for the specified city"
//	)
//
// Args:
//   - fn: The Go function to wrap as a tool
//   - name: A custom name for the tool (overrides the function name)
//   - description: A description of what the tool does
//
// Returns:
//   - A Tool that wraps the provided function
//   - An error if the function cannot be converted to a tool
func NewFunctionToolWithName(fn any, name string, description string) (Tool, error) {
	return NewFunctionTool(fn, FunctionToolOption{
		NameOverride:        name,
		DescriptionOverride: description,
	})
}
