// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"runtime"
	"strings"
)

// Tool represents a tool that can be used by agents
type Tool interface {
	Name() string
	Description() string

	// ParamsJSONSchema returns the JSON schema for the tool's parameters
	ParamsJSONSchema() map[string]any

	// Invoke executes the tool
	Invoke(ctx context.Context, paramsJSON string) (string, error)
}

// FunctionTool wraps a function as a tool
type FunctionTool struct {
	name          string
	description   string
	paramsSchema  map[string]any
	function      any
	reflectedFunc reflect.Value
	functionType  reflect.Type
}

func (t *FunctionTool) Name() string {
	return t.name
}

func (t *FunctionTool) Description() string {
	return t.description
}

// ParamsJSONSchema returns the JSON schema for the tool's parameters
func (t *FunctionTool) ParamsJSONSchema() map[string]any {
	return t.paramsSchema
}

// Invoke executes the tool
func (t *FunctionTool) Invoke(ctx context.Context, paramsJSON string) (string, error) {
	var params map[string]any
	if err := json.Unmarshal([]byte(paramsJSON), &params); err != nil {
		return "", fmt.Errorf("failed to parse parameters: %w", err)
	}

	args, err := t.prepareArgs(params)
	if err != nil {
		return "", err
	}

	// Call the function
	results := t.reflectedFunc.Call(args)

	// Process the result
	if len(results) == 0 {
		return "", nil
	}

	// Error check (if the last return value is an error)
	if t.functionType.NumOut() > 1 && t.functionType.Out(t.functionType.NumOut()-1).Implements(reflect.TypeOf((*error)(nil)).Elem()) {
		lastResult := results[len(results)-1]
		if !lastResult.IsNil() {
			return "", lastResult.Interface().(error)
		}
	}

	// Convert result to JSON
	result := results[0].Interface()
	resultJSON, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("failed to marshal result: %w", err)
	}

	return string(resultJSON), nil
}

// prepareArgs prepares the function arguments
func (t *FunctionTool) prepareArgs(params map[string]any) ([]reflect.Value, error) {
	// Implementation is simplified. In reality, more complex conversions might be needed
	args := make([]reflect.Value, t.functionType.NumIn())

	for i := 0; i < t.functionType.NumIn(); i++ {
		paramType := t.functionType.In(i)
		paramName := t.getParamName(i)

		// Special handling for context
		if paramType.Implements(reflect.TypeOf((*context.Context)(nil)).Elem()) {
			args[i] = reflect.ValueOf(context.Background()) // Simplified
			continue
		}

		// If parameter is not in the map
		paramValue, ok := params[paramName]
		if !ok {
			return nil, fmt.Errorf("missing parameter: %s", paramName)
		}

		// Convert the value
		convertedValue, err := convertValue(paramValue, paramType)
		if err != nil {
			return nil, err
		}

		args[i] = convertedValue
	}

	return args, nil
}

// getParamName gets the function parameter name
func (t *FunctionTool) getParamName(index int) string {
	// Implementation is simplified. In reality, parameter names can be obtained using reflection
	return fmt.Sprintf("param%d", index)
}

// convertValue converts a value to the specified type
func convertValue(value any, targetType reflect.Type) (reflect.Value, error) {
	// Implementation is simplified. In reality, more complex conversions might be needed
	reflectedValue := reflect.ValueOf(value)

	// If types match, return as is
	if reflectedValue.Type().AssignableTo(targetType) {
		return reflectedValue, nil
	}

	// Special handling for strings
	if targetType.Kind() == reflect.String && reflectedValue.Kind() != reflect.String {
		stringValue, err := json.Marshal(value)
		if err != nil {
			return reflect.Value{}, err
		}
		return reflect.ValueOf(string(stringValue)), nil
	}

	// Other conversions (in reality, more conversion logic would be needed)
	return reflect.Value{}, fmt.Errorf("cannot convert %v to %v", reflectedValue.Type(), targetType)
}

// FunctionToolOption represents options for creating a function tool.
// These options allow customizing the behavior and metadata of the tool.
type FunctionToolOption struct {
	// NameOverride allows specifying a custom name for the tool instead of using the function name.
	// This is useful for providing a more descriptive or standardized name for the tool.
	NameOverride string

	// DescriptionOverride allows providing a custom description for the tool.
	// The description explains what the tool does and helps the LLM understand when to use it.
	DescriptionOverride string
}

// NewFunctionTool creates a new tool from a function.
//
// This function allows converting any Go function into a tool that can be used by agents.
// By default, it will:
// 1. Extract the function name to use as the tool's name
// 2. Generate a JSON schema for the tool's parameters based on function signature
// 3. Use the provided description or a default one for the tool
//
// The function parameters will be automatically converted to corresponding JSON schema types.
// If the function has a context.Context parameter, it will be automatically populated during execution.
// The function should return a value that can be marshaled to JSON.
//
// Example usage:
//
//	func getWeather(city string) (string, error) {
//		// Implementation
//		return "Weather data for " + city, nil
//	}
//
//	weatherTool, err := NewFunctionTool(getWeather, FunctionToolOption{
//		DescriptionOverride: "Get weather information for the specified city",
//	})
//
// Args:
//   - function: The Go function to wrap as a tool
//   - options: Optional settings to customize the tool's name and description
//
// Returns:
//   - A FunctionTool that wraps the provided function
//   - An error if the function cannot be converted to a tool
func NewFunctionTool(function any, options ...FunctionToolOption) (*FunctionTool, error) {
	// Get function reflection info
	reflectedFunc := reflect.ValueOf(function)
	if reflectedFunc.Kind() != reflect.Func {
		return nil, fmt.Errorf("function must be a function, got %T", function)
	}

	functionType := reflectedFunc.Type()

	// Get function name
	funcName := runtime.FuncForPC(reflectedFunc.Pointer()).Name()
	// Remove package name
	parts := strings.Split(funcName, ".")
	name := parts[len(parts)-1]

	// Default description
	description := "No description provided"

	// Apply options
	for _, option := range options {
		if option.NameOverride != "" {
			name = option.NameOverride
		}
		if option.DescriptionOverride != "" {
			description = option.DescriptionOverride
		}
	}

	// Generate JSON schema for parameters
	paramsSchema := generateParamsSchema(functionType)

	return &FunctionTool{
		name:          name,
		description:   description,
		paramsSchema:  paramsSchema,
		function:      function,
		reflectedFunc: reflectedFunc,
		functionType:  functionType,
	}, nil
}

// generateParamsSchema generates JSON schema from function parameters
func generateParamsSchema(funcType reflect.Type) map[string]any {
	schema := map[string]any{
		"type":       "object",
		"properties": map[string]any{},
		"required":   []string{},
	}

	properties := schema["properties"].(map[string]any)
	required := schema["required"].([]string)

	for i := 0; i < funcType.NumIn(); i++ {
		paramType := funcType.In(i)

		// Skip context
		if paramType.Implements(reflect.TypeOf((*context.Context)(nil)).Elem()) {
			continue
		}

		paramName := fmt.Sprintf("param%d", i) // Simplified

		properties[paramName] = generateTypeSchema(paramType)
		required = append(required, paramName)
	}

	schema["required"] = required
	return schema
}

// generateTypeSchema generates JSON schema from type
func generateTypeSchema(t reflect.Type) map[string]any {
	schema := map[string]any{}

	switch t.Kind() {
	case reflect.String:
		schema["type"] = "string"
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		schema["type"] = "integer"
	case reflect.Float32, reflect.Float64:
		schema["type"] = "number"
	case reflect.Bool:
		schema["type"] = "boolean"
	case reflect.Slice, reflect.Array:
		schema["type"] = "array"
		schema["items"] = generateTypeSchema(t.Elem())
	case reflect.Map:
		schema["type"] = "object"
		// Map key/value type details are omitted for simplification
	case reflect.Struct:
		schema["type"] = "object"
		schema["properties"] = map[string]any{}
		required := []string{}

		for _, i := range make([]int, t.NumField()) {
			field := t.Field(i)
			// Skip unexported fields
			if field.PkgPath != "" {
				continue
			}

			// Get field name from JSON tag
			jsonTag := field.Tag.Get("json")
			fieldName := field.Name
			if jsonTag != "" {
				parts := strings.Split(jsonTag, ",")
				if parts[0] != "" && parts[0] != "-" {
					fieldName = parts[0]
				}
			}

			properties := schema["properties"].(map[string]any)
			properties[fieldName] = generateTypeSchema(field.Type)
			required = append(required, fieldName)
		}

		schema["required"] = required
	default:
		// Other types (interfaces, channels, etc.) are treated as strings
		schema["type"] = "string"
	}

	return schema
}
