// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package tool

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestFileSearchTool_Name(t *testing.T) {
	tool := NewFileSearchTool("test-vector-store")

	if tool.Name() != "file_search" {
		t.Errorf("Expected tool name to be 'file_search', got '%s'", tool.Name())
	}
}

func TestFileSearchTool_Description(t *testing.T) {
	vectorStoreName := "test-vector-store"
	tool := NewFileSearchTool(vectorStoreName)

	description := tool.Description()
	if !strings.Contains(description, vectorStoreName) {
		t.Errorf("Expected description to contain vector store name '%s', got '%s'", vectorStoreName, description)
	}
}

func TestFileSearchTool_ParamsJSONSchema(t *testing.T) {
	tool := NewFileSearchTool("test-vector-store")

	schema := tool.ParamsJSONSchema()

	// Check that schema is an object type
	if schemaType, ok := schema["type"].(string); !ok || schemaType != "object" {
		t.Errorf("Expected schema type to be 'object', got '%v'", schema["type"])
	}

	// Check that properties contains a query field
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("Expected properties to be a map, got %T", schema["properties"])
	}

	query, ok := properties["query"].(map[string]any)
	if !ok {
		t.Fatalf("Expected query to be a map, got %T", properties["query"])
	}

	if queryType, ok := query["type"].(string); !ok || queryType != "string" {
		t.Errorf("Expected query type to be 'string', got '%v'", query["type"])
	}

	// Check that query is required
	required, ok := schema["required"].([]string)
	if !ok {
		t.Fatalf("Expected required to be a string slice, got %T", schema["required"])
	}

	foundQuery := false
	for _, field := range required {
		if field == "query" {
			foundQuery = true
			break
		}
	}

	if !foundQuery {
		t.Errorf("Expected 'query' to be in required fields")
	}
}

func TestFileSearchTool_Invoke(t *testing.T) {
	vectorStoreName := "test-vector-store"
	tool := NewFileSearchTool(vectorStoreName)

	params := map[string]any{
		"query": "test query",
	}

	paramsJSON, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("Failed to marshal params: %v", err)
	}

	result, err := tool.Invoke(context.Background(), string(paramsJSON))
	if err != nil {
		t.Fatalf("Failed to invoke tool: %v", err)
	}

	// Check that result contains the query and vector store name
	if !strings.Contains(result, "test query") {
		t.Errorf("Expected result to contain query 'test query', got '%s'", result)
	}

	if !strings.Contains(result, vectorStoreName) {
		t.Errorf("Expected result to contain vector store name '%s', got '%s'", vectorStoreName, result)
	}
}

func TestFileSearchTool_WithOptions(t *testing.T) {
	tool := NewFileSearchTool("test-vector-store")

	// Test WithFilters
	filters := &FileSearchFilters{
		Type:        "document",
		Extension:   "pdf",
		Tags:        []string{"important", "report"},
		DocumentIDs: []string{"doc1", "doc2"},
	}

	tool = tool.WithFilters(filters)
	if tool.Filters != filters {
		t.Errorf("Expected Filters to be set correctly")
	}

	// Test WithMetadataQuery
	metadataQuery := map[string]any{
		"author": "John Doe",
		"year":   2023,
	}

	tool = tool.WithMetadataQuery(metadataQuery)
	if tool.MetadataQuery == nil {
		t.Errorf("Expected MetadataQuery to be set, but it is nil")
	}
	if author, ok := tool.MetadataQuery["author"].(string); !ok || author != "John Doe" {
		t.Errorf("Expected MetadataQuery[\"author\"] to be 'John Doe', got %v", tool.MetadataQuery["author"])
	}
	if year, ok := tool.MetadataQuery["year"].(int); !ok || year != 2023 {
		t.Errorf("Expected MetadataQuery[\"year\"] to be 2023, got %v", tool.MetadataQuery["year"])
	}

	// Test WithRankingOptions
	rankingOptions := &FileSearchRankingOptions{
		Recency:            true,
		RecencyWindow:      30,
		SemanticSimilarity: 0.8,
	}

	tool = tool.WithRankingOptions(rankingOptions)
	if tool.RankingOptions != rankingOptions {
		t.Errorf("Expected RankingOptions to be set correctly")
	}

	// Test WithMaxChunks
	tool = tool.WithMaxChunks(20)
	if tool.MaxChunks != 20 {
		t.Errorf("Expected MaxChunks to be 20, got %d", tool.MaxChunks)
	}
}
