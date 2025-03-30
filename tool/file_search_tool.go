// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package tool

import (
	"context"
	"encoding/json"
	"fmt"
)

// FileSearchTool is a hosted tool that lets the LLM search through a vector store.
// Currently only supported with OpenAI models, using the Responses API.
type FileSearchTool struct {
	// VectorStoreName is the name of the vector store to search.
	VectorStoreName string

	// Filters allows filtering the search results.
	Filters *FileSearchFilters

	// MetadataQuery allows filtering by document metadata.
	MetadataQuery map[string]any

	// RankingOptions controls how search results are ranked.
	RankingOptions *FileSearchRankingOptions

	// MaxChunks is the maximum number of chunks to return. Defaults to 10.
	MaxChunks int
}

// FileSearchFilters specifies filters for the file search.
type FileSearchFilters struct {
	// Type is the document type. For example, "document", "image", etc.
	Type string

	// Extension is the file extension. For example, "pdf", "png", etc.
	Extension string

	// Tags is a list of tags.
	Tags []string

	// DocumentIDs is a list of document IDs to restrict the search to.
	DocumentIDs []string
}

// FileSearchRankingOptions controls how search results are ranked.
type FileSearchRankingOptions struct {
	// Recency indicates whether to rank more recent chunks higher.
	Recency bool

	// RecencyWindow is the time window for recency ranking.
	RecencyWindow int

	// SemanticSimilarity indicates the weight given to semantic similarity
	// (vs lexical similarity).
	SemanticSimilarity float32
}

// NewFileSearchTool creates a new FileSearchTool with the specified vector store name.
func NewFileSearchTool(vectorStoreName string) *FileSearchTool {
	return &FileSearchTool{
		VectorStoreName: vectorStoreName,
		MaxChunks:       10, // Default value
	}
}

// WithFilters sets the filters for the FileSearchTool.
func (t *FileSearchTool) WithFilters(filters *FileSearchFilters) *FileSearchTool {
	t.Filters = filters
	return t
}

// WithMetadataQuery sets the metadata query for the FileSearchTool.
func (t *FileSearchTool) WithMetadataQuery(query map[string]any) *FileSearchTool {
	t.MetadataQuery = query
	return t
}

// WithRankingOptions sets the ranking options for the FileSearchTool.
func (t *FileSearchTool) WithRankingOptions(options *FileSearchRankingOptions) *FileSearchTool {
	t.RankingOptions = options
	return t
}

// WithMaxChunks sets the maximum number of chunks to return.
func (t *FileSearchTool) WithMaxChunks(maxChunks int) *FileSearchTool {
	t.MaxChunks = maxChunks
	return t
}

// Name returns the name of the tool.
func (t *FileSearchTool) Name() string {
	return "file_search"
}

// Description returns the description of the tool.
func (t *FileSearchTool) Description() string {
	return fmt.Sprintf("Search through files stored in the '%s' vector store. This tool can search through documents to find information.", t.VectorStoreName)
}

// ParamsJSONSchema returns the JSON schema for the tool's parameters.
func (t *FileSearchTool) ParamsJSONSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{
				"type":        "string",
				"description": "The search query",
			},
		},
		"required": []string{"query"},
	}
}

// Invoke executes the tool.
func (t *FileSearchTool) Invoke(ctx context.Context, paramsJSON string) (string, error) {
	// Parse parameters
	var params struct {
		Query string `json:"query"`
	}
	if err := json.Unmarshal([]byte(paramsJSON), &params); err != nil {
		return "", fmt.Errorf("failed to parse parameters: %w", err)
	}

	// Create the request payload
	payload := map[string]any{
		"vector_store": t.VectorStoreName,
		"query":        params.Query,
		"max_chunks":   t.MaxChunks,
	}

	// Add optional parameters if specified
	if t.Filters != nil {
		filters := make(map[string]any)
		if t.Filters.Type != "" {
			filters["type"] = t.Filters.Type
		}
		if t.Filters.Extension != "" {
			filters["extension"] = t.Filters.Extension
		}
		if len(t.Filters.Tags) > 0 {
			filters["tags"] = t.Filters.Tags
		}
		if len(t.Filters.DocumentIDs) > 0 {
			filters["document_ids"] = t.Filters.DocumentIDs
		}
		if len(filters) > 0 {
			payload["filters"] = filters
		}
	}

	if t.MetadataQuery != nil && len(t.MetadataQuery) > 0 {
		payload["metadata_query"] = t.MetadataQuery
	}

	if t.RankingOptions != nil {
		rankingOptions := make(map[string]any)
		rankingOptions["recency"] = t.RankingOptions.Recency
		if t.RankingOptions.RecencyWindow > 0 {
			rankingOptions["recency_window"] = t.RankingOptions.RecencyWindow
		}
		if t.RankingOptions.SemanticSimilarity > 0 {
			rankingOptions["semantic_similarity"] = t.RankingOptions.SemanticSimilarity
		}
		if len(rankingOptions) > 0 {
			payload["ranking_options"] = rankingOptions
		}
	}

	// In a real implementation, this would call the OpenAI Responses API
	// Since we can't make actual API calls, we'll return a simulated response
	return fmt.Sprintf("Search results for '%s' in vector store '%s':\n\n"+
		"1. Found document: Sample document with relevant content\n"+
		"2. Found document: Another document with matching information\n"+
		"Note: This is a simulated response. In a real implementation, "+
		"this would call the OpenAI Responses API to search the vector store.",
		params.Query, t.VectorStoreName), nil
}
