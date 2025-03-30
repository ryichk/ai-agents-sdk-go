// Copyright (c) 2025 ryichk
// Licensed under the MIT License.
// This is a Go implementation inspired by OpenAI's Agents SDK for Python.

package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/ryichk/ai-agents-sdk-go/agent"
	"github.com/ryichk/ai-agents-sdk-go/model"
	"github.com/ryichk/ai-agents-sdk-go/runner"
	"github.com/ryichk/ai-agents-sdk-go/tool"
)

func main() {
	// OpenAI API Keyの取得
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is not set")
	}

	// OpenAI Providerの初期化
	provider, err := model.NewOpenAIProvider(model.OpenAIConfig{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("Failed to create OpenAI provider: %v", err)
	}

	// FileSearchToolの作成
	fileSearchTool := tool.NewFileSearchTool("documents-vector-store").
		WithFilters(&tool.FileSearchFilters{
			Type:      "document",
			Extension: "pdf",
		}).
		WithRankingOptions(&tool.FileSearchRankingOptions{
			Recency:            true,
			SemanticSimilarity: 0.7,
		}).
		WithMaxChunks(5)

	// エージェントの作成
	researchAgent := agent.New(
		"research-assistant",
		"あなたは研究アシスタントです。ファイル検索ツールを使用して、ユーザーの質問に答えるために必要な情報を検索します。",
	)

	// ツールの追加
	researchAgent.AddTool(fileSearchTool)

	// 実行設定
	config := runner.DefaultRunConfig()
	config.ModelProvider = provider
	config.Model = "gpt-4o"

	// エージェントの実行
	fmt.Println("エージェントを起動しています...")
	fmt.Println("質問: AI技術についてのレポートはありますか？")

	result, err := runner.RunWithConfig(
		context.Background(),
		researchAgent,
		"AI技術についてのレポートはありますか？",
		config,
	)

	if err != nil {
		log.Fatalf("Failed to run agent: %v", err)
	}

	fmt.Println("\n回答:")
	fmt.Println(result.FinalOutput)

	// メッセージ履歴を表示
	fmt.Println("\n会話履歴:")
	for i, message := range result.History {
		fmt.Printf("%d. %s: %s\n", i+1, message.Role, message.Content)

		// ツール呼び出しがある場合は表示
		for _, toolCall := range message.ToolCalls {
			fmt.Printf("   ツール呼び出し: %s\n", toolCall.Function.Name)
			fmt.Printf("   引数: %s\n", toolCall.Function.Arguments)
		}
	}
}
