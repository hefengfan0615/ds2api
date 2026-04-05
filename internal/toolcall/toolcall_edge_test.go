package toolcall

import (
	"testing"
)

// ─── FormatOpenAIStreamToolCalls ─────────────────────────────────────

func TestFormatOpenAIStreamToolCalls(t *testing.T) {
	formatted := FormatOpenAIStreamToolCalls([]ParsedToolCall{
		{Name: "search", Input: map[string]any{"q": "test"}},
	})
	if len(formatted) != 1 {
		t.Fatalf("expected 1, got %d", len(formatted))
	}
	fn, _ := formatted[0]["function"].(map[string]any)
	if fn["name"] != "search" {
		t.Fatalf("unexpected function name: %#v", fn)
	}
	if formatted[0]["index"] != 0 {
		t.Fatalf("expected index 0, got %v", formatted[0]["index"])
	}
}

// ─── ParseToolCalls more edge cases ──────────────────────────────────

func TestParseToolCallsNoToolNames(t *testing.T) {
	text := `{"tool_calls":[{"name":"search","input":{"q":"go"}}]}`
	calls := ParseToolCalls(text, nil)
	if len(calls) != 1 {
		t.Fatalf("expected 1 call with nil tool names, got %d", len(calls))
	}
}

func TestParseToolCallsEmptyText(t *testing.T) {
	calls := ParseToolCalls("", []string{"search"})
	if len(calls) != 0 {
		t.Fatalf("expected 0 calls for empty text, got %d", len(calls))
	}
}

func TestParseToolCallsMultipleTools(t *testing.T) {
	text := `{"tool_calls":[{"name":"search","input":{"q":"go"}},{"name":"get_weather","input":{"city":"beijing"}}]}`
	calls := ParseToolCalls(text, []string{"search", "get_weather"})
	if len(calls) != 2 {
		t.Fatalf("expected 2 calls, got %d", len(calls))
	}
}

func TestParseToolCallsInputAsString(t *testing.T) {
	text := `{"tool_calls":[{"name":"search","input":"{\"q\":\"golang\"}"}]}`
	calls := ParseToolCalls(text, []string{"search"})
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Input["q"] != "golang" {
		t.Fatalf("expected parsed string input, got %#v", calls[0].Input)
	}
}

func TestParseToolCallsWithFunctionWrapper(t *testing.T) {
	text := `{"tool_calls":[{"function":{"name":"calc","arguments":{"x":1,"y":2}}}]}`
	calls := ParseToolCalls(text, []string{"calc"})
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Name != "calc" {
		t.Fatalf("expected calc, got %q", calls[0].Name)
	}
}

func TestParseStandaloneToolCallsFencedCodeBlock(t *testing.T) {
	fenced := "Here's an example:\n```json\n{\"tool_calls\":[{\"name\":\"search\",\"input\":{\"q\":\"go\"}}]}\n```\nDon't execute this."
	calls := ParseStandaloneToolCalls(fenced, []string{"search"})
	if len(calls) != 0 {
		t.Fatalf("expected fenced code block to be ignored, got %d calls", len(calls))
	}
}

// ─── looksLikeToolExampleContext ─────────────────────────────────────

func TestLooksLikeToolExampleContextNone(t *testing.T) {
	if looksLikeToolExampleContext("I will call the tool now") {
		t.Fatal("expected false for non-example context")
	}
}

func TestLooksLikeToolExampleContextFenced(t *testing.T) {
	if !looksLikeToolExampleContext("```json") {
		t.Fatal("expected true for fenced code block context")
	}
}
