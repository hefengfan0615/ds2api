package openai

import (
	"ds2api/internal/toolcall"
	"strings"

)

type toolStreamSieveState struct {
	pending          strings.Builder
	capture          strings.Builder
	capturing        bool
	recentTextTail   string
	pendingToolRaw   string
	pendingToolCalls []toolcall.ParsedToolCall
	disableDeltas    bool
	toolNameSent     bool
	toolName         string
	toolArgsStart    int
	toolArgsSent     int
	toolArgsString   bool
	toolArgsDone     bool
}

type toolStreamEvent struct {
	Content        string
	ToolCalls      []toolcall.ParsedToolCall
	ToolCallDeltas []toolCallDelta
}

type toolCallDelta struct {
	Index     int
	Name      string
	Arguments string
}

// Keep in sync with JS TOOL_SIEVE_CONTEXT_TAIL_LIMIT.
const toolSieveContextTailLimit = 2048

func (s *toolStreamSieveState) resetIncrementalToolState() {
	s.disableDeltas = false
	s.toolNameSent = false
	s.toolName = ""
	s.toolArgsStart = -1
	s.toolArgsSent = -1
	s.toolArgsString = false
	s.toolArgsDone = false
}

func (s *toolStreamSieveState) noteText(content string) {
	if content == "" {
		return
	}
	s.recentTextTail = appendTail(s.recentTextTail, content, toolSieveContextTailLimit)
}

func appendTail(prev, next string, max int) string {
	if max <= 0 {
		return ""
	}
	combined := prev + next
	if len(combined) <= max {
		return combined
	}
	return combined[len(combined)-max:]
}
