package toolcall

import (
	"regexp"
	"strings"
)

var textKVNamePattern = regexp.MustCompile(`(?is)function\.name:\s*([a-zA-Z0-9_\-.]+)`)

func parseTextKVToolCalls(text string) []ParsedToolCall {
	var out []ParsedToolCall
	matches := textKVNamePattern.FindAllStringSubmatchIndex(text, -1)
	if len(matches) == 0 {
		return nil
	}

	for i, match := range matches {
		name := text[match[2]:match[3]]

		offset := match[1]
		endSearch := len(text)
		if i+1 < len(matches) {
			endSearch = matches[i+1][0]
		}

		searchArea := text[offset:endSearch]
		argIdx := strings.Index(searchArea, "function.arguments:")
		if argIdx < 0 {
			continue
		}

		startIdx := offset + argIdx + len("function.arguments:")
		braceIdx := strings.IndexByte(text[startIdx:endSearch], '{')
		if braceIdx < 0 {
			continue
		}

		actualStart := startIdx + braceIdx
		objJson, _, ok := extractJSONObject(text, actualStart)
		if !ok {
			continue
		}

		input := parseToolCallInput(objJson)
		out = append(out, ParsedToolCall{
			Name:  name,
			Input: input,
		})
	}

	if len(out) == 0 {
		return nil
	}
	return out
}
