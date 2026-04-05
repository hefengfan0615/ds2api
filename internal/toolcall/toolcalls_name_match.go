package toolcall

import (
	"regexp"
	"strings"
)

var toolNameLoosePattern = regexp.MustCompile(`[^a-z0-9]+`)

func resolveAllowedToolNameWithLooseMatch(name string, allowed map[string]struct{}, allowedCanonical map[string]string) string {
	if _, ok := allowed[name]; ok {
		return name
	}
	lower := strings.ToLower(strings.TrimSpace(name))
	if canonical, ok := allowedCanonical[lower]; ok {
		return canonical
	}
	if idx := strings.LastIndex(lower, "."); idx >= 0 && idx < len(lower)-1 {
		if canonical, ok := allowedCanonical[lower[idx+1:]]; ok {
			return canonical
		}
	}
	loose := toolNameLoosePattern.ReplaceAllString(lower, "")
	if loose == "" {
		return ""
	}
	for candidateLower, canonical := range allowedCanonical {
		if toolNameLoosePattern.ReplaceAllString(candidateLower, "") == loose {
			return canonical
		}
	}
	return ""
}
