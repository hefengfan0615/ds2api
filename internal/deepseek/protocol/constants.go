package protocol

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math/rand"
	"time"
)

const (
	DeepSeekHost                 = "chat.deepseek.com"
	DeepSeekLoginURL             = "https://chat.deepseek.com/api/v0/users/login"
	DeepSeekCreateSessionURL     = "https://chat.deepseek.com/api/v0/chat_session/create"
	DeepSeekCreatePowURL         = "https://chat.deepseek.com/api/v0/chat/create_pow_challenge"
	DeepSeekCompletionURL        = "https://chat.deepseek.com/api/v0/chat/completion"
	DeepSeekContinueURL          = "https://chat.deepseek.com/api/v0/chat/continue"
	DeepSeekUploadFileURL        = "https://chat.deepseek.com/api/v0/file/upload_file"
	DeepSeekFetchFilesURL        = "https://chat.deepseek.com/api/v0/file/fetch_files"
	DeepSeekFetchSessionURL      = "https://chat.deepseek.com/api/v0/chat_session/fetch_page"
	DeepSeekDeleteSessionURL     = "https://chat.deepseek.com/api/v0/chat_session/delete"
	DeepSeekDeleteAllSessionsURL = "https://chat.deepseek.com/api/v0/chat_session/delete_all"
	DeepSeekCompletionTargetPath = "/api/v0/chat/completion"
	DeepSeekUploadTargetPath     = "/api/v0/file/upload_file"
)

var defaultStaticBaseHeaders = map[string]string{
	"Host":           "chat.deepseek.com",
	"Accept":         "application/json",
	"Content-Type":   "application/json",
	"accept-charset": "UTF-8",
}

var defaultSkipContainsPatterns = []string{
	"quasi_status",
	"elapsed_secs",
	"token_usage",
	"pending_fragment",
	"conversation_mode",
	"fragments/-1/status",
	"fragments/-2/status",
	"fragments/-3/status",
}

var defaultSkipExactPaths = []string{
	"response/search_status",
}

var ClientVersion string
var BaseHeaders = map[string]string{}
var SkipContainsPatterns = cloneStringSlice(defaultSkipContainsPatterns)
var SkipExactPathSet = toStringSet(defaultSkipExactPaths)

type clientConstants struct {
	Name            string `json:"name"`
	Platform        string `json:"platform"`
	Version         string `json:"version"`
	AndroidAPILevel string `json:"android_api_level"`
	Locale          string `json:"locale"`
}

type sharedConstants struct {
	Client              clientConstants   `json:"client"`
	BaseHeaders         map[string]string `json:"base_headers"`
	SkipContainsPattern []string          `json:"skip_contains_patterns"`
	SkipExactPaths      []string          `json:"skip_exact_paths"`
}

//go:embed constants_shared.json
var sharedConstantsJSON []byte

var (
	userAgentVariations = []string{
		"DeepSeek/2.0.4 Android/35",
		"DeepSeek/2.0.3 Android/34",
		"DeepSeek/2.0.2 Android/33",
		"DeepSeek/2.0.1 Android/32",
		"DeepSeek/1.9.9 Android/31",
	}
	acceptLanguageVariations = []string{
		"zh-CN,zh;q=0.9,en;q=0.8",
		"zh-CN,zh;q=0.8,en;q=0.7",
		"zh,en;q=0.9,zh-CN;q=0.8",
		"en,zh;q=0.8,zh-CN;q=0.7",
	}
)

func init() {
	rand.Seed(time.Now().UnixNano())
	cfg := sharedConstants{}
	if err := json.Unmarshal(sharedConstantsJSON, &cfg); err != nil {
		panic(fmt.Errorf("load DeepSeek shared constants: %w", err))
	}
	applySharedConstants(cfg)
}

func applySharedConstants(cfg sharedConstants) {
	client := normalizeClientConstants(cfg.Client)
	ClientVersion = client.Version
	// For initialization, use non-randomized headers
	BaseHeaders = buildBaseHeadersWithOptions(client, cfg.BaseHeaders, false)
	SkipContainsPatterns = cloneStringSlice(defaultSkipContainsPatterns)
	if len(cfg.SkipContainsPattern) > 0 {
		SkipContainsPatterns = cloneStringSlice(cfg.SkipContainsPattern)
	}
	SkipExactPathSet = toStringSet(defaultSkipExactPaths)
	if len(cfg.SkipExactPaths) > 0 {
		SkipExactPathSet = toStringSet(cfg.SkipExactPaths)
	}
}

// GetRandomizedHeaders returns a copy of BaseHeaders with randomized values
func GetRandomizedHeaders() map[string]string {
	out := cloneStringMap(BaseHeaders)
	// Add randomization for actual requests
	if _, hasUA := out["User-Agent"]; hasUA {
		// Re-randomize User-Agent
		userAgent := randomFromSlice(userAgentVariations)
		out["User-Agent"] = userAgent
	}
	if _, hasAcceptLang := out["Accept-Language"]; hasAcceptLang {
		out["Accept-Language"] = randomFromSlice(acceptLanguageVariations)
	}
	if _, hasXRequestedWith := out["X-Requested-With"]; !hasXRequestedWith && rand.Float32() > 0.7 {
		out["X-Requested-With"] = "XMLHttpRequest"
	}
	return out
}

func randomFromSlice(items []string) string {
	return items[rand.Intn(len(items))]
}

func buildBaseHeaders(client clientConstants, overrides map[string]string) map[string]string {
	// Keep non-randomized for backward compatibility and tests
	return buildBaseHeadersWithOptions(client, overrides, false)
}

func buildBaseHeadersWithOptions(client clientConstants, overrides map[string]string, randomize bool) map[string]string {
	out := cloneStringMap(defaultStaticBaseHeaders)
	for k, v := range overrides {
		if k == "" || v == "" {
			continue
		}
		out[k] = v
	}
	if client.Name != "" && client.Version != "" {
		// Always override the User-Agent regardless of what's in overrides (matches original behavior
		if randomize {
			userAgent := randomFromSlice(userAgentVariations)
			if rand.Float32() > 0.5 {
				userAgent = client.Name + "/" + client.Version
				if client.Platform == "android" && client.AndroidAPILevel != "" {
					userAgent += " Android/" + client.AndroidAPILevel
				}
			}
			out["User-Agent"] = userAgent
		} else {
			userAgent := client.Name + "/" + client.Version
			if client.Platform == "android" && client.AndroidAPILevel != "" {
				userAgent += " Android/" + client.AndroidAPILevel
			}
			out["User-Agent"] = userAgent
		}
	}
	if client.Platform != "" {
		// Always override x-client-platform
		out["x-client-platform"] = client.Platform
	}
	if client.Version != "" {
		// Always override x-client-version
		out["x-client-version"] = client.Version
	}
	if client.Locale != "" {
		// Always override x-client-locale
		out["x-client-locale"] = client.Locale
	}
	if _, hasAcceptLang := out["Accept-Language"]; !hasAcceptLang {
		if randomize {
			out["Accept-Language"] = randomFromSlice(acceptLanguageVariations)
		}
	}
	if _, hasXRequestedWith := out["X-Requested-With"]; !hasXRequestedWith && randomize && rand.Float32() > 0.7 {
		out["X-Requested-With"] = "XMLHttpRequest"
	}
	return out
}

func normalizeClientConstants(in clientConstants) clientConstants {
	if in.Name == "" {
		in.Name = "DeepSeek"
	}
	if in.Platform == "" {
		in.Platform = "android"
	}
	if in.AndroidAPILevel == "" {
		in.AndroidAPILevel = "35"
	}
	if in.Locale == "" {
		in.Locale = "zh_CN"
	}
	return in
}

func cloneStringMap(in map[string]string) map[string]string {
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func cloneStringSlice(in []string) []string {
	out := make([]string, len(in))
	copy(out, in)
	return out
}

func toStringSet(in []string) map[string]struct{} {
	out := make(map[string]struct{}, len(in))
	for _, v := range in {
		if v == "" {
			continue
		}
		out[v] = struct{}{}
	}
	return out
}

const (
	KeepAliveTimeout  = 5
	StreamIdleTimeout = 300
	MaxKeepaliveCount = 40
)
