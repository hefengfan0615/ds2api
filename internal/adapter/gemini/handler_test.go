package gemini

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"

	"ds2api/internal/auth"
)

type testGeminiConfig struct{}

func (testGeminiConfig) ModelAliases() map[string]string { return nil }

type testGeminiAuth struct {
	a   *auth.RequestAuth
	err error
}

func (m testGeminiAuth) Determine(_ *http.Request) (*auth.RequestAuth, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.a != nil {
		return m.a, nil
	}
	return &auth.RequestAuth{
		UseConfigToken: false,
		DeepSeekToken:  "direct-token",
		CallerID:       "caller:test",
		TriedAccounts:  map[string]bool{},
	}, nil
}

func (testGeminiAuth) Release(_ *auth.RequestAuth) {}

type testGeminiDS struct {
	resp *http.Response
	err  error
}

func (m testGeminiDS) CreateSession(_ context.Context, _ *auth.RequestAuth, _ int) (string, error) {
	return "session-id", nil
}

func (m testGeminiDS) GetPow(_ context.Context, _ *auth.RequestAuth, _ int) (string, error) {
	return "pow", nil
}

func (m testGeminiDS) CallCompletion(_ context.Context, _ *auth.RequestAuth, _ map[string]any, _ string, _ int) (*http.Response, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.resp, nil
}

type geminiOpenAIErrorStub struct {
	status int
	body   string
}

func (s geminiOpenAIErrorStub) ChatCompletions(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(s.status)
	_, _ = w.Write([]byte(s.body))
}

type geminiOpenAISuccessStub struct {
	stream bool
	body   string
}

func (s geminiOpenAISuccessStub) ChatCompletions(w http.ResponseWriter, _ *http.Request) {
	if s.stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello \"},\"finish_reason\":null}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		return
	}
	out := s.body
	if strings.TrimSpace(out) == "" {
		out = `{"id":"chatcmpl-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"eval_javascript","arguments":"{\"code\":\"1+1\"}"}}]},"finish_reason":"tool_calls"}]}`
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(out))
}

func makeGeminiUpstreamResponse(lines ...string) *http.Response {
	body := strings.Join(lines, "\n")
	if !strings.HasSuffix(body, "\n") {
		body += "\n"
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func TestGeminiRoutesRegistered(t *testing.T) {
	h := &Handler{
		Store: testGeminiConfig{},
		Auth:  testGeminiAuth{err: auth.ErrUnauthorized},
	}
	r := chi.NewRouter()
	RegisterRoutes(r, h)

	paths := []string{
		"/v1beta/models/gemini-2.5-pro:generateContent",
		"/v1beta/models/gemini-2.5-pro:streamGenerateContent",
		"/v1/models/gemini-2.5-pro:generateContent",
		"/v1/models/gemini-2.5-pro:streamGenerateContent",
	}
	for _, path := range paths {
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`))
		rec := httptest.NewRecorder()
		r.ServeHTTP(rec, req)
		if rec.Code == http.StatusNotFound {
			t.Fatalf("expected route %s to be registered, got 404", path)
		}
	}
}

func TestGenerateContentReturnsFunctionCallParts(t *testing.T) {
	h := &Handler{
		Store: testGeminiConfig{},
		OpenAI: geminiOpenAISuccessStub{
			body: `{"id":"chatcmpl-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"eval_javascript","arguments":"{\"code\":\"1+1\"}"}}]},"finish_reason":"tool_calls"}]}`,
		},
	}
	r := chi.NewRouter()
	RegisterRoutes(r, h)

	body := `{
		"contents":[{"role":"user","parts":[{"text":"call tool"}]}],
		"tools":[{"functionDeclarations":[{"name":"eval_javascript","description":"eval","parameters":{"type":"object","properties":{"code":{"type":"string"}}}}]}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-pro:generateContent", strings.NewReader(body))
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}

	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response failed: %v", err)
	}
	candidates, _ := out["candidates"].([]any)
	if len(candidates) == 0 {
		t.Fatalf("expected non-empty candidates: %#v", out)
	}
	c0, _ := candidates[0].(map[string]any)
	content, _ := c0["content"].(map[string]any)
	parts, _ := content["parts"].([]any)
	if len(parts) == 0 {
		t.Fatalf("expected non-empty parts: %#v", content)
	}
	part0, _ := parts[0].(map[string]any)
	functionCall, _ := part0["functionCall"].(map[string]any)
	if functionCall["name"] != "eval_javascript" {
		t.Fatalf("expected functionCall name eval_javascript, got %#v", functionCall)
	}
}

func TestGenerateContentMixedToolSnippetAlsoTriggersFunctionCall(t *testing.T) {
	h := &Handler{Store: testGeminiConfig{}, OpenAI: geminiOpenAISuccessStub{}}
	r := chi.NewRouter()
	RegisterRoutes(r, h)

	body := `{
		"contents":[{"role":"user","parts":[{"text":"call tool"}]}],
		"tools":[{"functionDeclarations":[{"name":"eval_javascript","description":"eval","parameters":{"type":"object","properties":{"code":{"type":"string"}}}}]}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-pro:generateContent", strings.NewReader(body))
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response failed: %v", err)
	}
	candidates, _ := out["candidates"].([]any)
	c0, _ := candidates[0].(map[string]any)
	content, _ := c0["content"].(map[string]any)
	parts, _ := content["parts"].([]any)
	part0, _ := parts[0].(map[string]any)
	functionCall, _ := part0["functionCall"].(map[string]any)
	if functionCall["name"] != "eval_javascript" {
		t.Fatalf("expected functionCall name eval_javascript for mixed snippet, got %#v", functionCall)
	}
}

func TestStreamGenerateContentEmitsSSE(t *testing.T) {
	h := &Handler{
		Store:  testGeminiConfig{},
		OpenAI: geminiOpenAISuccessStub{stream: true},
	}
	r := chi.NewRouter()
	RegisterRoutes(r, h)

	body := `{"contents":[{"role":"user","parts":[{"text":"hello"}]}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/models/gemini-2.5-pro:streamGenerateContent?alt=sse", strings.NewReader(body))
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}

	frames := extractGeminiSSEFrames(t, rec.Body.String())
	if len(frames) == 0 {
		t.Fatalf("expected non-empty stream frames, body=%s", rec.Body.String())
	}
	last := frames[len(frames)-1]
	candidates, _ := last["candidates"].([]any)
	if len(candidates) == 0 {
		t.Fatalf("expected finish frame candidates, got %#v", last)
	}
	c0, _ := candidates[0].(map[string]any)
	content, _ := c0["content"].(map[string]any)
	if content == nil {
		t.Fatalf("expected non-null content in finish frame, got %#v", c0)
	}
	parts, _ := content["parts"].([]any)
	if len(parts) == 0 {
		t.Fatalf("expected non-empty parts in finish frame content, got %#v", content)
	}
}

func TestGenerateContentOpenAIProxyErrorUsesGeminiEnvelope(t *testing.T) {
	h := &Handler{
		Store:  testGeminiConfig{},
		OpenAI: geminiOpenAIErrorStub{status: http.StatusUnauthorized, body: `{"error":{"message":"invalid api key"}}`},
	}
	r := chi.NewRouter()
	RegisterRoutes(r, h)

	req := httptest.NewRequest(http.MethodPost, "/v1/models/gemini-2.5-pro:generateContent", strings.NewReader(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`))
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d body=%s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("expected json body: %v", err)
	}
	errObj, _ := out["error"].(map[string]any)
	if errObj["status"] != "UNAUTHENTICATED" {
		t.Fatalf("expected Gemini status UNAUTHENTICATED, got=%v", errObj["status"])
	}
	if errObj["message"] != "invalid api key" {
		t.Fatalf("expected parsed error message, got=%v", errObj["message"])
	}
}

func extractGeminiSSEFrames(t *testing.T, body string) []map[string]any {
	t.Helper()
	scanner := bufio.NewScanner(strings.NewReader(body))
	out := make([]map[string]any, 0, 4)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		raw := line
		if strings.HasPrefix(line, "data: ") {
			raw = strings.TrimSpace(strings.TrimPrefix(line, "data: "))
		}
		if raw == "" {
			continue
		}
		var frame map[string]any
		if err := json.Unmarshal([]byte(raw), &frame); err != nil {
			continue
		}
		out = append(out, frame)
	}
	return out
}
