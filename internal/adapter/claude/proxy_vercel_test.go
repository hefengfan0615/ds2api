package claude

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type claudeProxyStoreStub struct {
	mapping map[string]string
}

func (s claudeProxyStoreStub) ClaudeMapping() map[string]string {
	return s.mapping
}

type openAIProxyStub struct {
	status int
	body   string
}

func (s openAIProxyStub) ChatCompletions(w http.ResponseWriter, _ *http.Request) {
	if s.status == 0 {
		s.status = http.StatusOK
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(s.status)
	_, _ = w.Write([]byte(s.body))
}

type openAIProxyCaptureStub struct {
	seenModel string
}

func (s *openAIProxyCaptureStub) ChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req map[string]any
	_ = json.NewDecoder(r.Body).Decode(&req)
	if m, ok := req["model"].(string); ok {
		s.seenModel = m
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"id":"ok","choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
}

func TestClaudeProxyViaOpenAIVercelPreparePassthrough(t *testing.T) {
	h := &Handler{OpenAI: openAIProxyStub{status: 200, body: `{"lease_id":"lease_123","payload":{"a":1}}`}}
	req := httptest.NewRequest(http.MethodPost, "/anthropic/v1/messages?__stream_prepare=1", strings.NewReader(`{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hi"}],"stream":true}`))
	rec := httptest.NewRecorder()

	h.Messages(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d body=%s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("expected json response, got err=%v body=%s", err, rec.Body.String())
	}
	if _, ok := out["lease_id"]; !ok {
		t.Fatalf("expected lease_id in prepare passthrough, got=%v", out)
	}
}

func TestClaudeProxyViaOpenAIPreservesClaudeMapping(t *testing.T) {
	openAI := &openAIProxyCaptureStub{}
	h := &Handler{
		Store:  claudeProxyStoreStub{mapping: map[string]string{"fast": "deepseek-chat", "slow": "deepseek-reasoner"}},
		OpenAI: openAI,
	}
	req := httptest.NewRequest(http.MethodPost, "/anthropic/v1/messages", strings.NewReader(`{"model":"claude-3-opus","messages":[{"role":"user","content":"hi"}],"stream":false}`))
	rec := httptest.NewRecorder()

	h.Messages(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d body=%s", rec.Code, rec.Body.String())
	}
	if got := strings.TrimSpace(openAI.seenModel); got != "deepseek-reasoner" {
		t.Fatalf("expected mapped proxy model deepseek-reasoner, got %q", got)
	}
}
