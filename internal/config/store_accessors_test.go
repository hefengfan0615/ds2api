package config

import "testing"

func TestStoreHistorySplitAccessors(t *testing.T) {
	store := &Store{cfg: Config{}}
	if store.HistorySplitEnabled() {
		t.Fatal("expected history split disabled by default")
	}
	if got := store.HistorySplitTriggerAfterTurns(); got != 1 {
		t.Fatalf("default history split trigger_after_turns=%d want=1", got)
	}

	enabled := true
	turns := 3
	store.cfg.HistorySplit = HistorySplitConfig{
		Enabled:           &enabled,
		TriggerAfterTurns: &turns,
	}

	if !store.HistorySplitEnabled() {
		t.Fatal("expected history split enabled")
	}
	if got := store.HistorySplitTriggerAfterTurns(); got != 3 {
		t.Fatalf("history split trigger_after_turns=%d want=3", got)
	}
}

func TestStoreHistorySplitDisabledConfigStaysDisabled(t *testing.T) {
	t.Setenv("DS2API_CONFIG_JSON", `{"keys":["k1"],"history_split":{"enabled":false,"trigger_after_turns":2}}`)
	store := LoadStore()
	if store.HistorySplitEnabled() {
		t.Fatal("expected history split disabled when config disables it")
	}
	snap := store.Snapshot()
	if snap.HistorySplit.Enabled == nil || *snap.HistorySplit.Enabled {
		t.Fatalf("expected history_split.enabled=false, got %#v", snap.HistorySplit.Enabled)
	}
	if got := store.HistorySplitTriggerAfterTurns(); got != 2 {
		t.Fatalf("history split trigger_after_turns=%d want=2", got)
	}
}

func TestStoreCurrentInputFileAccessors(t *testing.T) {
	store := &Store{cfg: Config{}}
	if !store.CurrentInputFileEnabled() {
		t.Fatal("expected current input file enabled by default")
	}
	if got := store.CurrentInputFileMinChars(); got != 0 {
		t.Fatalf("default current input file min_chars=%d want=0", got)
	}

	enabled := false
	store.cfg.CurrentInputFile = CurrentInputFileConfig{Enabled: &enabled, MinChars: 12345}
	if store.CurrentInputFileEnabled() {
		t.Fatal("expected current input file disabled")
	}

	enabled = true
	store.cfg.CurrentInputFile.Enabled = &enabled
	if !store.CurrentInputFileEnabled() {
		t.Fatal("expected current input file enabled")
	}
	if got := store.CurrentInputFileMinChars(); got != 12345 {
		t.Fatalf("current input file min_chars=%d want=12345", got)
	}

	historyEnabled := true
	store.cfg.HistorySplit.Enabled = &historyEnabled
	if store.CurrentInputFileEnabled() {
		t.Fatal("expected history split to suppress current input file mode")
	}
}

func TestStoreThinkingInjectionAccessors(t *testing.T) {
	store := &Store{cfg: Config{}}
	if !store.ThinkingInjectionEnabled() {
		t.Fatal("expected thinking injection enabled by default")
	}

	disabled := false
	store.cfg.ThinkingInjection.Enabled = &disabled
	if store.ThinkingInjectionEnabled() {
		t.Fatal("expected thinking injection disabled by explicit config")
	}

	store.cfg.ThinkingInjection.Prompt = "  custom thinking prompt  "
	if got := store.ThinkingInjectionPrompt(); got != "custom thinking prompt" {
		t.Fatalf("thinking injection prompt=%q want custom thinking prompt", got)
	}
}
