'use strict';

const DEEPSEEK_COMPLETION_URL = 'https://chat.deepseek.com/api/v0/chat/completion';
const DEEPSEEK_CONTINUE_URL = 'https://chat.deepseek.com/api/v0/chat/continue';
const EMPTY_OUTPUT_RETRY_SUFFIX = 'Previous reply had no visible output. Please regenerate the visible final answer or tool call now.';
const EMPTY_OUTPUT_RETRY_MAX_ATTEMPTS = 1;
const AUTO_CONTINUE_MAX_ROUNDS = 8;
const MIN_CONTINUATION_SNAPSHOT_LEN = 32;
const MIN_DELTA_FLUSH_CHARS = 16;
const MAX_DELTA_FLUSH_WAIT_MS = 20;

const BASE_HEADERS = Object.freeze({
  'Host': 'chat.deepseek.com',
  'Accept': 'application/json',
  'Content-Type': 'application/json',
  'accept-charset': 'UTF-8',
});

const SKIP_PATTERNS = Object.freeze([
  'quasi_status', 'elapsed_secs', 'token_usage', 'pending_fragment',
  'conversation_mode', 'fragments/-1/status', 'fragments/-2/status', 'fragments/-3/status',
]);

const SKIP_EXACT_PATHS = new Set(['response/search_status']);

const DEFAULT_CORS_ALLOW_HEADERS = [
  'Content-Type', 'Authorization', 'X-API-Key', 'X-Ds2-Target-Account',
  'X-Ds2-Source', 'X-Vercel-Protection-Bypass', 'X-Goog-Api-Key',
  'Anthropic-Version', 'Anthropic-Beta',
];

function writeOpenAIError(res, status, message) {
  res.status = status;
  res.headers.set('Content-Type', 'application/json');
  res.write(JSON.stringify({
    error: {
      message,
      type: openAIErrorType(status),
    },
  }));
  return res;
}

function openAIErrorType(status) {
  switch (status) {
    case 400: return 'invalid_request_error';
    case 401: return 'authentication_error';
    case 403: return 'permission_error';
    case 429: return 'rate_limit_error';
    case 503: return 'service_unavailable_error';
    default: return status >= 500 ? 'api_error' : 'invalid_request_error';
  }
}

function setCorsHeaders(res, req) {
  const origin = req.headers.get('origin') || '*';
  res.headers.set('Access-Control-Allow-Origin', origin);
  res.headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE');
  res.headers.set('Access-Control-Max-Age', '600');
  res.headers.set('Vary', 'Origin');
  const allowHeaders = [...DEFAULT_CORS_ALLOW_HEADERS];
  const reqHeaders = req.headers.get('access-control-request-headers');
  if (reqHeaders) {
    for (const h of reqHeaders.split(',')) {
      const trimmed = h.trim().toLowerCase();
      if (trimmed && !allowHeaders.map(x => x.toLowerCase()).includes(trimmed)) {
        allowHeaders.push(h.trim());
      }
    }
  }
  res.headers.set('Access-Control-Allow-Headers', allowHeaders.join(', '));
}

async function readRequestBody(request) {
  const contentLength = request.headers.get('content-length');
  if (!contentLength) {
    const clone = request.clone();
    const text = await clone.text();
    return new TextEncoder().encode(text);
  }
  const buffer = await request.arrayBuffer();
  return new Uint8Array(buffer);
}

function asString(v) {
  if (typeof v === 'string') return v.trim();
  if (Array.isArray(v)) return asString(v[0]);
  if (v == null) return '';
  return String(v).trim();
}

function isAbortError(err) {
  if (!err || typeof err !== 'object') return false;
  return err.name === 'AbortError' || err.code === 'ABORT_ERR';
}

function numberValue(v) {
  if (typeof v === 'number' && Number.isFinite(v)) return Math.trunc(v);
  const parsed = parseInt(asString(v), 10);
  return Number.isFinite(parsed) ? parsed : 0;
}

function trimContinuationOverlap(existing, incoming) {
  if (!incoming) return '';
  if (!existing) return incoming;
  if (incoming.length >= MIN_CONTINUATION_SNAPSHOT_LEN && incoming.startsWith(existing)) {
    return incoming.slice(existing.length);
  }
  if (incoming.length >= MIN_CONTINUATION_SNAPSHOT_LEN && existing.startsWith(incoming)) {
    return '';
  }
  return incoming;
}

function estimateTokens(text) {
  const t = asString(text);
  if (!t) return 0;
  let asciiChars = 0, nonASCIIChars = 0;
  for (const ch of Array.from(t)) {
    if (ch.charCodeAt(0) < 128) asciiChars++;
    else nonASCIIChars++;
  }
  const n = Math.floor(asciiChars / 4) + Math.floor((nonASCIIChars * 10 + 7) / 13);
  return n < 1 ? 1 : n;
}

function buildUsage(prompt, thinking, output) {
  const reasoningTokens = estimateTokens(thinking);
  const completionTokens = estimateTokens(output);
  const promptTokens = estimateTokens(prompt);
  return {
    prompt_tokens: promptTokens,
    completion_tokens: reasoningTokens + completionTokens,
    total_tokens: promptTokens + reasoningTokens + completionTokens,
    completion_tokens_details: { reasoning_tokens: reasoningTokens },
  };
}

function newCallID() {
  const chars = '0123456789abcdef';
  let id = '';
  const randomValues = new Uint8Array(16);
  crypto.getRandomValues(randomValues);
  for (let i = 0; i < 16; i++) {
    id += chars[randomValues[i] % 16];
  }
  return id;
}

function ensureStreamToolCallID(idStore, index) {
  const key = Number.isInteger(index) ? index : 0;
  const existing = idStore.get(key);
  if (existing) return existing;
  const next = `call_${newCallID()}`;
  idStore.set(key, next);
  return next;
}

function stripReferenceMarkersText(text) {
  if (!text) return text;
  return text.replace(/\[(?:citation|reference):\s*\d+\]/gi, '');
}

function stripThinkTags(text) {
  if (typeof text !== 'string' || !text) return text;
  return text.replace(/<\/?\s*think\s*>/gi, '');
}

function shouldSkipPath(pathValue) {
  if (pathValue && /^response\/fragments\/-?\d+\/status$/i.test(pathValue)) return true;
  if (SKIP_EXACT_PATHS.has(pathValue)) return true;
  for (const p of SKIP_PATTERNS) {
    if (pathValue && pathValue.includes(p)) return true;
  }
  return false;
}

function isStatusPath(pathValue) {
  return pathValue === 'response/status' || pathValue === 'status';
}

function isFinishedStatus(value) {
  return asString(value).toUpperCase() === 'FINISHED';
}

function hasContentFilterStatus(chunk) {
  if (!chunk || typeof chunk !== 'object') return false;
  const code = asString(chunk.code);
  if (code && code.toLowerCase() === 'content_filter') return true;
  return hasContentFilterStatusValue(chunk);
}

function hasContentFilterStatusValue(v) {
  if (Array.isArray(v)) {
    for (const item of v) {
      if (hasContentFilterStatusValue(item)) return true;
    }
    return false;
  }
  if (!v || typeof v !== 'object') return false;
  const pathValue = asString(v.p);
  if (pathValue && pathValue.toLowerCase().includes('status')) {
    if (asString(v.v).toLowerCase() === 'content_filter') return true;
  }
  if (asString(v.code).toLowerCase() === 'content_filter') return true;
  for (const value of Object.values(v)) {
    if (hasContentFilterStatusValue(value)) return true;
  }
  return false;
}

function isCitation(text) {
  return asString(text).trim().startsWith('[citation:');
}

function asContentString(v, stripReferenceMarkers = true) {
  if (typeof v === 'string') {
    return stripReferenceMarkers ? stripReferenceMarkersText(v) : v;
  }
  if (Array.isArray(v)) {
    let out = '';
    for (const item of v) {
      out += asContentString(item, stripReferenceMarkers);
    }
    return out;
  }
  if (v && typeof v === 'object') {
    if (Object.prototype.hasOwnProperty.call(v, 'content')) {
      return asContentString(v.content, stripReferenceMarkers);
    }
    if (Object.prototype.hasOwnProperty.call(v, 'v')) {
      return asContentString(v.v, stripReferenceMarkers);
    }
    if (Object.prototype.hasOwnProperty.call(v, 'text')) {
      return asContentString(v.text, stripReferenceMarkers);
    }
    if (Object.prototype.hasOwnProperty.call(v, 'value')) {
      return asContentString(v.value, stripReferenceMarkers);
    }
    return '';
  }
  if (v == null) return '';
  return stripReferenceMarkers ? stripReferenceMarkersText(String(v)) : String(v);
}

function filterLeakedContentFilterParts(parts) {
  if (!Array.isArray(parts) || parts.length === 0) return parts;
  const out = [];
  for (const p of parts) {
    if (!p || typeof p !== 'object') continue;
    const { text, stripped } = stripLeakedContentFilterSuffix(p.text);
    if (stripped && shouldDropCleanedLeakedChunk(text)) continue;
    if (stripped) {
      out.push({ ...p, text });
      continue;
    }
    out.push(p);
  }
  return out;
}

function stripLeakedContentFilterSuffix(text) {
  if (typeof text !== 'string' || text === '') return { text, stripped: false };
  const upperText = text.toUpperCase();
  const idx = upperText.indexOf('CONTENT_FILTER');
  if (idx < 0) return { text, stripped: false };
  return {
    text: text.slice(0, idx).replace(/[ \t\r]+$/g, ''),
    stripped: true,
  };
}

function shouldDropCleanedLeakedChunk(cleaned) {
  if (cleaned === '') return true;
  if (typeof cleaned === 'string' && cleaned.includes('\n')) return false;
  return asString(cleaned).trim() === '';
}

function splitThinkingParts(parts) {
  const out = [];
  let thinkingDone = false;
  for (const p of parts) {
    if (!p) continue;
    if (thinkingDone && p.type === 'thinking') {
      const cleaned = stripThinkTags(p.text);
      if (cleaned) out.push({ text: cleaned, type: 'text' });
      continue;
    }
    if (p.type !== 'thinking') {
      const cleaned = stripThinkTags(p.text);
      if (cleaned) out.push({ text: cleaned, type: p.type });
      continue;
    }
    const match = /<\/\s*think\s*>/i.exec(p.text);
    if (!match) {
      out.push(p);
      continue;
    }
    thinkingDone = true;
    const before = p.text.substring(0, match.index);
    let after = p.text.substring(match.index + match[0].length);
    if (before) out.push({ text: before, type: 'thinking' });
    after = stripThinkTags(after);
    if (after) out.push({ text: after, type: 'text' });
  }
  return { parts: out, transitioned: thinkingDone };
}

function finalizeThinkingParts(parts, thinkingEnabled, newType) {
  const splitResult = splitThinkingParts(parts);
  let finalType = newType;
  let finalParts = splitResult.parts;
  if (splitResult.transitioned) finalType = 'text';
  if (!thinkingEnabled) finalParts = finalParts.filter(p => p && p.type !== 'thinking');
  return { parts: finalParts, newType: finalType };
}

function parseChunkForContent(chunk, thinkingEnabled, currentType, stripReferenceMarkers = true) {
  if (!chunk || typeof chunk !== 'object') {
    return { parsed: false, parts: [], finished: false, contentFilter: false, errorMessage: '', newType: currentType };
  }

  if (Object.prototype.hasOwnProperty.call(chunk, 'error')) {
    return {
      parsed: true, parts: [], finished: true, contentFilter: false,
      errorMessage: formatErrorMessage(chunk.error), newType: currentType,
    };
  }

  const pathValue = asString(chunk.p);

  if (hasContentFilterStatus(chunk)) {
    return { parsed: true, parts: [], finished: true, contentFilter: true, errorMessage: '', newType: currentType };
  }

  if (shouldSkipPath(pathValue)) {
    return { parsed: true, parts: [], finished: false, contentFilter: false, errorMessage: '', newType: currentType };
  }

  if (isStatusPath(pathValue)) {
    if (isFinishedStatus(chunk.v)) {
      return { parsed: true, parts: [], finished: true, contentFilter: false, errorMessage: '', newType: currentType };
    }
    return { parsed: true, parts: [], finished: false, contentFilter: false, errorMessage: '', newType: currentType };
  }

  if (!Object.prototype.hasOwnProperty.call(chunk, 'v')) {
    return { parsed: true, parts: [], finished: false, contentFilter: false, errorMessage: '', newType: currentType };
  }

  let newType = currentType;
  const parts = [];

  if (pathValue === 'response/fragments' && asString(chunk.o).toUpperCase() === 'APPEND' && Array.isArray(chunk.v)) {
    for (const frag of chunk.v) {
      if (!frag || typeof frag !== 'object') continue;
      const fragType = asString(frag.type).toUpperCase();
      const content = asContentString(frag.content, stripReferenceMarkers);
      if (!content) continue;
      if (fragType === 'THINK' || fragType === 'THINKING') {
        newType = 'thinking';
        parts.push({ text: content, type: 'thinking' });
      } else if (fragType === 'RESPONSE') {
        newType = 'text';
        parts.push({ text: content, type: 'text' });
      } else {
        parts.push({ text: content, type: 'text' });
      }
    }
  }

  if (pathValue === 'response' && Array.isArray(chunk.v)) {
    for (const item of chunk.v) {
      if (!item || typeof item !== 'object') continue;
      if (item.p === 'fragments' && item.o === 'APPEND' && Array.isArray(item.v)) {
        for (const frag of item.v) {
          const fragType = asString(frag && frag.type).toUpperCase();
          if (fragType === 'THINK' || fragType === 'THINKING') newType = 'thinking';
          else if (fragType === 'RESPONSE') newType = 'text';
        }
      }
    }
  }

  if (pathValue === 'response/content') newType = 'text';
  else if (pathValue === 'response/thinking_content' && (!thinkingEnabled || newType !== 'text')) newType = 'thinking';

  let partType = 'text';
  if (pathValue === 'response/thinking_content') {
    if (!thinkingEnabled) partType = 'thinking';
    else if (newType === 'text') partType = 'text';
    else partType = 'thinking';
  } else if (pathValue === 'response/content') partType = 'text';
  else if (pathValue.includes('response/fragments') && pathValue.includes('/content')) partType = newType;
  else if (!pathValue) partType = newType || 'text';

  const val = chunk.v;
  if (typeof val === 'string') {
    if (isFinishedStatus(val) && (!pathValue || pathValue === 'status')) {
      return { parsed: true, parts: [], finished: true, contentFilter: false, errorMessage: '', newType };
    }
    if (isStatusPath(pathValue)) {
      return { parsed: true, parts: [], finished: false, contentFilter: false, errorMessage: '', newType };
    }
    const content = asContentString(val, stripReferenceMarkers);
    if (content) parts.push({ text: content, type: partType });
    let resolvedParts = filterLeakedContentFilterParts(parts);
    const finalized = finalizeThinkingParts(resolvedParts, thinkingEnabled, newType);
    return { parsed: true, parts: finalized.parts, finished: false, contentFilter: false, errorMessage: '', newType: finalized.newType };
  }

  if (Array.isArray(val)) {
    const extracted = extractContentRecursive(val, partType, stripReferenceMarkers);
    if (extracted.finished) {
      return { parsed: true, parts: [], finished: true, contentFilter: false, errorMessage: '', newType };
    }
    parts.push(...extracted.parts);
    let resolvedParts = filterLeakedContentFilterParts(parts);
    const finalized = finalizeThinkingParts(resolvedParts, thinkingEnabled, newType);
    return { parsed: true, parts: finalized.parts, finished: false, contentFilter: false, errorMessage: '', newType: finalized.newType };
  }

  if (val && typeof val === 'object') {
    const directContent = asContentString(val, stripReferenceMarkers);
    if (directContent) parts.push({ text: directContent, type: partType });
    const resp = val.response && typeof val.response === 'object' ? val.response : val;
    if (Array.isArray(resp.fragments)) {
      for (const frag of resp.fragments) {
        if (!frag || typeof frag !== 'object') continue;
        const content = asContentString(frag.content, stripReferenceMarkers);
        if (!content) continue;
        const t = asString(frag.type).toUpperCase();
        if (t === 'THINK' || t === 'THINKING') {
          newType = 'thinking';
          parts.push({ text: content, type: 'thinking' });
        } else if (t === 'RESPONSE') {
          newType = 'text';
          parts.push({ text: content, type: 'text' });
        } else {
          parts.push({ text: content, type: partType });
        }
      }
    }
  }

  let resolvedParts = filterLeakedContentFilterParts(parts);
  const finalized = finalizeThinkingParts(resolvedParts, thinkingEnabled, newType);
  return { parsed: true, parts: finalized.parts, finished: false, contentFilter: false, errorMessage: '', newType: finalized.newType };
}

function extractContentRecursive(items, defaultType, stripReferenceMarkers = true) {
  const parts = [];
  for (const it of items) {
    if (!it || typeof it !== 'object') continue;
    if (!Object.prototype.hasOwnProperty.call(it, 'v')) continue;
    const itemPath = asString(it.p);
    const itemV = it.v;
    if (isStatusPath(itemPath)) {
      if (isFinishedStatus(itemV)) return { parts: [], finished: true };
      continue;
    }
    if (shouldSkipPath(itemPath)) continue;
    const content = asContentString(it.content, stripReferenceMarkers);
    if (content) {
      const typeName = asString(it.type).toUpperCase();
      if (typeName === 'THINK' || typeName === 'THINKING') parts.push({ text: content, type: 'thinking' });
      else if (typeName === 'RESPONSE') parts.push({ text: content, type: 'text' });
      else parts.push({ text: content, type: defaultType });
      continue;
    }
    let partType = defaultType;
    if (itemPath.includes('thinking')) partType = 'thinking';
    else if (itemPath.includes('content') || itemPath === 'response' || itemPath === 'fragments') partType = 'text';
    if (typeof itemV === 'string') {
      if (isStatusPath(itemPath)) continue;
      if (itemV && itemV !== 'FINISHED') {
        const c = asContentString(itemV, stripReferenceMarkers);
        if (c) parts.push({ text: c, type: partType });
      }
      continue;
    }
    if (!Array.isArray(itemV)) continue;
    for (const inner of itemV) {
      if (typeof inner === 'string') {
        if (inner) {
          const c = asContentString(inner, stripReferenceMarkers);
          if (c) parts.push({ text: c, type: partType });
        }
        continue;
      }
      if (!inner || typeof inner !== 'object') continue;
      const ct = asContentString(inner.content, stripReferenceMarkers);
      if (!ct) continue;
      const typeName = asString(inner.type).toUpperCase();
      if (typeName === 'THINK' || typeName === 'THINKING') parts.push({ text: ct, type: 'thinking' });
      else if (typeName === 'RESPONSE') parts.push({ text: ct, type: 'text' });
      else parts.push({ text: ct, type: partType });
    }
  }
  return { parts, finished: false };
}

function formatErrorMessage(v) {
  if (typeof v === 'string') return v;
  if (v == null) return String(v);
  try { return JSON.stringify(v); } catch { return String(v); }
}

function extractToolNames(tools) {
  if (!Array.isArray(tools) || tools.length === 0) return [];
  const out = [];
  const seen = new Set();
  for (const t of tools) {
    if (!t || typeof t !== 'object') continue;
    const fn = t.function && typeof t.function === 'object' ? t.function : t;
    const name = asString(fn.name);
    if (!name || seen.has(name)) continue;
    seen.add(name);
    out.push(name);
  }
  return out;
}

function resolveToolcallPolicy(tools) {
  const toolNames = extractToolNames(tools);
  if (toolNames.length === 0 && Array.isArray(tools) && tools.length > 0) {
    return { toolNames: ['__any_tool__'], toolSieveEnabled: false, emitEarlyToolDeltas: false };
  }
  return {
    toolNames,
    toolSieveEnabled: toolNames.length > 0,
    emitEarlyToolDeltas: true,
  };
}

function formatIncrementalToolCallDeltas(deltas, idStore) {
  if (!Array.isArray(deltas) || deltas.length === 0) return [];
  const out = [];
  for (const d of deltas) {
    if (!d || typeof d !== 'object') continue;
    const index = Number.isInteger(d.index) ? d.index : 0;
    const id = ensureStreamToolCallID(idStore, index);
    const item = { index, id, type: 'function' };
    const fn = {};
    if (asString(d.name)) fn.name = asString(d.name);
    if (typeof d.arguments === 'string' && d.arguments !== '') fn.arguments = d.arguments;
    if (Object.keys(fn).length === 0) continue;
    item.function = fn;
    out.push(item);
  }
  return out;
}

function resetStreamToolCallState(idStore, seenNames) {
  if (idStore instanceof Map) idStore.clear();
  if (seenNames instanceof Map) seenNames.clear();
}

function createToolSieveState() {
  return {
    pending: '',
    capture: '',
    capturing: false,
    codeFenceStack: [],
    pendingToolRaw: '',
    pendingToolCalls: [],
    disableDeltas: false,
    toolNameSent: false,
    toolName: '',
    toolArgsStart: -1,
    toolArgsSent: -1,
    toolArgsString: false,
    toolArgsDone: false,
  };
}

function processToolSieveChunk(state, chunk, toolNames) {
  if (!state) return [];
  if (chunk) state.pending += chunk;
  const events = [];

  while (true) {
    if (Array.isArray(state.pendingToolCalls) && state.pendingToolCalls.length > 0) {
      events.push({ type: 'tool_calls', calls: state.pendingToolCalls });
      state.pendingToolRaw = '';
      state.pendingToolCalls = [];
      continue;
    }

    if (state.capturing) {
      if (state.pending) {
        state.capture += state.pending;
        state.pending = '';
      }
      const consumed = consumeToolCapture(state, toolNames);
      if (!consumed.ready) break;
      const captured = state.capture;
      state.capture = '';
      state.capturing = false;
      resetIncrementalToolState(state);

      if (Array.isArray(consumed.calls) && consumed.calls.length > 0) {
        if (consumed.prefix) events.push({ type: 'text', text: consumed.prefix });
        state.pendingToolRaw = captured;
        state.pendingToolCalls = consumed.calls;
        if (consumed.suffix) state.pending = consumed.suffix + state.pending;
        continue;
      }
      if (consumed.prefix) events.push({ type: 'text', text: consumed.prefix });
      if (consumed.suffix) state.pending += consumed.suffix;
      continue;
    }

    const pending = state.pending || '';
    if (!pending) break;

    const start = findToolSegmentStart(state, pending);
    if (start >= 0) {
      const prefix = pending.slice(0, start);
      if (prefix) events.push({ type: 'text', text: prefix });
      state.pending = '';
      state.capture += pending.slice(start);
      state.capturing = true;
      resetIncrementalToolState(state);
      continue;
    }

    const [safe, hold] = splitSafeContentForToolDetection(state, pending);
    if (!safe) break;
    state.pending = hold;
    events.push({ type: 'text', text: safe });
  }

  return events;
}

function flushToolSieve(state, toolNames) {
  if (!state) return [];
  const events = processToolSieveChunk(state, '', toolNames);
  if (Array.isArray(state.pendingToolCalls) && state.pendingToolCalls.length > 0) {
    events.push({ type: 'tool_calls', calls: state.pendingToolCalls });
    state.pendingToolRaw = '';
    state.pendingToolCalls = [];
  }
  if (state.capturing) {
    const consumed = consumeToolCapture(state, toolNames);
    if (consumed.ready) {
      if (consumed.prefix) events.push({ type: 'text', text: consumed.prefix });
      if (Array.isArray(consumed.calls) && consumed.calls.length > 0) {
        events.push({ type: 'tool_calls', calls: consumed.calls });
      }
      if (consumed.suffix) events.push({ type: 'text', text: consumed.suffix });
    } else if (state.capture) {
      events.push({ type: 'text', text: state.capture });
    }
    state.capture = '';
    state.capturing = false;
    resetIncrementalToolState(state);
  }
  if (state.pending) {
    events.push({ type: 'text', text: state.pending });
    state.pending = '';
  }
  return events;
}

function resetIncrementalToolState(state) {
  state.disableDeltas = false;
  state.toolNameSent = false;
  state.toolName = '';
  state.toolArgsStart = -1;
  state.toolArgsSent = -1;
  state.toolArgsString = false;
  state.toolArgsDone = false;
}

function insideCodeFence(text) {
  const t = typeof text === 'string' ? text : '';
  if (!t) return false;
  const ticks = (t.match(/```/g) || []).length;
  const tildes = (t.match(/  ~/g) || []).length;
  return (ticks % 2 !== 0) || (tildes % 2 !== 0);
}

function insideCodeFenceWithState(state, text) {
  if (!state) return insideCodeFence(text);
  const ticks = (text.match(/```/g) || []).length;
  const tildes = (text.match(/  ~/g) || []).length;
  return ticks > 0 || tildes > 0;
}

function splitSafeContentForToolDetection(state, s) {
  const text = s || '';
  if (!text) return ['', ''];
  const xmlIdx = findPartialXMLToolTagStart(text);
  if (xmlIdx >= 0) {
    if (insideCodeFenceWithState(state, text.slice(0, xmlIdx))) return [text, ''];
    if (xmlIdx > 0) return [text.slice(0, xmlIdx), text.slice(xmlIdx)];
    return ['', text];
  }
  return [text, ''];
}

function findToolSegmentStart(state, s) {
  if (!s) return -1;
  const patterns = ['<tool_calls', '<invoke', '<tool-call', '<tool_call'];
  for (const p of patterns) {
    const idx = s.toLowerCase().indexOf(p);
    if (idx >= 0 && !insideCodeFenceWithState(state, s.slice(0, idx))) {
      return idx;
    }
  }
  return -1;
}

function findPartialXMLToolTagStart(text) {
  if (!text) return -1;
  const patterns = ['<tool_calls', '<invoke', '<tool_calls', '<tool_call', '</tool'];
  let minIdx = -1;
  for (const p of patterns) {
    const idx = text.toLowerCase().indexOf(p);
    if (idx >= 0 && (minIdx < 0 || idx < minIdx)) minIdx = idx;
  }
  if (minIdx < 0) return -1;
  const afterTag = text.slice(minIdx);
  if (afterTag.includes('>')) return -1;
  return minIdx;
}

function consumeToolCapture(state, toolNames) {
  const captured = state.capture || '';
  if (!captured) return { ready: false, prefix: '', calls: [], suffix: '' };

  const xmlResult = consumeXMLToolCapture(captured, toolNames);
  if (xmlResult.ready) return xmlResult;

  if (hasOpenXMLToolTag(captured)) {
    return { ready: false, prefix: '', calls: [], suffix: '' };
  }

  return { ready: true, prefix: captured, calls: [], suffix: '' };
}

function hasOpenXMLToolTag(text) {
  const patterns = ['<tool_calls', '<invoke', '<tool_call'];
  for (const p of patterns) {
    if (text.toLowerCase().includes(p)) return true;
  }
  return false;
}

function consumeXMLToolCapture(text, toolNames) {
  const toolCallPattern = /<tool_calls[^>]*>([\s\S]*?)<\/tool_calls>/gi;
  const invokePattern = /<invoke\s+name=["']?([^"'\s>]+)["']?\s*>([\s\S]*?)<\/invoke>/gi;
  const paramPattern = /<parameter\s+name=["']?([^"'\s>]+)["']?\s*>([\s\S]*?)<\/parameter>/gi;
  const jsonBlockPattern = /<thinking_content[\s\S]*?>([\s\S]*?)<\/thinking_content>/gi;

  let calls = [];
  let match;
  const toolCallBlocks = [];

  while ((match = toolCallPattern.exec(text)) !== null) {
    toolCallBlocks.push({ start: match.index, end: match.index + match[0].length, content: match[1] });
  }

  while ((match = invokePattern.exec(text)) !== null) {
    const name = match[1];
    const inner = match[2];
    const params = {};
    let paramMatch;
    const paramRegex = /<parameter\s+name=["']?([^"'\s>]+)["']?\s*>([\s\S]*?)<\/parameter>/gi;
    while ((paramMatch = paramRegex.exec(inner)) !== null) {
      const pName = paramMatch[1];
      let pValue = paramMatch[2].trim();
      pValue = pValue.replace(/<!\[CDATA\[([\s\S]*?)\]\]>/gi, '$1');
      try {
        const parsed = JSON.parse(pValue);
        params[pName] = parsed;
      } catch {
        params[pName] = pValue;
      }
    }
    toolCallBlocks.push({ start: match.index, end: match.index + match[0].length });
    calls.push({ name, input: params });
  }

  for (const block of toolCallBlocks) {
    text = text.slice(0, block.start) + text.slice(block.end);
  }

  return { ready: true, prefix: text, calls, suffix: '' };
}

function formatOpenAIStreamToolCalls(calls, idStore) {
  if (!Array.isArray(calls) || calls.length === 0) return [];
  return calls.map((c, idx) => ({
    index: idx,
    id: ensureStreamToolCallID(idStore, idx),
    type: 'function',
    function: {
      name: c.name,
      arguments: JSON.stringify(c.input || {}),
    },
  }));
}

function createChatCompletionEmitter({ stream, sessionID, created, model, isClosed }) {
  let firstChunkSent = false;
  let controller;

  const sendFrame = (obj) => {
    if (isClosed() || stream.destroyed) return;
    stream.write(`data: ${JSON.stringify(obj)}\n\n`);
  };

  const sendDeltaFrame = (delta) => {
    const payloadDelta = { ...delta };
    if (!firstChunkSent) {
      payloadDelta.role = 'assistant';
      firstChunkSent = true;
    }
    sendFrame({
      id: sessionID,
      object: 'chat.completion.chunk',
      created,
      model,
      choices: [{ delta: payloadDelta, index: 0 }],
    });
  };

  return { sendFrame, sendDeltaFrame };
}

function createDeltaCoalescer({ sendDeltaFrame }) {
  let pendingField = '';
  let pendingText = '';
  let flushTimer = null;

  const clearFlushTimer = () => {
    if (flushTimer) {
      clearTimeout(flushTimer);
      flushTimer = null;
    }
  };

  const flush = () => {
    clearFlushTimer();
    if (!pendingField || !pendingText) return;
    const delta = { [pendingField]: pendingText };
    pendingField = '';
    pendingText = '';
    sendDeltaFrame(delta);
  };

  const scheduleFlush = () => {
    if (flushTimer || MAX_DELTA_FLUSH_WAIT_MS <= 0) return;
    flushTimer = setTimeout(flush, MAX_DELTA_FLUSH_WAIT_MS);
  };

  const append = (field, text) => {
    if (!field || !text) return;
    if (pendingField && pendingField !== field) flush();
    pendingField = field;
    pendingText += text;
    if ([...pendingText].length >= MIN_DELTA_FLUSH_CHARS) {
      flush();
      return;
    }
    scheduleFlush();
  };

  return { append, flush };
}

function createContinueState(sessionID) {
  return {
    sessionID: asString(sessionID),
    responseMessageID: 0,
    lastStatus: '',
    finished: false,
  };
}

function observeContinueState(state, chunk) {
  if (!state || !chunk || typeof chunk !== 'object') return;
  const topID = numberValue(chunk.response_message_id);
  if (topID > 0) state.responseMessageID = topID;
  observeContinueResponseObject(state, chunk.v);
  const response = chunk.v && typeof chunk.v === 'object' ? chunk.v.response : null;
  observeContinueResponseObject(state, response);
}

function observeContinueResponseObject(state, response) {
  if (!state || !response || typeof response !== 'object') return;
  const id = numberValue(response.message_id);
  if (id > 0) state.responseMessageID = id;
  setContinueStatus(state, asString(response.status));
  if (response.auto_continue === true) state.lastStatus = 'AUTO_CONTINUE';
}

function setContinueStatus(state, status) {
  const normalized = asString(status).trim();
  if (!normalized) return;
  state.lastStatus = normalized;
  if (['FINISHED', 'CONTENT_FILTER'].includes(normalized.toUpperCase())) state.finished = true;
}

function shouldAutoContinue(state) {
  if (!state || state.finished || !state.sessionID || state.responseMessageID <= 0) return false;
  return ['INCOMPLETE', 'AUTO_CONTINUE'].includes(asString(state.lastStatus).trim().toUpperCase());
}

function upstreamEmptyOutputDetail(contentFilter, _text, thinking) {
  if (contentFilter) return { status: 400, message: 'Upstream content filtered the response and returned no output.', code: 'content_filter' };
  if (thinking !== '') return { status: 429, message: 'Upstream account hit a rate limit and returned reasoning without visible output.', code: 'upstream_empty_output' };
  return { status: 429, message: 'Upstream account hit a rate limit and returned empty output.', code: 'upstream_empty_output' };
}

function sendFailedChunk(stream, status, message, code) {
  if (stream.destroyed) return;
  stream.write(`data: ${JSON.stringify({
    status_code: status,
    error: { message, type: openAIErrorType(status), code, param: null },
  })}\n\n`);
  if (!stream.destroyed) stream.write('data: [DONE]\n\n');
}

async function handleStreamRequest(request, env) {
  const rawBody = await readRequestBody(request);
  let payload;
  try {
    payload = JSON.parse(new TextDecoder().decode(rawBody));
  } catch {
    return new Response(JSON.stringify({ error: { message: 'invalid json', type: 'invalid_request_error' } }), {
      status: 400, headers: { 'Content-Type': 'application/json' }
    });
  }

  const deepseekToken = env.DS2API_DEEPSEEK_TOKEN || payload.api_key || '';
  if (!deepseekToken) {
    return new Response(JSON.stringify({ error: { message: 'DeepSeek token required', type: 'authentication_error' } }), {
      status: 401, headers: { 'Content-Type': 'application/json' }
    });
  }

  const model = payload.model || 'deepseek-chat';
  const sessionID = `chatcmpl-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  const created = Math.floor(Date.now() / 1000);
  const thinkingEnabled = payload.thinking_enabled !== false;
  const searchEnabled = payload.search_enabled === true;
  const toolPolicy = resolveToolcallPolicy(payload.tools);
  const toolNames = toolPolicy.toolNames;
  const stripReferenceMarkers = true;

  const preparedPayload = buildDeepSeekPayload(payload);
  if (!preparedPayload) {
    return new Response(JSON.stringify({ error: { message: 'Failed to build request payload', type: 'invalid_request_error' } }), {
      status: 400, headers: { 'Content-Type': 'application/json' }
    });
  }

  let powHeader = '';
  try {
    const powRes = await fetch(DEEPSEEK_COMPLETION_URL, {
      method: 'POST',
      headers: { ...BASE_HEADERS, 'Authorization': `Bearer ${deepseekToken}`, 'x-ds-pow-request': '1' },
      body: JSON.stringify(preparedPayload),
    });
    if (powRes.ok) {
      const powData = await powRes.json();
      powHeader = powData.pow_header || powData.x_dspow_response || '';
    }
  } catch {}

  if (!powHeader) {
    powHeader = await generateSimplePoW();
  }

  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const encoder = new TextEncoder();
  let closed = false;

  const markClosed = () => { if (!closed) { closed = true; } };

  const { sendFrame, sendDeltaFrame } = createChatCompletionEmitter({
    stream: { write: (data) => { if (!closed) writer.write(encoder.encode(data)); }, get destroyed() { return closed; } },
    sessionID,
    created,
    model,
    isClosed: () => closed,
  });

  const deltaCoalescer = createDeltaCoalescer({ sendDeltaFrame });

  let currentType = thinkingEnabled ? 'thinking' : 'text';
  let thinkingText = '';
  let outputText = '';
  const toolSieveEnabled = toolPolicy.toolSieveEnabled;
  const toolSieveState = createToolSieveState();
  let toolCallsEmitted = false;
  let toolCallsDoneEmitted = false;
  const streamToolCallIDs = new Map();
  const streamToolNames = new Map();

  const finish = async (reason) => {
    if (closed) return true;
    deltaCoalescer.flush();
    if (toolSieveEnabled) {
      const tailEvents = flushToolSieve(toolSieveState, toolNames);
      for (const evt of tailEvents) {
        if (evt.type === 'tool_calls' && Array.isArray(evt.calls) && evt.calls.length > 0) {
          deltaCoalescer.flush();
          toolCallsEmitted = true;
          toolCallsDoneEmitted = true;
          sendDeltaFrame({ tool_calls: formatOpenAIStreamToolCalls(evt.calls, streamToolCallIDs) });
          resetStreamToolCallState(streamToolCallIDs, streamToolNames);
          continue;
        }
        if (evt.text) deltaCoalescer.append('content', evt.text);
      }
      deltaCoalescer.flush();
    }

    if (toolCallsEmitted || toolCallsDoneEmitted) reason = 'tool_calls';
    if (toolCallsEmitted.length === 0 && !toolCallsEmitted && outputText.trim() === '') {
      const detail = upstreamEmptyOutputDetail(false, outputText, thinkingText);
      sendFailedChunk({ write: (data) => { if (!closed) writer.write(encoder.encode(data)); }, get destroyed() { return closed; } }, detail.status, detail.message, detail.code);
      markClosed();
      writer.close();
      return true;
    }

    sendFrame({
      id: sessionID, object: 'chat.completion.chunk', created, model,
      choices: [{ delta: {}, index: 0, finish_reason: reason }],
      usage: buildUsage('', thinkingText, outputText),
    });
    if (!closed) writer.write(encoder.encode('data: [DONE]\n\n'));
    markClosed();
    writer.close();
    return true;
  };

  const processStream = async (response) => {
    if (!response.ok || !response.body) {
      const detail = response.ok ? 'Stream error' : `Upstream error: ${response.status}`;
      writeOpenAIError({ status: response.status || 500, headers: new Headers({ 'Content-Type': 'application/json' }) }, response.status || 500, detail);
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffered = '';
    let continueState = createContinueState(sessionID);
    let continueRounds = 0;

    try {
      while (true) {
        if (closed) break;
        const { value, done } = await reader.read();
        if (done) break;
        buffered += decoder.decode(value, { stream: true });
        const lines = buffered.split('\n');
        buffered = lines.pop() || '';

        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (!line.startsWith('data:')) continue;
          const dataStr = line.slice(5).trim();
          if (!dataStr || dataStr === '[DONE]') continue;
          let chunk;
          try { chunk = JSON.parse(dataStr); } catch { continue; }
          observeContinueState(continueState, chunk);
          const parsed = parseChunkForContent(chunk, thinkingEnabled, currentType, stripReferenceMarkers);
          if (!parsed.parsed) continue;
          currentType = parsed.newType;
          if (parsed.errorMessage) { await finish('content_filter'); return; }
          if (parsed.contentFilter) { await finish(outputText.trim() === '' ? 'content_filter' : 'stop'); return; }
          if (parsed.finished) { break; }

          for (const p of parsed.parts) {
            if (!p.text) continue;
            if (p.type === 'thinking') {
              if (thinkingEnabled) {
                const trimmed = trimContinuationOverlap(thinkingText, p.text);
                if (!trimmed) continue;
                thinkingText += trimmed;
                deltaCoalescer.append('reasoning_content', trimmed);
              }
            } else {
              const trimmed = trimContinuationOverlap(outputText, p.text);
              if (!trimmed) continue;
              if (searchEnabled && isCitation(trimmed)) continue;
              outputText += trimmed;
              if (!toolSieveEnabled) {
                deltaCoalescer.append('content', trimmed);
                continue;
              }
              const events = processToolSieveChunk(toolSieveState, trimmed, toolNames);
              for (const evt of events) {
                if (evt.type === 'tool_call_deltas') {
                  if (!toolPolicy.emitEarlyToolDeltas) continue;
                  const formatted = formatIncrementalToolCallDeltas(evt.deltas, streamToolCallIDs);
                  if (formatted.length > 0) {
                    toolCallsEmitted = true;
                    deltaCoalescer.flush();
                    sendDeltaFrame({ tool_calls: formatted });
                  }
                  continue;
                }
                if (evt.type === 'tool_calls') {
                  toolCallsEmitted = true;
                  toolCallsDoneEmitted = true;
                  deltaCoalescer.flush();
                  sendDeltaFrame({ tool_calls: formatOpenAIStreamToolCalls(evt.calls, streamToolCallIDs) });
                  resetStreamToolCallState(streamToolCallIDs, streamToolNames);
                  continue;
                }
                if (evt.text) deltaCoalescer.append('content', evt.text);
              }
            }
          }
        }

        if (shouldAutoContinue(continueState) && continueRounds < AUTO_CONTINUE_MAX_ROUNDS) {
          continueRounds++;
          const nextRes = await fetch(DEEPSEEK_CONTINUE_URL, {
            method: 'POST',
            headers: { ...BASE_HEADERS, 'Authorization': `Bearer ${deepseekToken}`, 'x-ds-pow-response': powHeader },
            body: JSON.stringify({ chat_session_id: continueState.sessionID, message_id: continueState.responseMessageID, fallback_to_resume: true }),
          });
          if (!nextRes.ok || !nextRes.body) break;
          continueState = { ...continueState, lastStatus: '', finished: false };
          continue;
        }
        break;
      }
    } catch (err) {
      if (!isAbortError(err)) console.error('Stream error:', err);
    }

    await finish('stop');
  };

  try {
    const response = await fetch(DEEPSEEK_COMPLETION_URL, {
      method: 'POST',
      headers: { ...BASE_HEADERS, 'Authorization': `Bearer ${deepseekToken}`, 'x-ds-pow-response': powHeader },
      body: JSON.stringify(preparedPayload),
    });

    await processStream(response);
  } catch (err) {
    console.error('Request error:', err);
    return new Response(JSON.stringify({ error: { message: 'Request failed', type: 'api_error' } }), {
      status: 500, headers: { 'Content-Type': 'application/json' }
    });
  }

  return new Response(readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  });
}

function buildDeepSeekPayload(payload) {
  if (!payload.messages || !Array.isArray(payload.messages)) return null;

  const messages = payload.messages.map(m => ({
    role: m.role,
    content: typeof m.content === 'string' ? m.content : JSON.stringify(m.content),
  }));

  const result = {
    prompt: messages.map(m => `${m.role}: ${m.content}`).join('\n'),
    temperature: payload.temperature !== undefined ? payload.temperature : 1.0,
    top_p: payload.top_p !== undefined ? payload.top_p : 1.0,
    max_tokens: payload.max_tokens || 4096,
    stop: payload.stop || null,
  };

  if (payload.tools && payload.tools.length > 0) {
    result.tools = payload.tools.map(t => ({
      type: 'function',
      function: {
        name: t.function?.name || t.name || 'unknown',
        description: t.function?.description || '',
        parameters: t.function?.parameters || { type: 'object', properties: {} },
      },
    }));
  }

  return result;
}

async function generateSimplePoW() {
  const timestamp = Date.now();
  const random = Array.from(crypto.getRandomValues(new Uint8Array(16))).map(b => b.toString(16).padStart(2, '0')).join('');
  return `ct_${timestamp}_${random}`;
}

async function handleNonStreamRequest(request, env) {
  const rawBody = await readRequestBody(request);
  let payload;
  try {
    payload = JSON.parse(new TextDecoder().decode(rawBody));
  } catch {
    return new Response(JSON.stringify({ error: { message: 'invalid json', type: 'invalid_request_error' } }), {
      status: 400, headers: { 'Content-Type': 'application/json' }
    });
  }

  const deepseekToken = env.DS2API_DEEPSEEK_TOKEN || payload.api_key || '';
  if (!deepseekToken) {
    return new Response(JSON.stringify({ error: { message: 'DeepSeek token required', type: 'authentication_error' } }), {
      status: 401, headers: { 'Content-Type': 'application/json' }
    });
  }

  const preparedPayload = buildDeepSeekPayload(payload);
  if (!preparedPayload) {
    return new Response(JSON.stringify({ error: { message: 'Failed to build request payload', type: 'invalid_request_error' } }), {
      status: 400, headers: { 'Content-Type': 'application/json' }
    });
  }

  let powHeader = '';
  try {
    const powRes = await fetch(DEEPSEEK_COMPLETION_URL, {
      method: 'POST',
      headers: { ...BASE_HEADERS, 'Authorization': `Bearer ${deepseekToken}`, 'x-ds-pow-request': '1' },
      body: JSON.stringify(preparedPayload),
    });
    if (powRes.ok) {
      const powData = await powRes.json();
      powHeader = powData.pow_header || powData.x_dspow_response || '';
    }
  } catch {}

  if (!powHeader) {
    powHeader = await generateSimplePoW();
  }

  const response = await fetch(DEEPSEEK_COMPLETION_URL, {
    method: 'POST',
    headers: { ...BASE_HEADERS, 'Authorization': `Bearer ${deepseekToken}`, 'x-ds-pow-response': powHeader },
    body: JSON.stringify(preparedPayload),
  });

  if (!response.ok) {
    const errorText = await response.text();
    return new Response(errorText, { status: response.status, headers: { 'Content-Type': 'application/json' } });
  }

  let fullContent = '';
  let thinkingContent = '';
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffered = '';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffered += decoder.decode(value, { stream: true });
      const lines = buffered.split('\n');
      buffered = lines.pop() || '';

      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line.startsWith('data:')) continue;
        const dataStr = line.slice(5).trim();
        if (!dataStr || dataStr === '[DONE]') continue;
        try {
          const chunk = JSON.parse(dataStr);
          const parsed = parseChunkForContent(chunk, false, 'text', true);
          for (const p of parsed.parts) {
            if (p.type === 'thinking') thinkingContent += p.text;
            else fullContent += p.text;
          }
        } catch {}
      }
    }
  } catch {}

  const sessionID = `chatcmpl-${Date.now()}`;
  const created = Math.floor(Date.now() / 1000);
  const model = payload.model || 'deepseek-chat';

  return new Response(JSON.stringify({
    id: sessionID,
    object: 'chat.completion',
    created,
    model,
    choices: [{
      index: 0,
      message: { role: 'assistant', content: fullContent },
      finish_reason: fullContent ? 'stop' : 'length',
    }],
    usage: buildUsage('', thinkingContent, fullContent),
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
}

async function handleModelsRequest(env) {
  const models = [
    { id: 'deepseek-chat', object: 'model', created: 1700000000, owned_by: 'deepseek', permission: [] },
    { id: 'deepseek-v4-flash', object: 'model', created: 1700000000, owned_by: 'deepseek', permission: [] },
    { id: 'deepseek-v4-pro', object: 'model', created: 1700000000, owned_by: 'deepseek', permission: [] },
    { id: 'deepseek-v4-nothinking', object: 'model', created: 1700000000, owned_by: 'deepseek', permission: [] },
  ];

  return new Response(JSON.stringify({
    object: 'list',
    data: models,
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
}

async function handler(request, env, ctx) {
  const url = new URL(request.url);
  const path = url.pathname;

  const response = new Response('', { status: 200 });
  setCorsHeaders(response, request);

  if (request.method === 'OPTIONS') {
    return new Response('', { status: 204, headers: response.headers });
  }

  if (request.method !== 'POST' && request.method !== 'GET') {
    return writeOpenAIError({ status: 405, headers: response.headers }, 405, 'method not allowed');
  }

  if (path === '/v1/chat/completions') {
    const rawBody = await readRequestBody(request);
    let payload;
    try {
      payload = JSON.parse(new TextDecoder().decode(rawBody));
    } catch {
      return writeOpenAIError({ status: 400, headers: response.headers }, 400, 'invalid json');
    }

    const isStream = payload.stream === true;

    if (isStream) {
      return handleStreamRequest(request, env);
    } else {
      return handleNonStreamRequest(request, env);
    }
  }

  if (path === '/v1/models') {
    return handleModelsRequest(env);
  }

  if (path === '/healthz') {
    return new Response(JSON.stringify({ status: 'ok' }), {
      headers: { 'Content-Type': 'application/json' },
    });
  }

  return writeOpenAIError({ status: 404, headers: response.headers }, 404, 'not found');
}

addEventListener('fetch', event => {
  event.respondWith(handler(event.request, event.env, event));
});
