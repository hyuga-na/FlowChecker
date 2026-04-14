import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.0";

// ====== Config ======
const LINE_NAMES = ["背景", "課題", "解法", "結果", "考察"];
const FLOW_LABELS = ["問題なし", "飛躍", "不足", "未定義", "過剰"];

// 実運用では、動作確認済みモデルに差し替えてください。
// small/medium/large の「大小」は UI 上の論理名として扱い、
// 実際のモデル選定はブラウザメモリ事情に応じて調整するのが安全です。
const MODEL_PROFILES = {
  small: {
    label: "GPU小: 0.5B",
    kind: "llm",
    // 0.5B 級はブラウザ実運用で候補が限られるため、まずは小さめの instruct 系に寄せる
    // 必要ならローカル配布モデルへ差し替え
    modelId: "onnx-community/Phi-3.5-mini-instruct-onnx-web",
    preferredDevice: "webgpu",
    maxNewTokens: 220,
  },
  medium: {
    label: "GPU中: 3B",
    kind: "llm",
    modelId: "microsoft/Phi-3-mini-4k-instruct-onnx-web",
    preferredDevice: "webgpu",
    maxNewTokens: 260,
  },
  large: {
    label: "GPU大: 7B",
    kind: "llm",
    // 7B はブラウザではかなり重いので、まずは medium と別設定にしておき、
    // 実際には別モデルへ差し替える設計だけ用意する
    modelId: "microsoft/Phi-3-mini-4k-instruct-onnx-web",
    preferredDevice: "webgpu",
    maxNewTokens: 320,
  },
  cpu: {
    label: "CPU分類器",
    kind: "rule",
  },
};

const APP_STATE = {
  analyzer: null,
  modelKey: "medium",
  modelLoaded: false,
  isLoading: false,
  isChecking: false,
  selectedDevice: "none",
};

let nodeCounter = 0;

// ====== Tree State ======
function createEmptyResults() {
  return [
    {
      label: "問題なし",
      reason: "1行目は比較対象がないため固定で問題なしです。",
      scoreText: "固定判定",
      relation: "比較対象なし",
      analyzed: true,
    },
    {
      label: "未判定",
      reason: "",
      scoreText: "",
      relation: "背景 → 課題",
      analyzed: false,
    },
    {
      label: "未判定",
      reason: "",
      scoreText: "",
      relation: "課題 → 解法",
      analyzed: false,
    },
    {
      label: "未判定",
      reason: "",
      scoreText: "",
      relation: "解法 → 結果",
      analyzed: false,
    },
    {
      label: "未判定",
      reason: "",
      scoreText: "",
      relation: "結果 → 考察",
      analyzed: false,
    },
  ];
}

function createNode(depth = 0, title = "ルート", parent = null, parentLineIndex = null) {
  return {
    id: `node-${++nodeCounter}`,
    depth,
    title,
    parent,
    parentLineIndex,
    lines: ["", "", "", "", ""],
    children: [null, null, null, null, null],
    expanded: [false, false, false, false, false],
    results: createEmptyResults(),
  };
}

let tree = createNode(0, "ルート");

// ====== DOM ======
const els = {
  modelSelect: document.getElementById("model-select"),
  loadBtn: document.getElementById("load-btn"),
  analyzeRootBtn: document.getElementById("analyze-root-btn"),
  analyzeAllBtn: document.getElementById("analyze-all-btn"),
  copyMdBtn: document.getElementById("copy-md-btn"),
  clearBtn: document.getElementById("clear-btn"),
  modelStatus: document.getElementById("model-status"),
  modelDot: document.getElementById("model-dot"),
  deviceStatus: document.getElementById("device-status"),
  deviceDot: document.getElementById("device-dot"),
  progressText: document.getElementById("progress-text"),
  progressFill: document.getElementById("progress-fill"),
  treeRoot: document.getElementById("tree-root"),
  reasonsRoot: document.getElementById("reasons-root"),
};

function supportsWebGPU() {
  return typeof navigator !== "undefined" && !!navigator.gpu;
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setDot(el, state) {
  el.className = `dot ${state}`;
}

function setModelState(state, text) {
  setDot(els.modelDot, state);
  els.modelStatus.textContent = text;
}

function setDeviceState(state, text) {
  setDot(els.deviceDot, state);
  els.deviceStatus.textContent = text;
}

function setProgress(text, percent = null) {
  els.progressText.textContent = text;
  els.progressFill.style.width =
    typeof percent === "number" ? `${Math.max(0, Math.min(100, percent))}%` : "0%";
}

function badgeClass(label) {
  switch (label) {
    case "問題なし":
      return "badge-problem-none";
    case "飛躍":
      return "badge-jump";
    case "不足":
      return "badge-lack";
    case "未定義":
      return "badge-undefined";
    case "過剰":
      return "badge-excess";
    default:
      return "badge-wait";
  }
}

function shortLabel(label) {
  return label || "未判定";
}

// ====== Utilities ======
function countNonEmptyLines(node) {
  return node.lines.filter(v => v.trim()).length;
}

function summarizeNodeLines(node) {
  return node.lines
    .map((line, i) => `${LINE_NAMES[i]}: ${line.trim() || "未入力"}`)
    .join("\n");
}

function getNodePathTitles(node) {
  const titles = [];
  let current = node;
  while (current) {
    titles.unshift(current.title);
    current = current.parent;
  }
  return titles;
}

function getAncestorContext(node, maxDepth = 1) {
  const contexts = [];
  let current = node.parent;
  let depth = 0;
  while (current && depth < maxDepth) {
    contexts.unshift(
      [
        `親ブロック名: ${current.title}`,
        summarizeNodeLines(current),
      ].join("\n")
    );
    current = current.parent;
    depth += 1;
  }
  return contexts.join("\n\n");
}

function findNodeById(node, id) {
  if (node.id === id) return node;
  for (const child of node.children) {
    if (!child) continue;
    const found = findNodeById(child, id);
    if (found) return found;
  }
  return null;
}

function ensureChild(node, lineIndex) {
  if (!node.children[lineIndex]) {
    node.children[lineIndex] = createNode(
      node.depth + 1,
      `${LINE_NAMES[lineIndex]} の詳細`,
      node,
      lineIndex
    );
  }
}

function collapseChildrenRecursively(node) {
  node.expanded = [false, false, false, false, false];
  for (const child of node.children) {
    if (child) collapseChildrenRecursively(child);
  }
}

function collectNodes(node, list = []) {
  list.push(node);
  for (const child of node.children) {
    if (child) collectNodes(child, list);
  }
  return list;
}

function collectReasonItems(node, path = "ルート", items = []) {
  node.results.forEach((res, i) => {
    if (!res.analyzed) return;
    items.push({
      path,
      lineName: LINE_NAMES[i],
      lineIndex: i + 1,
      label: res.label,
      relation: res.relation || (i === 0 ? "比較対象なし" : `${LINE_NAMES[i - 1]} → ${LINE_NAMES[i]}`),
      reason: res.reason || "",
      scoreText: res.scoreText || "",
    });
  });

  node.children.forEach((child, i) => {
    if (!child) return;
    collectReasonItems(child, `${path} > ${LINE_NAMES[i]} の詳細`, items);
  });

  return items;
}

function nodeToMarkdown(node, level = 0) {
  const indent = "  ".repeat(level);
  const lines = [];

  for (let i = 0; i < 5; i++) {
    const text = node.lines[i].trim();
    if (!text && !node.children[i]) continue;

    lines.push(`${indent}- **${LINE_NAMES[i]}**: ${text || ""}`.trimEnd());

    if (node.children[i]) {
      const childMd = nodeToMarkdown(node.children[i], level + 1);
      if (childMd) lines.push(childMd);
    }
  }

  return lines.join("\n");
}

// ====== CPU Rule Analyzer ======
class CpuRuleAnalyzer {
  constructor() {
    this.name = "CPUルール分類器";
    this.device = "cpu";
  }

  async analyzeTransition(node, prevIndex, currIndex, prevText, currText) {
    const label = getHeuristicFlowLabel(prevIndex, currIndex, prevText, currText);
    return {
      label,
      scoreText: "rule-based",
    };
  }
}

function getHeuristicFlowLabel(prevIndex, currIndex, prevText, currText) {
  const prev = prevText.trim();
  const curr = currText.trim();

  if (!prev || !curr) return "問題なし";
  if (curr.length < 6) return "不足";

  const hasMethod =
    /提案|手法|方法|FiLM|LoRA|用いる|設計|導入|構築|学習/.test(curr);
  const hasResult =
    /結果|性能|向上|改善|精度|評価|実験|有効/.test(curr);
  const hasDiscussion =
    /示唆|考察|可能性|解釈|意味|有用|妥当/.test(curr);
  const hasProblem =
    /課題|問題|必要|困難|不足|未解決|ボトルネック/.test(curr);
  const hasDefinitionLike =
    /とは|定義|本研究では|ここでは/.test(curr);

  if (prevIndex === 0 && currIndex === 1) {
    if (hasProblem) return "問題なし";
    if (hasMethod || hasResult || hasDiscussion) return "飛躍";
    if (hasDefinitionLike) return "未定義";
    return "飛躍";
  }
  if (prevIndex === 1 && currIndex === 2) {
    if (hasMethod) return "問題なし";
    if (hasResult || hasDiscussion) return "飛躍";
    return "不足";
  }
  if (prevIndex === 2 && currIndex === 3) {
    if (hasResult) return "問題なし";
    if (hasDiscussion) return "飛躍";
    return "不足";
  }
  if (prevIndex === 3 && currIndex === 4) {
    if (hasDiscussion) return "問題なし";
    if (hasMethod) return "過剰";
    return "不足";
  }

  return "飛躍";
}

// ====== LLM Analyzer ======
class BrowserLLMAnalyzer {
  constructor(generator, profile, actualDevice) {
    this.generator = generator;
    this.profile = profile;
    this.device = actualDevice;
    this.name = profile.modelId;
  }

  buildPrompt(node, prevIndex, currIndex, prevText, currText) {
    const pathText = getNodePathTitles(node).join(" > ");
    const blockSummary = summarizeNodeLines(node);
    const ancestorContext = getAncestorContext(node, 1);
    const parentLineInfo =
      node.parent && node.parentLineIndex !== null
        ? `このブロックは親ブロックの「${LINE_NAMES[node.parentLineIndex]}」を詳細化したものです。`
        : "このブロックは最上位ブロックです。";

    return `
あなたは研究発表の論理構成チェッカーです。
必ず JSON のみを1個返してください。説明文は不要です。

出力形式:
{"label":"問題なし|飛躍|不足|未定義|過剰","reason":"40字以内の簡潔な理由"}

判定基準:
- 問題なし: 前の行から自然につながる
- 飛躍: 論理段階が飛んでいる
- 不足: 必要情報が足りない
- 未定義: 用語・主語・対象が曖昧
- 過剰: その段階で言い過ぎている

ブロック階層: ${pathText}
${parentLineInfo}

${ancestorContext ? `上位文脈:\n${ancestorContext}\n` : ""}

現在のブロック全体:
${blockSummary}

判定対象:
前の行の役割: ${LINE_NAMES[prevIndex]}
次の行の役割: ${LINE_NAMES[currIndex]}
前の文: ${prevText}
次の文: ${currText}
`.trim();
  }

  async analyzeTransition(node, prevIndex, currIndex, prevText, currText) {
    if (!prevText.trim() || !currText.trim()) {
      return { label: "問題なし", scoreText: "空欄を含むため問題なし扱い" };
    }

    const messages = [
      {
        role: "system",
        content: "You are a strict Japanese research logic checker. Return only JSON.",
      },
      {
        role: "user",
        content: this.buildPrompt(node, prevIndex, currIndex, prevText, currText),
      },
    ];

    const out = await this.generator(messages, {
      max_new_tokens: this.profile.maxNewTokens ?? 220,
      do_sample: false,
      temperature: 0,
      return_full_text: false,
    });

    const text = extractGeneratedText(out);
    const parsed = tryParseJsonFromText(text);

    if (!parsed || !FLOW_LABELS.includes(parsed.label)) {
      return {
        label: getHeuristicFlowLabel(prevIndex, currIndex, prevText, currText),
        scoreText: "LLM parse fallback",
      };
    }

    return {
      label: parsed.label,
      scoreText: `LLM(${this.device})`,
      llmReason: String(parsed.reason || "").trim(),
    };
  }
}

function extractGeneratedText(output) {
  if (typeof output === "string") return output;

  if (Array.isArray(output) && output.length > 0) {
    const first = output[0];
    if (typeof first === "string") return first;

    if (first?.generated_text) {
      if (typeof first.generated_text === "string") return first.generated_text;
      if (Array.isArray(first.generated_text)) {
        const last = first.generated_text.at(-1);
        if (typeof last?.content === "string") return last.content;
      }
    }
  }

  return "";
}

function tryParseJsonFromText(text) {
  if (!text) return null;

  try {
    return JSON.parse(text);
  } catch (_) {}

  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return null;

  try {
    return JSON.parse(match[0]);
  } catch (_) {
    return null;
  }
}

// ====== Model Manager ======
async function createAnalyzer(modelKey) {
  const profile = MODEL_PROFILES[modelKey];
  if (!profile) throw new Error(`Unknown model key: ${modelKey}`);

  if (profile.kind === "rule") {
    return new CpuRuleAnalyzer();
  }

  if (!supportsWebGPU()) {
    return new CpuRuleAnalyzer();
  }

  const generator = await pipeline("text-generation", profile.modelId, {
    device: profile.preferredDevice,
    dtype: "q4",
    progress_callback: (progress) => {
      if (!progress) return;
      const p = progress.progress ?? 0;
      const status = progress.status ?? "loading";
      setProgress(`読込中: ${status}`, p * 100);
    },
  });

  return new BrowserLLMAnalyzer(generator, profile, "webgpu");
}

async function loadModel() {
  if (APP_STATE.isLoading) return;

  APP_STATE.isLoading = true;
  lockUi(true);

  const modelKey = els.modelSelect.value;
  APP_STATE.modelKey = modelKey;

  const profile = MODEL_PROFILES[modelKey];
  setModelState("loading", `${profile.label} を読込中`);
  setProgress("モデル初期化中...", 0);

  try {
    const analyzer = await createAnalyzer(modelKey);
    APP_STATE.analyzer = analyzer;
    APP_STATE.modelLoaded = true;
    APP_STATE.selectedDevice = analyzer.device;

    setModelState("ready", `${profile.label} 読込完了`);
    setDeviceState(
      "ready",
      analyzer.device === "webgpu"
        ? "WebGPU で実行"
        : "CPU ルール分類器で実行"
    );
    setProgress("準備完了", 100);
  } catch (err) {
    console.error(err);
    APP_STATE.analyzer = new CpuRuleAnalyzer();
    APP_STATE.modelLoaded = true;
    APP_STATE.selectedDevice = "cpu";

    setModelState("error", "GPUモデル読込失敗");
    setDeviceState("ready", "CPU ルール分類器へフォールバック");
    setProgress("CPU モードで準備完了", 100);
  } finally {
    APP_STATE.isLoading = false;
    lockUi(false);
    render();
  }
}

// ====== Analysis ======
function buildReason(lineIndex, label, prevText, currText, llmReason = "") {
  const currRole = LINE_NAMES[lineIndex];
  const prevRole = lineIndex > 0 ? LINE_NAMES[lineIndex - 1] : null;

  if (lineIndex === 0) {
    return "1行目は比較対象がないため固定で問題なしです。";
  }
  if (!currText.trim()) {
    return `${currRole}が空欄です。`;
  }
  if (!prevText.trim()) {
    return `前の行が空欄のため接続を確定できません。`;
  }
  if (llmReason) return llmReason;

  switch (label) {
    case "問題なし":
      return `${prevRole}から${currRole}への接続は自然です。`;
    case "飛躍":
      return `${prevRole}から${currRole}への論理段階が飛んでいます。`;
    case "不足":
      return `${currRole}に必要な説明が不足しています。`;
    case "未定義":
      return `${currRole}の対象や用語が曖昧です。`;
    case "過剰":
      return `${currRole}としては踏み込みすぎています。`;
    default:
      return `${prevRole}から${currRole}の接続に問題があります。`;
  }
}

async function analyzeNode(node) {
  node.results[0] = {
    label: "問題なし",
    reason: "1行目は比較対象がないため固定で問題なしです。",
    scoreText: "固定判定",
    relation: "比較対象なし",
    analyzed: true,
  };

  for (let i = 1; i < 5; i++) {
    const prevText = node.lines[i - 1];
    const currText = node.lines[i];

    const result = await APP_STATE.analyzer.analyzeTransition(
      node, i - 1, i, prevText, currText
    );

    node.results[i] = {
      label: result.label,
      reason: buildReason(i, result.label, prevText, currText, result.llmReason),
      scoreText: result.scoreText,
      relation: `${LINE_NAMES[i - 1]} → ${LINE_NAMES[i]}`,
      analyzed: true,
    };
  }
}

async function analyzeSingleNode(nodeId) {
  if (!APP_STATE.modelLoaded || APP_STATE.isChecking) return;
  const node = findNodeById(tree, nodeId);
  if (!node) return;

  APP_STATE.isChecking = true;
  lockUi(true);

  try {
    setProgress(`ブロック ${node.title} を判定中...`, null);
    await analyzeNode(node);
    render();
    setProgress("ブロック判定完了", 100);
  } finally {
    APP_STATE.isChecking = false;
    lockUi(false);
  }
}

async function analyzeAllNodes() {
  if (!APP_STATE.modelLoaded || APP_STATE.isChecking) return;

  APP_STATE.isChecking = true;
  lockUi(true);

  try {
    const nodes = collectNodes(tree, []);
    for (let idx = 0; idx < nodes.length; idx++) {
      setProgress(`全体判定中... ${idx + 1}/${nodes.length}`, ((idx + 1) / nodes.length) * 100);
      await analyzeNode(nodes[idx]);
      render();
      await tick();
    }
    setProgress("全ブロックの判定完了", 100);
  } finally {
    APP_STATE.isChecking = false;
    lockUi(false);
  }
}

function tick() {
  return new Promise(resolve => setTimeout(resolve, 0));
}

// ====== Clipboard / Clear ======
async function copyTextToClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.setAttribute("readonly", "");
  textArea.style.position = "fixed";
  textArea.style.top = "-9999px";
  textArea.style.left = "-9999px";
  document.body.appendChild(textArea);

  textArea.focus();
  textArea.select();

  const successful = document.execCommand("copy");
  document.body.removeChild(textArea);

  if (!successful) {
    throw new Error("copy failed");
  }
}

async function copyMarkdown() {
  const md = nodeToMarkdown(tree, 0);
  if (!md.trim()) {
    alert("コピーする内容がありません。");
    return;
  }

  const originalText = els.copyMdBtn.textContent;
  els.copyMdBtn.disabled = true;

  try {
    await copyTextToClipboard(md);
    els.copyMdBtn.textContent = "コピーしました";
    setProgress("Markdownをクリップボードへ保存しました", 100);
  } catch (e) {
    console.error(e);
    alert("クリップボードへの保存に失敗しました。");
  } finally {
    setTimeout(() => {
      els.copyMdBtn.textContent = originalText;
      els.copyMdBtn.disabled = false;
    }, 1200);
  }
}

function clearAll() {
  tree = createNode(0, "ルート");
  render();
  setProgress(APP_STATE.modelLoaded ? "準備完了" : "待機中", APP_STATE.modelLoaded ? 100 : 0);
}

// ====== Render ======
function renderReasonsPanel() {
  const items = collectReasonItems(tree);

  if (items.length === 0) {
    els.reasonsRoot.innerHTML = `<div class="empty-reasons">まだ判定結果がありません。</div>`;
    return;
  }

  els.reasonsRoot.innerHTML = items.map((item) => `
    <div class="reason-item">
      <div class="reason-head">
        <div class="reason-title">${escapeHtml(item.path)} / ${item.lineIndex}行目（${item.lineName}）</div>
        <span class="line-badge ${badgeClass(item.label)}">${escapeHtml(item.label)}</span>
      </div>
      <div class="reason-body">${escapeHtml(item.relation)}
${escapeHtml(item.reason)}</div>
      <div class="reason-meta">${escapeHtml(item.scoreText)}</div>
    </div>
  `).join("");
}

function renderNode(node) {
  const linesHtml = node.lines.map((line, i) => {
    const child = node.children[i];
    const hasChild = !!child;
    const isExpanded = !!node.expanded[i];
    const buttonLabel = hasChild ? (isExpanded ? "詳細を閉じる" : "詳細を開く") : "詳細を作成";

    return `
      <div class="line-row">
        <div class="line-top">
          <div class="line-label">${i + 1}. ${LINE_NAMES[i]}</div>
          <div class="line-controls">
            <button class="ghost small" data-action="toggle-child" data-node-id="${node.id}" data-line-index="${i}">
              ${escapeHtml(buttonLabel)}
            </button>
            ${hasChild ? `
              <button class="ghost small" data-action="remove-child" data-node-id="${node.id}" data-line-index="${i}">
                詳細を削除
              </button>
            ` : ""}
          </div>
        </div>

        <textarea data-action="edit-line" data-node-id="${node.id}" data-line-index="${i}">${escapeHtml(line)}</textarea>

        ${hasChild && isExpanded ? `
          <div class="child-wrap">
            ${renderNode(child)}
          </div>
        ` : ""}
      </div>
    `;
  }).join("");

  const badgesHtml = node.results.map((res, i) => `
    <span class="line-badge ${badgeClass(res.label)}">
      ${i + 1}:${escapeHtml(shortLabel(res.label))}
    </span>
  `).join("");

  return `
    <section class="block depth-${Math.min(node.depth, 5)}">
      <div class="block-header">
        <div class="block-title-wrap">
          <div class="block-title">${escapeHtml(node.title)}</div>
          <div class="block-meta">深さ ${node.depth} / 入力済み ${countNonEmptyLines(node)} 行 / ID ${escapeHtml(node.id)}</div>
        </div>

        <div class="block-side">
          <div class="block-badges">${badgesHtml}</div>
          <div class="block-actions">
            <button class="secondary small" data-action="analyze-node" data-node-id="${node.id}" ${APP_STATE.modelLoaded ? "" : "disabled"}>
              このブロックを判定
            </button>
            ${node.depth > 0 ? `
              <button class="ghost small" data-action="collapse-all-children" data-node-id="${node.id}">
                子をたたむ
              </button>
            ` : ""}
          </div>
        </div>
      </div>

      ${linesHtml}
    </section>
  `;
}

function render() {
  els.treeRoot.innerHTML = renderNode(tree);
  renderReasonsPanel();

  els.analyzeRootBtn.disabled = !APP_STATE.modelLoaded;
  els.analyzeAllBtn.disabled = !APP_STATE.modelLoaded;
}

function lockUi(locked) {
  els.loadBtn.disabled = locked;
  els.analyzeRootBtn.disabled = locked || !APP_STATE.modelLoaded;
  els.analyzeAllBtn.disabled = locked || !APP_STATE.modelLoaded;
  els.copyMdBtn.disabled = locked;
  els.clearBtn.disabled = locked;
  els.modelSelect.disabled = locked;
}

// ====== Events ======
els.modelSelect.addEventListener("change", () => {
  APP_STATE.modelLoaded = false;
  APP_STATE.analyzer = null;

  const selected = MODEL_PROFILES[els.modelSelect.value];
  setModelState("ready", `${selected.label} 未ロード`);

  if (selected.kind === "rule") {
    setDeviceState("ready", "CPU を使用予定");
  } else {
    setDeviceState(
      supportsWebGPU() ? "ready" : "error",
      supportsWebGPU() ? "WebGPU 使用予定" : "WebGPU 非対応のため CPU へフォールバック予定"
    );
  }

  setProgress("待機中", 0);
  render();
});

els.treeRoot.addEventListener("input", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLTextAreaElement)) return;
  if (target.dataset.action !== "edit-line") return;

  const nodeId = target.dataset.nodeId;
  const lineIndex = Number(target.dataset.lineIndex);
  const node = findNodeById(tree, nodeId);
  if (!node || Number.isNaN(lineIndex)) return;

  node.lines[lineIndex] = target.value;
});

els.treeRoot.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;

  const button = target.closest("button");
  if (!button) return;

  const action = button.dataset.action;
  const nodeId = button.dataset.nodeId;
  const lineIndex = Number(button.dataset.lineIndex);
  const node = nodeId ? findNodeById(tree, nodeId) : null;

  if (action === "toggle-child") {
    if (!node || Number.isNaN(lineIndex)) return;
    if (!node.children[lineIndex]) ensureChild(node, lineIndex);
    node.expanded[lineIndex] = !node.expanded[lineIndex];
    render();
    return;
  }

  if (action === "remove-child") {
    if (!node || Number.isNaN(lineIndex)) return;
    node.children[lineIndex] = null;
    node.expanded[lineIndex] = false;
    render();
    return;
  }

  if (action === "analyze-node") {
    if (!node) return;
    await analyzeSingleNode(node.id);
    return;
  }

  if (action === "collapse-all-children") {
    if (!node) return;
    collapseChildrenRecursively(node);
    render();
  }
});

els.loadBtn.addEventListener("click", loadModel);
els.analyzeRootBtn.addEventListener("click", async () => {
  await analyzeSingleNode(tree.id);
});
els.analyzeAllBtn.addEventListener("click", analyzeAllNodes);
els.copyMdBtn.addEventListener("click", copyMarkdown);
els.clearBtn.addEventListener("click", clearAll);

// ====== Init ======
env.allowRemoteModels = true;

setModelState("ready", "未ロード");
setDeviceState(
  supportsWebGPU() ? "ready" : "error",
  supportsWebGPU() ? "WebGPU 利用可能" : "WebGPU 非対応"
);
setProgress("待機中", 0);
render();
