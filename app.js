const APP_VERSION = "v0.3.2";
console.log("FlowChecker version:", APP_VERSION);
document.getElementById("app-version").textContent = `Version: ${APP_VERSION}`;


import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@next";

const LINE_NAMES = ["背景", "課題", "解法", "結果", "考察"];
const CPU_FLOW_LABELS = ["問題なし", "飛躍"];

const SCORE_MAP = {
  "問題なし": 1.0,
  "要確認": 0.5,
  "飛躍": 0.0,
};

const MODEL_OPTIONS = {
  small: {
    kind: "generator",
    label: "GPU小: 0.5B",
    model: "onnx-community/Qwen2.5-0.5B-Instruct",
    device: "webgpu",
    dtype: "q4",
  },
  medium: {
    kind: "generator",
    label: "GPU中: 3B",
    model: "onnx-community/Llama-3.2-3B-Instruct-ONNX",
    device: "webgpu",
    dtype: "q4f16",
  },
  large: {
    kind: "generator",
    label: "GPU大: 7B",
    model: "onnx-community/Olmo-3-7B-Instruct-ONNX",
    device: "webgpu",
    dtype: "q4f16",
  },
  cpu: {
    kind: "classifier",
    label: "CPU分類器",
    model: "onnx-community/multilingual-MiniLMv2-L6-mnli-xnli-ONNX",
    device: "wasm",
    dtype: "q8",
  },
};

let generator = null;
let classifier = null;
let activeBackend = null;
let isLoading = false;
let isChecking = false;
let nodeCounter = 0;

const CPU_BORDER_MARGIN = 0.18;

function createEmptyResults() {
  return [
    {
      label: "問題なし",
      score: 1.0,
      reason: "1行目は比較対象がないため固定で問題なしです。",
      scoreText: "固定判定",
      relation: "比較対象なし",
      analyzed: true,
    },
    {
      label: "未判定",
      score: null,
      reason: "",
      scoreText: "",
      relation: "背景 → 課題",
      analyzed: false,
    },
    {
      label: "未判定",
      score: null,
      reason: "",
      scoreText: "",
      relation: "課題 → 解法",
      analyzed: false,
    },
    {
      label: "未判定",
      score: null,
      reason: "",
      scoreText: "",
      relation: "解法 → 結果",
      analyzed: false,
    },
    {
      label: "未判定",
      score: null,
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

function setModelState(state, text) {
  els.modelStatus.textContent = text;
  els.modelDot.className = `dot ${state}`;
}

function setDeviceState(state, text) {
  els.deviceStatus.textContent = text;
  els.deviceDot.className = `dot ${state}`;
}

function setProgress(text, percent = null) {
  els.progressText.textContent = text;
  if (percent === null || Number.isNaN(percent)) {
    els.progressFill.style.width = "0%";
  } else {
    const p = Math.max(0, Math.min(100, percent));
    els.progressFill.style.width = `${p}%`;
  }
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function badgeClass(label) {
  switch (label) {
    case "問題なし":
      return "badge-problem-none";
    case "飛躍":
      return "badge-jump";
    case "要確認":
      return "badge-wait";
    default:
      return "badge-wait";
  }
}

function countNonEmptyLines(node) {
  return node.lines.filter((v) => v.trim()).length;
}

function shortLabel(label) {
  switch (label) {
    case "問題なし":
      return "問題なし";
    case "飛躍":
      return "飛躍";
    case "要確認":
      return "要確認";
    default:
      return "未判定";
  }
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

function buildGenerationPrompt(node, prevIndex, currIndex, prevText, currText) {
  const pathText = getNodePathTitles(node).join(" > ");
  const blockSummary = summarizeNodeLines(node);
  const ancestorContext = getAncestorContext(node, 1);
  const parentLineInfo =
    node.parent && node.parentLineIndex !== null
      ? `This block expands the parent line "${LINE_NAMES[node.parentLineIndex]}".`
      : "This is the top-level block.";

  return [
    "You evaluate the transition between two adjacent lines in a research outline.",
    "Judge only the flow from the previous line to the next line.",
    "Do not rely only on topic similarity. Focus on research-structure transition.",
    "",
    `Block path: ${pathText}`,
    parentLineInfo,
    ancestorContext ? `Ancestor context:\n${ancestorContext}` : "",
    `Current block:\n${blockSummary}`,
    "",
    `Previous role: ${LINE_NAMES[prevIndex]}`,
    `Next role: ${LINE_NAMES[currIndex]}`,
    `Previous line: ${prevText}`,
    `Next line: ${currText}`,
    "",
    "Return only one word.",
    "Allowed outputs:",
    "GOOD",
    "BORDER",
    "BAD",
    "",
    "Do not output any explanation.",
    "Do not output JSON.",
    "Do not output punctuation."
  ]
    .filter(Boolean)
    .join("\n");
}

function buildClassifierPrompt(node, prevIndex, currIndex, prevText, currText) {
  const pathText = getNodePathTitles(node).join(" > ");
  const blockSummary = summarizeNodeLines(node);
  const ancestorContext = getAncestorContext(node, 1);
  const parentLineInfo =
    node.parent && node.parentLineIndex !== null
      ? `このブロックは親ブロックの「${LINE_NAMES[node.parentLineIndex]}」を詳細化したものです。`
      : "このブロックは最上位ブロックです。";

  return [
    `ブロック階層: ${pathText}`,
    parentLineInfo,
    ancestorContext ? `上位文脈:\n${ancestorContext}` : "",
    `現在のブロック全体:\n${blockSummary}`,
    "",
    `前の行の役割: ${LINE_NAMES[prevIndex]}`,
    `次の行の役割: ${LINE_NAMES[currIndex]}`,
    `前の文: ${prevText}`,
    `次の文: ${currText}`,
    "",
    "前の行から次の行への接続が研究構成上自然かを判定する。",
    "表面的な話題類似だけではなく、役割の遷移が自然かを見る。",
    "使用可能なラベルは次の2つのみ。",
    "問題なし: 前の行から次の行へ自然につながっている。",
    "飛躍: 前の行から次の行への遷移が不自然である。",
  ]
    .filter(Boolean)
    .join("\n");
}

function buildReason(lineIndex, label, prevText, currText, extra = "") {
  const currRole = LINE_NAMES[lineIndex];
  const prevRole = lineIndex > 0 ? LINE_NAMES[lineIndex - 1] : null;

  if (!currText.trim()) {
    return `${currRole}が空欄のため、問題なし扱いにしています。`;
  }
  if (lineIndex === 0) {
    return "1行目は比較対象がないため固定で問題なしです。";
  }
  if (!prevText.trim()) {
    return `前の行が空欄のため、接続判定は行わず問題なし扱いにしています。`;
  }
  if (label === "飛躍") {
    return `${prevRole}から${currRole}への遷移が不自然です。${extra}`.trim();
  }
  if (label === "要確認") {
    return `${prevRole}から${currRole}への接続は弱い、または判断が難しいです。${extra}`.trim();
  }
  return `${prevRole}から${currRole}への接続は自然です。${extra}`.trim();
}

function labelToScore(label) {
  return SCORE_MAP[label] ?? null;
}

function scoreToPercent(score) {
  if (score == null) return null;
  return Math.round(score * 100);
}

function computeNodeScore(node) {
  const vals = node.results
    .filter((r) => typeof r.score === "number")
    .map((r) => r.score);
  if (vals.length === 0) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

function computeTreeScore(node) {
  const vals = [];

  function walk(n) {
    for (const r of n.results) {
      if (typeof r.score === "number") vals.push(r.score);
    }
    for (const c of n.children) {
      if (c) walk(c);
    }
  }

  walk(node);
  if (vals.length === 0) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

function extractBand(text) {
  const upper = String(text || "").toUpperCase();

  if (upper.includes("GOOD")) return "GOOD";
  if (upper.includes("BORDER")) return "BORDER";
  if (upper.includes("BAD")) return "BAD";

  return null;
}

async function detectWebGPU() {
  try {
    if (!window.isSecureContext) {
      return { ok: false, reason: "secure context ではありません" };
    }
    if (!("gpu" in navigator) || !navigator.gpu) {
      return { ok: false, reason: "navigator.gpu がありません" };
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return { ok: false, reason: "GPU adapter を取得できません" };
    }
    await adapter.requestDevice();
    return { ok: true, reason: "WebGPU 利用可能" };
  } catch (e) {
    return { ok: false, reason: `WebGPU 初期化失敗: ${e?.message || "unknown error"}` };
  }
}

async function runGeneratorJudge(node, prevIndex, currIndex, prevText, currText) {
  const prompt = buildGenerationPrompt(node, prevIndex, currIndex, prevText, currText);

  const out = await generator(prompt, {
    max_new_tokens: 8,
    do_sample: false,
    temperature: 0,
    return_full_text: false,
  });

  const generated = Array.isArray(out) ? out[0]?.generated_text ?? "" : "";
  const band = extractBand(generated);

  if (!band) {
    throw new Error(`生成結果から band を抽出できませんでした: ${generated}`);
  }

  let label = "要確認";
  if (band === "GOOD") label = "問題なし";
  if (band === "BAD") label = "飛躍";
  if (band === "BORDER") label = "要確認";

  return {
    label,
    score: labelToScore(label),
    reason: buildReason(currIndex, label, prevText, currText),
    scoreText: `generator band=${band}`,
  };
}

async function runClassifierJudge(node, prevIndex, currIndex, prevText, currText) {
  const sequence = buildClassifierPrompt(node, prevIndex, currIndex, prevText, currText);
  const output = await classifier(sequence, CPU_FLOW_LABELS, {
    multi_label: false,
    hypothesis_template: "この接続は {}。",
  });

  const labels = Array.isArray(output.labels) ? output.labels : [];
  const scores = Array.isArray(output.scores) ? output.scores : [];
  const topLabel = labels[0] || "飛躍";
  const topScore = typeof scores[0] === "number" ? scores[0] : 0;
  const secondScore = typeof scores[1] === "number" ? scores[1] : 0;
  const margin = Math.abs(topScore - secondScore);

  let label = CPU_FLOW_LABELS.includes(topLabel) ? topLabel : "飛躍";
  if (margin < CPU_BORDER_MARGIN) {
    label = "要確認";
  }

  return {
    label,
    score: labelToScore(label),
    reason: buildReason(
      currIndex,
      label,
      prevText,
      currText,
      label === "要確認" ? "分類スコア差が小さいため、境界的とみなしました。" : ""
    ),
    scoreText: `classifier top=${topScore.toFixed(3)}, margin=${margin.toFixed(3)}`,
  };
}

async function classifyFlow(node, prevIndex, currIndex, prevText, currText) {
  if (!prevText.trim() || !currText.trim()) {
    return {
      label: "問題なし",
      score: 1.0,
      reason: buildReason(currIndex, "問題なし", prevText, currText),
      scoreText: "空欄を含むため問題なし扱い",
    };
  }

  try {
    if (activeBackend?.kind === "generator" && generator) {
      return await runGeneratorJudge(node, prevIndex, currIndex, prevText, currText);
    }
    if (activeBackend?.kind === "classifier" && classifier) {
      return await runClassifierJudge(node, prevIndex, currIndex, prevText, currText);
    }
  } catch (e) {
    console.warn("モデル推論失敗:", e);
  }

  return {
    label: "要確認",
    score: 0.5,
    reason: buildReason(currIndex, "要確認", prevText, currText, "モデル推論に失敗したため判定を保留しました。"),
    scoreText: "model error",
  };
}

function resetNodeResults(node) {
  node.results = createEmptyResults();
  for (const child of node.children) {
    if (child) resetNodeResults(child);
  }
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
    if (child) {
      collapseChildrenRecursively(child);
    }
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
      score: res.score,
    });
  });

  node.children.forEach((child, i) => {
    if (!child) return;
    collectReasonItems(child, `${path} > ${LINE_NAMES[i]} の詳細`, items);
  });

  return items;
}

function renderReasonsPanel() {
  const overallScore = computeTreeScore(tree);
  const items = collectReasonItems(tree);

  let html = `
    <div class="reason-item">
      <div class="reason-head">
        <div class="reason-title">全体スコア</div>
        <span class="line-badge ${badgeClass(
          overallScore == null
            ? "要確認"
            : overallScore >= 0.75
              ? "問題なし"
              : overallScore >= 0.4
                ? "要確認"
                : "飛躍"
        )}">
          ${overallScore == null ? "未計算" : `${scoreToPercent(overallScore)}点`}
        </span>
      </div>
      <div class="reason-body">現在ロード中の判定系: ${escapeHtml(activeBackend ? activeBackend.label : "未ロード")}</div>
      <div class="reason-meta">${
        overallScore == null
          ? "まだスコアがありません。"
          : `平均スコア=${overallScore.toFixed(3)}`
      }</div>
    </div>
  `;

  if (items.length === 0) {
    html += `<div class="empty-reasons">まだ判定結果がありません。</div>`;
    els.reasonsRoot.innerHTML = html;
    return;
  }

  html += items.map((item) => `
    <div class="reason-item">
      <div class="reason-head">
        <div class="reason-title">${escapeHtml(item.path)} / ${item.lineIndex}行目（${item.lineName}）</div>
        <span class="line-badge ${badgeClass(item.label)}">
          ${escapeHtml(item.label)}${typeof item.score === "number" ? ` (${scoreToPercent(item.score)}点)` : ""}
        </span>
      </div>
      <div class="reason-body">${escapeHtml(item.relation)}
${escapeHtml(item.reason)}</div>
      <div class="reason-meta">${escapeHtml(item.scoreText)}</div>
    </div>
  `).join("");

  els.reasonsRoot.innerHTML = html;
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

  const nodeScore = computeNodeScore(node);

  return `
    <section class="block depth-${Math.min(node.depth, 5)}">
      <div class="block-header">
        <div class="block-title-wrap">
          <div class="block-title">${escapeHtml(node.title)}</div>
          <div class="block-meta">
            深さ ${node.depth} / 入力済み ${countNonEmptyLines(node)} 行 / ID ${escapeHtml(node.id)}
            ${nodeScore == null ? "" : ` / スコア ${scoreToPercent(nodeScore)}点`}
          </div>
        </div>

        <div class="block-side">
          <div class="block-badges">${badgesHtml}</div>
          <div class="block-actions">
            <button class="secondary small" data-action="analyze-node" data-node-id="${node.id}" ${activeBackend ? "" : "disabled"}>
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
  els.analyzeRootBtn.disabled = !activeBackend;
  els.analyzeAllBtn.disabled = !activeBackend;
}

async function analyzeNode(node) {
  node.results[0] = {
    label: "問題なし",
    score: 1.0,
    reason: "1行目は比較対象がないため固定で問題なしです。",
    scoreText: "固定判定",
    relation: "比較対象なし",
    analyzed: true,
  };

  for (let i = 1; i < 5; i++) {
    const flowResult = await classifyFlow(node, i - 1, i, node.lines[i - 1], node.lines[i]);
    node.results[i] = {
      label: flowResult.label,
      score: flowResult.score,
      reason: flowResult.reason,
      scoreText: flowResult.scoreText,
      relation: `${LINE_NAMES[i - 1]} → ${LINE_NAMES[i]}`,
      analyzed: true,
    };
  }
}

async function analyzeSingleNode(nodeId) {
  if (!activeBackend || isChecking) return;
  const node = findNodeById(tree, nodeId);
  if (!node) return;

  isChecking = true;
  els.analyzeAllBtn.disabled = true;
  els.analyzeRootBtn.disabled = true;
  els.loadBtn.disabled = true;
  els.copyMdBtn.disabled = true;
  if (els.modelSelect) els.modelSelect.disabled = true;

  setProgress(`ブロック ${node.title} を判定中...`, null);
  await analyzeNode(node);
  render();

  setProgress("ブロック判定完了", 100);
  els.analyzeAllBtn.disabled = false;
  els.analyzeRootBtn.disabled = false;
  els.loadBtn.disabled = false;
  els.copyMdBtn.disabled = false;
  if (els.modelSelect) els.modelSelect.disabled = false;
  isChecking = false;
}

async function analyzeAllNodes() {
  if (!activeBackend || isChecking) return;

  isChecking = true;
  els.analyzeAllBtn.disabled = true;
  els.analyzeRootBtn.disabled = true;
  els.loadBtn.disabled = true;
  els.copyMdBtn.disabled = true;
  if (els.modelSelect) els.modelSelect.disabled = true;

  const nodes = collectNodes(tree, []);
  for (let idx = 0; idx < nodes.length; idx++) {
    setProgress(`全体判定中... ${idx + 1}/${nodes.length}`, ((idx + 1) / nodes.length) * 100);
    await analyzeNode(nodes[idx]);
    render();
  }

  setProgress("全ブロックの判定完了", 100);
  els.analyzeAllBtn.disabled = false;
  els.analyzeRootBtn.disabled = false;
  els.loadBtn.disabled = false;
  els.copyMdBtn.disabled = false;
  if (els.modelSelect) els.modelSelect.disabled = false;
  isChecking = false;
}

async function loadModel() {
  if (isLoading) return;

  isLoading = true;
  generator = null;
  classifier = null;
  activeBackend = null;

  els.loadBtn.disabled = true;
  els.analyzeRootBtn.disabled = true;
  els.analyzeAllBtn.disabled = true;
  els.copyMdBtn.disabled = true;
  if (els.modelSelect) els.modelSelect.disabled = true;

  const selectedKey = els.modelSelect ? els.modelSelect.value : "medium";
  const selected = MODEL_OPTIONS[selectedKey] || MODEL_OPTIONS.medium;

  setModelState("loading", `モデル読込中: ${selected.label}`);
  setProgress("モデルを読み込んでいます...", 0);

  try {
    if (selected.kind === "generator") {
      const gpu = await detectWebGPU();

      if (!gpu.ok) {
        setModelState("error", `GPUモデル読込失敗: ${selected.label}`);
        setDeviceState("error", `WebGPU 使用不可: ${gpu.reason}`);
        setProgress("GPUが使用できないため停止しました", 0);
        alert(`GPUモデルを選択しましたが、WebGPUが使用できませんでした。\n理由: ${gpu.reason}`);
        return;
      }

      setDeviceState("loading", "WebGPU を使用");
      generator = await pipeline("text-generation", selected.model, {
        device: selected.device,
        dtype: selected.dtype,
        progress_callback: (progress) => {
          if (!progress) return;
          const p = progress.progress ?? 0;
          const status = progress.status ?? "loading";
          setProgress(`読込中: ${status}`, p * 100);
        },
      });

      activeBackend = selected;
      setModelState("ready", `読込完了: ${selected.label}`);
      setDeviceState("ready", "WebGPU で実行");
    } else {
      setDeviceState("loading", "CPU (WASM) を使用");
      classifier = await pipeline("zero-shot-classification", selected.model, {
        device: selected.device,
        dtype: selected.dtype,
        progress_callback: (progress) => {
          if (!progress) return;
          const p = progress.progress ?? 0;
          const status = progress.status ?? "loading";
          setProgress(`読込中: ${status}`, p * 100);
        },
      });

      activeBackend = selected;
      setModelState("ready", `読込完了: ${selected.label}`);
      setDeviceState("ready", "CPU (WASM) で実行");
    }

    setProgress("準備完了", 100);
    els.analyzeRootBtn.disabled = false;
    els.analyzeAllBtn.disabled = false;
  } catch (err) {
    console.error(err);
    activeBackend = null;
    generator = null;
    classifier = null;

    setModelState("error", `モデル読込失敗: ${selected.label}`);
    setDeviceState("error", selected.kind === "generator" ? "WebGPU 初期化またはモデル読込失敗" : "CPU分類器の読込失敗");
    setProgress("読込に失敗しました", 0);
    alert(`モデルの読み込みに失敗しました。\n選択モデル: ${selected.label}`);
  } finally {
    isLoading = false;
    els.loadBtn.disabled = false;
    els.copyMdBtn.disabled = false;
    if (els.modelSelect) els.modelSelect.disabled = false;
    render();
  }
}

function clearAll() {
  tree = createNode(0, "ルート");
  resetNodeResults(tree);
  render();
  setProgress(activeBackend ? "準備完了" : "待機中", activeBackend ? 100 : 0);
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

  const selection = document.getSelection();
  const originalRange =
    selection && selection.rangeCount > 0 ? selection.getRangeAt(0) : null;

  textArea.focus();
  textArea.select();

  const successful = document.execCommand("copy");
  document.body.removeChild(textArea);

  if (originalRange && selection) {
    selection.removeAllRanges();
    selection.addRange(originalRange);
  }

  if (!successful) {
    throw new Error("execCommand copy failed");
  }
}

async function copyMarkdown() {
  const md = nodeToMarkdown(tree, 0);
  if (!md.trim()) {
    alert("コピーする内容がありません。");
    return;
  }

  els.copyMdBtn.disabled = true;
  const originalText = els.copyMdBtn.textContent;

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
    if (!node.children[lineIndex]) {
      ensureChild(node, lineIndex);
    }
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
    if (node.parent && node.parentLineIndex !== null) {
      node.parent.expanded[node.parentLineIndex] = false;
    }
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

setDeviceState("ready", "未確認");
setModelState("ready", "未ロード");
setProgress("待機中", 0);

render();
