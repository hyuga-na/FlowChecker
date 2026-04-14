import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.0";

const MODEL_ID = "xavierbarbier/xlm-roberta-large-xnli";
const LINE_NAMES = ["背景", "課題", "解法", "結果", "考察"];
const LABELS = ["問題なし", "飛躍", "不足", "未定義", "過剰"];

let classifier = null;
let selectedDevice = "wasm";
let isLoading = false;
let isChecking = false;
let nodeCounter = 0;

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

const els = {
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
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

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

function countNonEmptyLines(node) {
  return node.lines.filter((v) => v.trim()).length;
}

function shortLabel(label) {
  switch (label) {
    case "問題なし":
      return "問題なし";
    case "飛躍":
      return "飛躍";
    case "不足":
      return "不足";
    case "未定義":
      return "未定義";
    case "過剰":
      return "過剰";
    default:
      return "未判定";
  }
}

function lineSpecificReason(label, prevRole, currRole, prevText, currText) {
  const hasPrev = !!prevText.trim();
  const hasCurr = !!currText.trim();

  if (!hasPrev || !hasCurr) {
    return "空欄を含むため判定対象外とし、問題なし扱いにしました。";
  }

  const transition = `${prevRole}から${currRole}への接続`;

  switch (label) {
    case "問題なし":
      return `${transition}は自然です。直前の内容から次の主張が無理なく導かれています。`;
    case "飛躍":
      return `${transition}に中間説明が不足しています。前の行から次の主張へ移る論理の橋渡しが弱いです。`;
    case "不足":
      return `${currRole}を支えるための情報が${prevRole}に十分含まれていません。前提や条件の補足が必要です。`;
    case "未定義":
      return `${currRole}で使われている概念・対象・手法の一部が、それ以前の行で導入されていません。`;
    case "過剰":
      return `${currRole}の主張が${prevRole}に対して強すぎます。直前の内容からはそこまで言い切れません。`;
    default:
      return "";
  }
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

function summarizeNodeLines(node) {
  return node.lines
    .map((line, i) => `${LINE_NAMES[i]}: ${line.trim() || "未入力"}`)
    .join("\n");
}

function getAncestorContext(node, maxDepth = 2) {
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

function buildPrompt(node, prevIndex, currIndex, prevText, currText) {
  const pathText = getNodePathTitles(node).join(" > ");
  const blockSummary = summarizeNodeLines(node);
  const ancestorContext = getAncestorContext(node, 2);
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
    `判定対象の前の行の役割: ${LINE_NAMES[prevIndex]}`,
    `判定対象の次の行の役割: ${LINE_NAMES[currIndex]}`,
    `判定対象の前の文: ${prevText}`,
    `判定対象の次の文: ${currText}`,
    "",
    "上位文脈と現在ブロック全体を踏まえて、前の行から次の行への研究ロジック上の接続を判定する。",
    "表面的な話題類似ではなく、論点・目的・問題設定・手法・結果・考察の整合性を見る。",
    "使用可能なラベルは次の5つのみ。",
    "問題なし: 自然につながっている。",
    "飛躍: 中間説明がなく、論理の橋渡しが弱い。",
    "不足: 次の行を支える前提情報が足りない。",
    "未定義: 次の行で使う概念や対象が未導入である。",
    "過剰: 次の行の主張が強すぎる。",
    "",
    "この接続に最も適切なラベルを1つ選ぶ。"
  ]
    .filter(Boolean)
    .join("\n");
}

function getHeuristicFeatures(prevText, currText) {
  const prev = prevText || "";
  const curr = currText || "";

  const feature = {
    overlap: 0,
    hasMethodWord: false,
    hasResultWord: false,
    hasDiscussionWord: false,
    hasAbstractRef: false,
    introducesNewTerms: false,
    strongClaim: false,
  };

  const splitWords = (text) =>
    text
      .split(/[\s、。，．。・()（）「」『』【】\[\],.!?！？:：;；/]+/)
      .map((s) => s.trim())
      .filter((s) => s.length >= 2);

  const prevWords = splitWords(prev);
  const currWords = splitWords(curr);

  const prevSet = new Set(prevWords);
  const currSet = new Set(currWords);

  let overlap = 0;
  for (const w of currSet) {
    if (prevSet.has(w)) overlap++;
  }
  feature.overlap = overlap;

  const methodWords = ["提案", "手法", "モデル", "学習", "推定", "分類", "生成", "FiLM", "手順", "設計"];
  const resultWords = ["結果", "性能", "向上", "改善", "精度", "有意", "実験", "評価", "達成"];
  const discussionWords = ["有効", "示唆", "考えられる", "可能性", "解釈", "意味", "有用"];
  const abstractRefs = ["これ", "それ", "この", "その", "両者", "提案手法", "本手法", "本研究"];
  const strongClaims = ["証明", "完全", "必ず", "明らか", "万能", "決定的", "全て", "十分である", "有効である"];

  feature.hasMethodWord = methodWords.some((w) => curr.includes(w));
  feature.hasResultWord = resultWords.some((w) => curr.includes(w));
  feature.hasDiscussionWord = discussionWords.some((w) => curr.includes(w));
  feature.hasAbstractRef = abstractRefs.some((w) => curr.includes(w));

  const newTerms = currWords.filter((w) => !prevSet.has(w) && w.length >= 3);
  feature.introducesNewTerms = newTerms.length >= 2;

  feature.strongClaim = strongClaims.some((w) => curr.includes(w));

  return feature;
}

function heuristicClassify(prevText, currText, currIndex) {
  const hasPrev = !!prevText.trim();
  const hasCurr = !!currText.trim();

  if (!hasPrev || !hasCurr) {
    return {
      label: "問題なし",
      scoreText: "空欄を含むため問題なし扱い",
    };
  }

  const f = getHeuristicFeatures(prevText, currText);
  let label = "問題なし";

  if (currIndex === 1) {
    if (f.introducesNewTerms && !f.hasAbstractRef) label = "未定義";
    else if (f.strongClaim) label = "過剰";
    else if (f.overlap === 0) label = "飛躍";
  } else if (currIndex === 2) {
    if (f.introducesNewTerms && f.hasMethodWord && f.overlap === 0) label = "未定義";
    else if (f.overlap === 0) label = "飛躍";
    else if (!f.hasMethodWord) label = "不足";
  } else if (currIndex === 3) {
    if (f.strongClaim) label = "過剰";
    else if (!f.hasResultWord) label = "不足";
    else if (f.overlap === 0) label = "飛躍";
  } else if (currIndex === 4) {
    if (f.strongClaim) label = "過剰";
    else if (f.introducesNewTerms && !f.hasAbstractRef) label = "未定義";
    else if (!f.hasDiscussionWord && f.overlap === 0) label = "飛躍";
  }

  return {
    label,
    scoreText: `fallback: overlap=${f.overlap}, method=${f.hasMethodWord}, result=${f.hasResultWord}, discussion=${f.hasDiscussionWord}, newTerms=${f.introducesNewTerms}, strongClaim=${f.strongClaim}`,
  };
}

async function classifyPair(node, prevIndex, currIndex, prevText, currText) {
  const relation = `${LINE_NAMES[prevIndex]} → ${LINE_NAMES[currIndex]}`;
  const hasPrev = !!prevText.trim();
  const hasCurr = !!currText.trim();

  if (!hasPrev || !hasCurr) {
    return {
      label: "問題なし",
      reason: lineSpecificReason("問題なし", LINE_NAMES[prevIndex], LINE_NAMES[currIndex], prevText, currText),
      relation,
      scoreText: "空欄を含むため問題なし扱い",
      analyzed: true,
    };
  }

  if (!classifier) {
    const fallback = heuristicClassify(prevText, currText, currIndex);
    return {
      label: fallback.label,
      reason: lineSpecificReason(fallback.label, LINE_NAMES[prevIndex], LINE_NAMES[currIndex], prevText, currText),
      relation,
      scoreText: fallback.scoreText,
      analyzed: true,
    };
  }

  const sequence = buildPrompt(node, prevIndex, currIndex, prevText, currText);

  try {
    const output = await classifier(sequence, LABELS, {
      multi_label: false,
      hypothesis_template: "この接続の判定は {}。",
    });

    const labels = Array.isArray(output.labels) ? output.labels : [];
    const scores = Array.isArray(output.scores) ? output.scores : [];
    const topLabel = labels[0] || "飛躍";
    const topScore = typeof scores[0] === "number" ? scores[0] : null;
    const safeLabel = LABELS.includes(topLabel) ? topLabel : "飛躍";

    return {
      label: safeLabel,
      reason: lineSpecificReason(safeLabel, LINE_NAMES[prevIndex], LINE_NAMES[currIndex], prevText, currText),
      relation,
      scoreText: topScore === null ? "model score unavailable" : `model score=${topScore.toFixed(3)}`,
      analyzed: true,
    };
  } catch (e) {
    const fallback = heuristicClassify(prevText, currText, currIndex);
    return {
      label: fallback.label,
      reason: lineSpecificReason(fallback.label, LINE_NAMES[prevIndex], LINE_NAMES[currIndex], prevText, currText),
      relation,
      scoreText: `fallback after error: ${fallback.scoreText}`,
      analyzed: true,
    };
  }
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
      <div class="reason-body">${escapeHtml(item.relation)}\n${escapeHtml(item.reason)}</div>
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
            <button class="secondary small" data-action="analyze-node" data-node-id="${node.id}" ${classifier ? "" : "disabled"}>
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
  els.analyzeRootBtn.disabled = !classifier;
  els.analyzeAllBtn.disabled = !classifier;
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
    const result = await classifyPair(node, i - 1, i, node.lines[i - 1], node.lines[i]);
    node.results[i] = result;
  }
}

async function analyzeSingleNode(nodeId) {
  if (!classifier || isChecking) return;
  const node = findNodeById(tree, nodeId);
  if (!node) return;

  isChecking = true;
  els.analyzeAllBtn.disabled = true;
  els.analyzeRootBtn.disabled = true;
  els.loadBtn.disabled = true;
  els.copyMdBtn.disabled = true;

  setProgress(`ブロック ${node.title} を判定中...`, null);
  await analyzeNode(node);
  render();

  setProgress("ブロック判定完了", 100);
  els.analyzeAllBtn.disabled = false;
  els.analyzeRootBtn.disabled = false;
  els.loadBtn.disabled = false;
  els.copyMdBtn.disabled = false;
  isChecking = false;
}

async function analyzeAllNodes() {
  if (!classifier || isChecking) return;

  isChecking = true;
  els.analyzeAllBtn.disabled = true;
  els.analyzeRootBtn.disabled = true;
  els.loadBtn.disabled = true;
  els.copyMdBtn.disabled = true;

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
  isChecking = false;
}

async function loadModel() {
  if (classifier || isLoading) return;

  isLoading = true;
  els.loadBtn.disabled = true;
  els.analyzeRootBtn.disabled = true;
  els.analyzeAllBtn.disabled = true;
  els.copyMdBtn.disabled = true;

  selectedDevice = supportsWebGPU() ? "webgpu" : "wasm";
  setDeviceState("loading", selectedDevice === "webgpu" ? "WebGPU を試行中" : "CPU (WASM) を使用");
  setModelState("loading", "モデル読込中");
  setProgress("モデルを読み込んでいます...", 0);

  try {
    classifier = await pipeline("zero-shot-classification", MODEL_ID, {
      device: selectedDevice,
      dtype: selectedDevice === "webgpu" ? "fp16" : "q8",
      progress_callback: (progress) => {
        if (!progress) return;
        const p = progress.progress ?? 0;
        const status = progress.status ?? "loading";
        setProgress(`読込中: ${status}`, p * 100);
      },
    });

    setModelState("ready", "モデル読込完了");
    setDeviceState("ready", selectedDevice === "webgpu" ? "WebGPU で実行" : "CPU (WASM) で実行");
    setProgress("準備完了", 100);
    els.analyzeRootBtn.disabled = false;
    els.analyzeAllBtn.disabled = false;
  } catch (e) {
    try {
      selectedDevice = "wasm";
      setDeviceState("loading", "CPU (WASM) に切替中");
      setProgress("CPU モードで再読込中...", 10);

      classifier = await pipeline("zero-shot-classification", MODEL_ID, {
        device: "wasm",
        dtype: "q8",
        progress_callback: (progress) => {
          if (!progress) return;
          const p = progress.progress ?? 0;
          const status = progress.status ?? "loading";
          setProgress(`読込中: ${status}`, p * 100);
        },
      });

      setModelState("ready", "モデル読込完了");
      setDeviceState("ready", "CPU (WASM) で実行");
      setProgress("準備完了", 100);
      els.analyzeRootBtn.disabled = false;
      els.analyzeAllBtn.disabled = false;
    } catch (err) {
      console.error(err);
      setModelState("error", "モデル読込失敗");
      setDeviceState("error", "推論デバイス初期化失敗");
      setProgress("読込に失敗しました", 0);
      alert("モデルの読み込みに失敗しました。通信環境かブラウザ設定を確認してください。");
    }
  } finally {
    isLoading = false;
    els.loadBtn.disabled = false;
    els.copyMdBtn.disabled = false;
    render();
  }
}

function resetAllParentLinks(node, parent = null, parentLineIndex = null) {
  node.parent = parent;
  node.parentLineIndex = parentLineIndex;
  node.children.forEach((child, i) => {
    if (child) resetAllParentLinks(child, node, i);
  });
}

function clearAll() {
  tree = createNode(0, "ルート");
  resetNodeResults(tree);
  render();
  setProgress(classifier ? "準備完了" : "待機中", classifier ? 100 : 0);
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
    alert("クリップボードへの保存に失敗しました。HTTPSで開いているか、ブラウザの権限設定を確認してください。");
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

if (supportsWebGPU()) {
  setDeviceState("ready", "WebGPU 利用可能");
} else {
  setDeviceState("ready", "WebGPU 非対応 / CPU (WASM) 想定");
}
setModelState("ready", "未ロード");
setProgress("待機中", 0);

resetAllParentLinks(tree);
render();
