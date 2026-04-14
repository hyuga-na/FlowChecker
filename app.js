import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.0";

const MODEL_ID = "onnx-community/multilingual-MiniLMv2-L6-mnli-xnli-ONNX";
const LINE_NAMES = ["背景", "課題", "解法", "結果", "考察"];
const FLOW_LABELS = ["問題なし", "飛躍"];

let classifier = null;
let isLoading = false;
let isChecking = false;
let nodeCounter = 0;

const SCORE_UNKNOWN_THRESHOLD = 0.18;
const MIN_TEXT_LENGTH = 6;

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
    case "不明":
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
    case "不明":
      return "不明";
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

function buildFlowPrompt(node, prevIndex, currIndex, prevText, currText) {
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

function getRoleHints(text) {
  const t = text.trim();

  return {
    hasMethod:
      t.includes("提案") ||
      t.includes("手法") ||
      t.includes("FiLM") ||
      t.includes("用いる") ||
      t.includes("注入") ||
      t.includes("モデル") ||
      t.includes("設計") ||
      t.includes("分離する"),
    hasResult:
      t.includes("結果") ||
      t.includes("性能") ||
      t.includes("向上") ||
      t.includes("改善") ||
      t.includes("実験") ||
      t.includes("評価") ||
      t.includes("精度") ||
      t.includes("示した") ||
      t.includes("達成"),
    hasDiscussion:
      t.includes("有効") ||
      t.includes("可能性") ||
      t.includes("示唆") ||
      t.includes("考えられる") ||
      t.includes("解釈") ||
      t.includes("意味"),
    hasProblem:
      t.includes("課題") ||
      t.includes("問題") ||
      t.includes("必要") ||
      t.includes("困難") ||
      t.includes("不足") ||
      t.includes("未解決") ||
      t.includes("コスト") ||
      t.includes("高い"),
    hasBackground:
      t.includes("近年") ||
      t.includes("従来") ||
      t.includes("一般に") ||
      t.includes("知られている") ||
      t.includes("提案されている") ||
      t.includes("含まれている"),
  };
}

function ruleBasedPrecheck(prevIndex, currIndex, prevText, currText) {
  const prev = prevText.trim();
  const curr = currText.trim();

  if (!prev || !curr) {
    return {
      label: "問題なし",
      confidence: "high",
      why: "空欄を含む",
    };
  }

  if (prev.length < MIN_TEXT_LENGTH || curr.length < MIN_TEXT_LENGTH) {
    return {
      label: "不明",
      confidence: "low",
      why: "文が短すぎる",
    };
  }

  const h = getRoleHints(curr);

  if (prevIndex === 0 && currIndex === 1) {
    if (h.hasMethod || h.hasResult || h.hasDiscussion) {
      return {
        label: "飛躍",
        confidence: "high",
        why: "背景の次に課題ではなく別段階の内容が来ている",
      };
    }
    if (h.hasProblem) {
      return {
        label: "問題なし",
        confidence: "medium",
        why: "背景の次に課題らしい内容が来ている",
      };
    }
    return {
      label: "不明",
      confidence: "low",
      why: "課題らしさが弱い",
    };
  }

  if (prevIndex === 1 && currIndex === 2) {
    if (h.hasMethod) {
      return {
        label: "問題なし",
        confidence: "medium",
        why: "課題の次に解法らしい内容が来ている",
      };
    }
    if (h.hasResult || h.hasDiscussion) {
      return {
        label: "飛躍",
        confidence: "high",
        why: "課題の次に結果や考察へ飛んでいる",
      };
    }
    return {
      label: "不明",
      confidence: "low",
      why: "解法らしさが弱い",
    };
  }

  if (prevIndex === 2 && currIndex === 3) {
    if (h.hasResult) {
      return {
        label: "問題なし",
        confidence: "medium",
        why: "解法の次に結果らしい内容が来ている",
      };
    }
    if (h.hasDiscussion) {
      return {
        label: "飛躍",
        confidence: "high",
        why: "解法の次に考察へ飛んでいる",
      };
    }
    return {
      label: "不明",
      confidence: "low",
      why: "結果らしさが弱い",
    };
  }

  if (prevIndex === 3 && currIndex === 4) {
    if (h.hasDiscussion) {
      return {
        label: "問題なし",
        confidence: "medium",
        why: "結果の次に考察らしい内容が来ている",
      };
    }
    if (h.hasMethod || h.hasProblem) {
      return {
        label: "飛躍",
        confidence: "high",
        why: "結果の次に別段階へ戻っている",
      };
    }
    return {
      label: "不明",
      confidence: "low",
      why: "考察らしさが弱い",
    };
  }

  return {
    label: "不明",
    confidence: "low",
    why: "規則が当てはまらない",
  };
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

  if (label === "不明") {
    return `${prevRole}から${currRole}への接続を自信を持って判定できませんでした。${extra}`.trim();
  }

  return `${prevRole}から${currRole}への接続は自然です。${extra}`.trim();
}

async function runZeroShot(sequence) {
  const output = await classifier(sequence, FLOW_LABELS, {
    multi_label: false,
    hypothesis_template: "この接続は {}。",
  });

  const labels = Array.isArray(output.labels) ? output.labels : [];
  const scores = Array.isArray(output.scores) ? output.scores : [];

  const firstLabel = labels[0] || "問題なし";
  const firstScore = typeof scores[0] === "number" ? scores[0] : 0;
  const secondScore = typeof scores[1] === "number" ? scores[1] : 0;
  const margin = Math.abs(firstScore - secondScore);

  return {
    label: FLOW_LABELS.includes(firstLabel) ? firstLabel : "問題なし",
    topScore: firstScore,
    secondScore,
    margin,
  };
}

async function classifyFlow(node, prevIndex, currIndex, prevText, currText) {
  if (!prevText.trim() || !currText.trim()) {
    return {
      label: "問題なし",
      reason: buildReason(currIndex, "問題なし", prevText, currText),
      scoreText: "空欄を含むため問題なし扱い",
    };
  }

  const precheck = ruleBasedPrecheck(prevIndex, currIndex, prevText, currText);

  if (precheck.confidence === "high") {
    return {
      label: precheck.label,
      reason: buildReason(currIndex, precheck.label, prevText, currText, precheck.why),
      scoreText: `rule: ${precheck.why}`,
    };
  }

  if (!classifier) {
    return {
      label: precheck.label,
      reason: buildReason(currIndex, precheck.label, prevText, currText, precheck.why),
      scoreText: `rule fallback: ${precheck.why}`,
    };
  }

  try {
    const sequence = buildFlowPrompt(node, prevIndex, currIndex, prevText, currText);
    const result = await runZeroShot(sequence);

    if (result.margin < SCORE_UNKNOWN_THRESHOLD) {
      return {
        label: "不明",
        reason: buildReason(
          currIndex,
          "不明",
          prevText,
          currText,
          `分類スコア差が小さく、判定が不安定です。`
        ),
        scoreText: `model uncertain: top=${result.topScore.toFixed(3)}, margin=${result.margin.toFixed(3)}`,
      };
    }

    return {
      label: result.label,
      reason: buildReason(currIndex, result.label, prevText, currText),
      scoreText: `model: top=${result.topScore.toFixed(3)}, margin=${result.margin.toFixed(3)}`,
    };
  } catch (e) {
    return {
      label: precheck.label,
      reason: buildReason(currIndex, precheck.label, prevText, currText, precheck.why),
      scoreText: `model error / rule fallback: ${precheck.why}`,
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
    const flowResult = await classifyFlow(node, i - 1, i, node.lines[i - 1], node.lines[i]);
    node.results[i] = {
      label: flowResult.label,
      reason: flowResult.reason,
      scoreText: flowResult.scoreText,
      relation: `${LINE_NAMES[i - 1]} → ${LINE_NAMES[i]}`,
      analyzed: true,
    };
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

  setDeviceState("loading", "CPU (WASM) を使用");
  setModelState("loading", "モデル読込中");
  setProgress("軽量モデルを読み込んでいます...", 0);

  try {
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
  } finally {
    isLoading = false;
    els.loadBtn.disabled = false;
    els.copyMdBtn.disabled = false;
    render();
  }
}

function clearAll() {
  tree = createNode(0, "ルート");
  resetNodeResults(tree);
  render();
  setProgress(classifier ? "準備完了" : "待機中", classifier ? 100 : 0);
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

setDeviceState("ready", "CPU (WASM) 想定");
setModelState("ready", "未ロード");
setProgress("待機中", 0);

render();
