/* SemIf desk lab — talks to the local arc-demo server. No external requests. */
"use strict";

const $ = (selector) => document.querySelector(selector);
const MIN_OPTIONS = 2;
const MAX_OPTIONS = 16;
const MAX_CRITERIA = 16;

const presets = {
  account: {
    state: "A customer says a password reset succeeded, but every login attempt still returns ‘account locked’. Two unlock emails were requested and neither arrived.",
    criteria: [
      {
        question: "Which queue should handle this request?",
        options: ["Account access support", "Billing support", "Close as resolved"],
      },
    ],
  },
  email: {
    state: "An email claims to be from the payroll team and says the recipient’s salary payment will be suspended today. It comes from payroll-review@outlook.com and links to a non-company sign-in page asking for a password and verification code.",
    criteria: [
      {
        question: "How should this email be classified?",
        options: ["Legitimate", "Spam", "Phishing"],
      },
    ],
  },
  trio: {
    state: "A customer says a password reset succeeded, but every login attempt still returns ‘account locked’. Two unlock emails were requested and neither arrived.",
    criteria: [
      { question: "Which queue should handle this request?", options: ["Account access support", "Billing support"] },
      { question: "Should this ticket be escalated right now?", options: ["Yes, escalate", "No, keep in queue"] },
      { question: "Is the reported symptom fully described?", options: ["Yes, it is complete", "No, details are missing"] },
    ],
  },
};

function criteriaRoot() {
  return $("#criteria");
}

function addCriterion(question = "", options = ["", "", ""]) {
  const root = criteriaRoot();
  if (root.children.length >= MAX_CRITERIA) return;
  const index = root.children.length;
  const box = document.createElement("div");
  box.className = "criterion";
  box.innerHTML = `
    <div class="criterion-head"><small>criterion ${index + 1}</small><button type="button" class="remove-criterion">remove</button></div>
    <input class="criterion-question" placeholder="Question" value="" />
    <fieldset><legend>Allowed options</legend><div class="option-list"></div>
      <div class="option-actions"><button type="button" class="option-remove">− option</button><span class="option-count"></span><button type="button" class="option-add">+ option</button></div>
    </fieldset>`;
  box.querySelector(".criterion-question").value = question;
  const list = box.querySelector(".option-list");
  options.forEach((text) => list.appendChild(makeOptionRow(text)));
  box.querySelector(".option-add").addEventListener("click", () => {
    if (list.children.length < MAX_OPTIONS) list.appendChild(makeOptionRow(""));
    refreshCounts(box);
  });
  box.querySelector(".option-remove").addEventListener("click", () => {
    if (list.children.length > MIN_OPTIONS) list.lastChild.remove();
    refreshCounts(box);
  });
  box.querySelector(".remove-criterion").addEventListener("click", () => {
    box.remove();
    relabel();
  });
  root.appendChild(box);
  refreshCounts(box);
  relabel();
}

function makeOptionRow(text) {
  const row = document.createElement("label");
  row.innerHTML = `<b></b><input class="option" value="" />`;
  row.querySelector("input").value = text;
  return row;
}

function refreshCounts(box) {
  const list = box.querySelector(".option-list");
  box.querySelector(".option-count").textContent = `${list.children.length} / ${MAX_OPTIONS}`;
  Array.from(list.children).forEach((row, index) => {
    row.querySelector("b").textContent = String.fromCharCode(65 + index);
  });
}

function relabel() {
  Array.from(criteriaRoot().children).forEach((box, index) => {
    box.querySelector(".criterion-head small").textContent = `criterion ${index + 1}`;
  });
  $("#criterion-count").textContent = `${criteriaRoot().children.length} / ${MAX_CRITERIA}`;
}

function collect() {
  const state = $("#state").value.trim();
  const criteria = Array.from(criteriaRoot().children).map((box) => ({
    question: box.querySelector(".criterion-question").value.trim(),
    options: Array.from(box.querySelectorAll(".option")).map((input) => input.value.trim()),
  }));
  return { state, criteria, modes: ["shared", "fresh"] };
}

function validate(payload) {
  if (!payload.state) return "the state text is empty";
  if (!payload.criteria.length) return "add at least one criterion";
  for (const [index, criterion] of payload.criteria.entries()) {
    if (!criterion.question) return `criterion ${index + 1} has no question`;
    const filled = criterion.options.filter(Boolean);
    if (filled.length !== criterion.options.length) return `criterion ${index + 1} has an empty option`;
    if (filled.length < MIN_OPTIONS) return `criterion ${index + 1} needs at least ${MIN_OPTIONS} options`;
  }
  return null;
}

function setSupport(kind, text) {
  const node = $("#support .support");
  node.className = `support ${kind}`;
  node.textContent = text;
}

async function checkHealth() {
  try {
    const reply = await fetch("/api/health");
    const data = await reply.json();
    if (!reply.ok || !data.ready) throw new Error(data.error || "not ready");
    setSupport("ok", `server ready · ${data.model} on ${data.attention} attention`);
    const cells = $("#health-row").children;
    cells[0].textContent = data.model;
    cells[1].textContent = String(data.revision).slice(0, 12) + "…";
    cells[2].textContent = data.dtype;
    cells[3].textContent = data.attention;
    cells[4].textContent = data.torch;
    $("#selected-model").textContent = shortName(data.model);
    $("#model-size").textContent = `${shortName(data.model)} · bf16`;
  } catch (error) {
    setSupport("error", `server not reachable: ${error.message}`);
  }
}

function shortName(source) {
  return String(source || "").split("/").pop();
}

function renderLane(outputNode, results, emptyText) {
  outputNode.classList.remove("empty");
  outputNode.replaceChildren();
  if (!results) {
    outputNode.classList.add("empty");
    outputNode.textContent = emptyText;
    return;
  }
  for (const result of results) {
    const block = document.createElement("div");
    block.className = "criterion-result";
    const title = document.createElement("small");
    title.textContent = result.question || result.id;
    block.appendChild(title);
    for (const [index, probability] of result.probabilities.entries()) {
      const row = document.createElement("div");
      row.className = "choice";
      const label = document.createElement("div");
      label.className = "choice-label";
      label.innerHTML = `<b></b><small></small>`;
      label.querySelector("b").textContent = String.fromCharCode(65 + index);
      label.querySelector("small").textContent = result.option_descriptions[index];
      const bar = document.createElement("div");
      bar.className = "bar";
      bar.innerHTML = "<i></i>";
      bar.querySelector("i").style.width = `${(probability * 100).toFixed(1)}%`;
      const value = document.createElement("em");
      value.textContent = `${(probability * 100).toFixed(1)}%`;
      row.append(label, bar, value);
      block.appendChild(row);
    }
    outputNode.appendChild(block);
  }
}

function seconds(value) {
  return value == null ? "—" : `${value.toFixed(2)} s`;
}

async function run() {
  const payload = collect();
  const problem = validate(payload);
  if (problem) {
    setSupport("error", problem);
    return;
  }
  const button = $("#run");
  button.disabled = true;
  setSupport("", "scoring ...");
  $("#shared-output").textContent = "waiting for the server";
  $("#fresh-output").textContent = "waiting for the server";
  try {
    const reply = await fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await reply.json();
    if (!reply.ok) throw new Error(data.error || `HTTP ${reply.status}`);
    renderLane($("#shared-output"), data.shared.results, null);
    renderLane($("#fresh-output"), data.fresh.results, null);
    const shared = data.shared.timing;
    $("#shared-total").textContent = seconds(shared.total_seconds);
    $("#shared-prefill").textContent = seconds(shared.prefill_seconds);
    $("#shared-suffix").textContent = seconds(shared.suffix_forward_seconds);
    $("#fresh-total").textContent = seconds(data.fresh.timing.total_seconds);
    $("#fresh-per").textContent = seconds(data.fresh.timing.total_seconds / payload.criteria.length);
    $("#fresh-passes").textContent = String(payload.criteria.length);
    const ratio = data.fresh.timing.total_seconds / shared.total_seconds;
    $("#ratio").textContent = `${ratio.toFixed(2)}x`;
    setSupport("ok", "done");
  } catch (error) {
    setSupport("error", `run failed: ${error.message}`);
  } finally {
    button.disabled = false;
  }
}

document.querySelectorAll("[data-preset]").forEach((button) => {
  button.addEventListener("click", () => {
    const preset = presets[button.dataset.preset];
    if (!preset) return;
    $("#state").value = preset.state;
    criteriaRoot().replaceChildren();
    preset.criteria.forEach((criterion) => addCriterion(criterion.question, criterion.options));
  });
});

$("#add-criterion").addEventListener("click", () => addCriterion());
$("#remove-criterion").addEventListener("click", () => {
  const root = criteriaRoot();
  if (root.children.length > 1) root.lastChild.remove();
  relabel();
});
$("#run").addEventListener("click", run);
$("#refresh").addEventListener("click", checkHealth);

addCriterion();
checkHealth();
