const state = {
  data: null,
  mode: "films",
  variant: "all",
  tab: "picks",
  search: "",
  resultCategory: "all",
  sortBy: "taste_score",
};

const categoryLabels = {
  safe_bet: "Safe Bet",
  wild_card: "Wild Card",
  skip: "Skip",
  below_threshold: "Skip",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function genresText(item) {
  const raw = item.genres || [];
  const genres = Array.isArray(raw) ? raw : String(raw).split("|");
  return genres.filter(Boolean).slice(0, 4).join(" · ");
}

function yearText(year) {
  const n = Number(year);
  return Number.isFinite(n) ? String(Math.trunc(n)) : "";
}

function normalizedCategory(item) {
  return item.category === "below_threshold" ? "skip" : item.category;
}

function currentModeData() {
  const modeData = state.data?.[state.mode] || {};
  if (modeData.variants) {
    return modeData.variants[state.variant] || modeData.variants.all || {};
  }
  return modeData;
}

function renderCard(item, nNeighbours) {
  const category = normalizedCategory(item);
  const year = yearText(item.year);
  const reasons = item.reasons || [];
  const shownReasons = reasons.slice(0, 3);
  const hiddenReasons = reasons.slice(3);
  const title = `${escapeHtml(item.title || "Unknown")}${year ? ` (${year})` : ""}`;
  return `
    <article class="card">
      <div class="card-head">
        <span class="badge ${category}">${categoryLabels[category] || category}</span>
        <span class="score">${Math.round(Number(item.taste_score || 0))}</span>
      </div>
      <div class="title">${title}</div>
      <div class="meta">${escapeHtml(genresText(item))}</div>
      <div class="meta">${Number(item.recommended_by || 0)}/${nNeighbours || 0} neighbours · avg ${Number(item.neighbour_score || 0).toFixed(1)}★</div>
      <ul class="reasons">
        ${shownReasons.map((r) => `<li>↳ ${escapeHtml(r)}</li>`).join("")}
      </ul>
      ${hiddenReasons.length ? `
        <details class="details">
          <summary>Full score breakdown</summary>
          <ul class="reasons">${hiddenReasons.map((r) => `<li>↳ ${escapeHtml(r)}</li>`).join("")}</ul>
        </details>
      ` : ""}
    </article>
  `;
}

function renderPicks(data) {
  const n = data.n_neighbours || 0;
  document.getElementById("runMeta").textContent = `${n} neighbours · ${data.run_time_s || 0}s`;
  const safe = data.safe_bets || [];
  const wild = data.wild_cards || [];
  const recent = data.recent_picks || [];
  document.getElementById("safeBets").innerHTML = safe.length
    ? safe.map((item) => renderCard(item, n)).join("")
    : `<div class="empty">No safe bets exported for this mode.</div>`;
  document.getElementById("wildCards").innerHTML = wild.length
    ? wild.map((item) => renderCard(item, n)).join("")
    : `<div class="empty">No wild cards exported for this mode.</div>`;
  const recentSection = document.getElementById("recentPicksSection");
  const recentEl = document.getElementById("recentPicks");
  if (recent.length) {
    recentSection.classList.remove("hidden");
    recentEl.innerHTML = recent.map((item) => renderCard(item, n)).join("");
  } else {
    recentSection.classList.add("hidden");
    recentEl.innerHTML = "";
  }
}

function candidateRows(data) {
  const term = state.search.trim().toLowerCase();
  return (data.all_scored || [])
    .filter((item) => state.resultCategory === "all" || normalizedCategory(item) === state.resultCategory)
    .filter((item) => {
      if (!term) return true;
      return `${item.title || ""} ${genresText(item)}`.toLowerCase().includes(term);
    })
    .sort((a, b) => Number(b[state.sortBy] || 0) - Number(a[state.sortBy] || 0));
}

function renderScatter(rows) {
  const el = document.getElementById("scatter");
  if (!rows.length) {
    el.innerHTML = `<div class="empty">No candidates to chart.</div>`;
    return;
  }
  const width = 900;
  const height = 320;
  const pad = 38;
  const xs = rows.map((r) => Number(r.diversity_score || r.weighted_score || 0));
  const ys = rows.map((r) => Number(r.taste_score || 0));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys, 0);
  const maxY = Math.max(...ys, 100);
  const scaleX = (x) => pad + ((x - minX) / Math.max(maxX - minX, 1)) * (width - pad * 2);
  const scaleY = (y) => height - pad - ((y - minY) / Math.max(maxY - minY, 1)) * (height - pad * 2);
  const color = (cat) => cat === "safe_bet" ? "#f5b642" : cat === "wild_card" ? "#26c6b8" : "#89909d";
  el.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Diversity score versus taste score">
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#505866"/>
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#505866"/>
      <text x="${pad}" y="22" fill="#a6adbb" font-size="13">Taste score</text>
      <text x="${width - 160}" y="${height - 10}" fill="#a6adbb" font-size="13">Diversity score</text>
      ${rows.map((r) => {
        const cat = normalizedCategory(r);
        return `<circle cx="${scaleX(Number(r.diversity_score || r.weighted_score || 0)).toFixed(1)}" cy="${scaleY(Number(r.taste_score || 0)).toFixed(1)}" r="5" fill="${color(cat)}" opacity="0.85"><title>${escapeHtml(r.title)} · ${Number(r.taste_score || 0).toFixed(1)}</title></circle>`;
      }).join("")}
    </svg>
  `;
}

function renderCandidates(data) {
  const rows = candidateRows(data);
  renderScatter(rows);
  if (!rows.length) {
    document.getElementById("candidateTable").innerHTML = `<div class="empty">No candidates match the current filters.</div>`;
    return;
  }
  document.getElementById("candidateTable").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Title</th>
          <th>Category</th>
          <th>Genres</th>
          <th class="numeric">Taste</th>
          <th class="numeric">Diversity</th>
          <th class="numeric">Neighbours</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map((item) => {
          const category = normalizedCategory(item);
          const year = yearText(item.year);
          return `
            <tr>
              <td>${escapeHtml(item.title || "Unknown")}${year ? ` <span class="muted">(${year})</span>` : ""}</td>
              <td><span class="badge ${category}">${categoryLabels[category] || category}</span></td>
              <td>${escapeHtml(genresText(item))}</td>
              <td class="numeric">${Number(item.taste_score || 0).toFixed(1)}</td>
              <td class="numeric">${Number(item.diversity_score || 0).toFixed(2)}</td>
              <td class="numeric">${Number(item.recommended_by || 0)}</td>
            </tr>
          `;
        }).join("")}
      </tbody>
    </table>
  `;
}

function renderTaste(data) {
  const profile = data.taste_profile || {};
  const rows = Object.entries(profile).sort((a, b) => b[1] - a[1]);
  if (!rows.length) {
    document.getElementById("tasteChart").innerHTML = `<div class="empty">No taste profile data exported for this mode.</div>`;
    return;
  }
  document.getElementById("tasteChart").innerHTML = rows.map(([genre, rating]) => {
    const pct = Math.max(0, Math.min(100, (Number(rating) / 10) * 100));
    return `
      <div class="bar-row">
        <div>${escapeHtml(genre)}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        <div class="numeric">${Number(rating).toFixed(1)}</div>
      </div>
    `;
  }).join("");
}

function render() {
  const data = currentModeData();
  const variants = state.data?.variants || [
    { key: "all", label: "All" },
    { key: "post_2020", label: "Post-2020" },
  ];
  const variantSelect = document.getElementById("variantSelect");
  variantSelect.innerHTML = variants
    .map((variant) => `<option value="${escapeHtml(variant.key)}">${escapeHtml(variant.label)}</option>`)
    .join("");
  variantSelect.value = state.variant;

  document.querySelectorAll(".mode-button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === state.mode);
  });
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === state.tab);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === state.tab);
  });

  const status = document.getElementById("status");
  if (data.status === "error") {
    status.textContent = `${state.mode} data is not ready: ${data.error}`;
    status.classList.remove("hidden");
  } else {
    status.classList.add("hidden");
  }

  document.getElementById("exportMeta").textContent = state.data
    ? `Exported ${new Date(state.data.exported_at).toLocaleString()} · ${data.label || "All"} · ${data.session_filters || "none"}`
    : "No export loaded.";

  renderPicks(data);
  renderCandidates(data);
  renderTaste(data);
}

function bindEvents() {
  document.querySelectorAll(".mode-button").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.mode = btn.dataset.mode;
      render();
    });
  });
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.tab = btn.dataset.tab;
      render();
    });
  });
  document.getElementById("search").addEventListener("input", (event) => {
    state.search = event.target.value;
    renderCandidates(currentModeData());
  });
  document.getElementById("categoryFilter").addEventListener("change", (event) => {
    state.resultCategory = event.target.value;
    renderCandidates(currentModeData());
  });
  document.getElementById("variantSelect").addEventListener("change", (event) => {
    state.variant = event.target.value;
    state.search = "";
    document.getElementById("search").value = "";
    renderCandidates(currentModeData());
    render();
  });
  document.getElementById("sortBy").addEventListener("change", (event) => {
    state.sortBy = event.target.value;
    renderCandidates(currentModeData());
  });
}

async function init() {
  bindEvents();
  try {
    const response = await fetch("data/recs.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
  } catch (error) {
    state.data = {
      exported_at: new Date().toISOString(),
      variants: [{ key: "all", label: "All" }, { key: "post_2020", label: "Post-2020" }],
      films: { variants: { all: { status: "error", label: "All", error: `Could not load data/recs.json: ${error.message}`, safe_bets: [], wild_cards: [], all_scored: [], taste_profile: {} } } },
      shows: { variants: { all: { status: "error", label: "All", error: `Could not load data/recs.json: ${error.message}`, safe_bets: [], wild_cards: [], all_scored: [], taste_profile: {} } } },
    };
  }
  render();
}

init();
