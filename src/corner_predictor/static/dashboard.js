(() => {
  const els = {
    homeTeam: document.getElementById("home-team"),
    awayTeam: document.getElementById("away-team"),
    threshold: document.getElementById("threshold"),
    seed: document.getElementById("seed"),
    startBtn: document.getElementById("start-btn"),
    status: document.getElementById("status"),
    minute: document.getElementById("minute"),
    score: document.getElementById("score"),
    corners: document.getElementById("corners"),
    possession: document.getElementById("possession"),
    currentThreshold: document.getElementById("current-threshold"),
    probOver: document.getElementById("prob-over"),
    expectedTotal: document.getElementById("expected-total"),
    matchListBody: document.getElementById("match-list-body"),
  };

  let socket = null;

  const probChart = new Chart(document.getElementById("prob-chart"), {
    type: "line",
    data: { datasets: [{ label: "P(总角球 > 阈值)", data: [], borderColor: "#2f6feb", tension: 0.2, pointRadius: 0 }] },
    options: {
      parsing: false,
      scales: {
        x: { type: "linear", title: { display: true, text: "比赛分钟" }, min: 0, max: 90 },
        y: { min: 0, max: 1, title: { display: true, text: "概率" } },
      },
      animation: false,
    },
  });

  const pmfChart = new Chart(document.getElementById("pmf-chart"), {
    type: "bar",
    data: { labels: [], datasets: [{ label: "P(总角球数 = k)", data: [], backgroundColor: "#7aa2f7" }] },
    options: {
      scales: {
        x: { title: { display: true, text: "全场角球数" } },
        y: { min: 0, title: { display: true, text: "概率" } },
      },
      animation: false,
    },
  });

  function resetCharts() {
    probChart.data.datasets[0].data = [];
    probChart.update();
    pmfChart.data.labels = [];
    pmfChart.data.datasets[0].data = [];
    pmfChart.update();
  }

  function applyUpdate(update) {
    els.minute.textContent = update.minute.toFixed(0);
    els.score.textContent = `${update.score_home} : ${update.score_away}`;
    els.corners.textContent = `${update.corners_home} / ${update.corners_away}`;
    els.possession.textContent = `${update.possession_home_pct.toFixed(0)}%`;
    els.currentThreshold.textContent = update.threshold;
    els.probOver.textContent = `${(update.prob_over * 100).toFixed(1)}%`;
    els.expectedTotal.textContent = update.expected_total_corners.toFixed(1);

    probChart.data.datasets[0].data.push({ x: update.minute, y: update.prob_over });
    probChart.update();

    pmfChart.data.labels = update.pmf.map((pair) => pair[0]);
    pmfChart.data.datasets[0].data = update.pmf.map((pair) => pair[1]);
    pmfChart.update();

    if (!update.is_live) {
      els.status.textContent = "比赛结束";
      refreshMatchList();
    }
  }

  function connect(matchId) {
    if (socket) {
      socket.close();
    }
    resetCharts();
    els.status.textContent = `已连接：${matchId}`;

    const protocol = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${protocol}://${location.host}/ws/matches/${matchId}`);
    socket.onmessage = (event) => applyUpdate(JSON.parse(event.data));
    socket.onclose = () => {
      if (els.status.textContent.startsWith("已连接")) {
        els.status.textContent = "连接已断开";
      }
    };
    socket.onerror = () => {
      els.status.textContent = "连接出错";
    };
  }

  async function startMatch() {
    els.status.textContent = "正在启动比赛...";
    const response = await fetch("/api/matches/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        home_team: els.homeTeam.value || "Home FC",
        away_team: els.awayTeam.value || "Away United",
        threshold: parseFloat(els.threshold.value) || 9.5,
        seed: els.seed.value ? parseInt(els.seed.value, 10) : null,
      }),
    });
    const data = await response.json();
    connect(data.match_id);
    refreshMatchList();
  }

  async function loadFinishedMatchHistory(matchId) {
    resetCharts();
    els.status.textContent = `查看历史：${matchId}`;
    const response = await fetch(`/api/matches/${matchId}/history`);
    const ticks = await response.json();
    probChart.data.datasets[0].data = ticks.map((t) => ({ x: t.minute, y: t.prob_over }));
    probChart.update();
  }

  async function refreshMatchList() {
    const response = await fetch("/api/matches");
    const matches = await response.json();
    els.matchListBody.innerHTML = "";
    for (const m of matches) {
      const row = document.createElement("tr");
      const corners =
        m.final_corners_home != null ? `${m.final_corners_home} - ${m.final_corners_away}` : "--";
      row.innerHTML = `
        <td>${m.home_team} vs ${m.away_team}</td>
        <td>${m.is_running ? "进行中" : m.status}</td>
        <td>${corners}</td>
        <td><button class="link-btn" data-id="${m.id}" data-live="${m.is_running}">查看</button></td>
      `;
      els.matchListBody.appendChild(row);
    }
    els.matchListBody.querySelectorAll("button.link-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.dataset.live === "true") {
          connect(btn.dataset.id);
        } else {
          loadFinishedMatchHistory(btn.dataset.id);
        }
      });
    });
  }

  els.startBtn.addEventListener("click", startMatch);
  refreshMatchList();
})();
