document.addEventListener("DOMContentLoaded", function () {
  // ---- Tab switching ----
  const tabButtons = document.querySelectorAll(".tab-btn");
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabButtons.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.target).classList.add("active");
    });
  });

  // ---- Wind history chart (selected year highlighted) ----
  const windCtx = document.getElementById("windChart");
  if (windCtx && window.WIND_CHART) {
    const labels = window.WIND_CHART.labels;
    const selectedYear = window.WIND_CHART.selected_year;
    const pointColors = labels.map((y) => (y === selectedYear ? "#f97316" : "#2563eb"));
    const pointRadii = labels.map((y) => (y === selectedYear ? 7 : 3));
    new Chart(windCtx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [{
          label: "Avg wind speed (m/s)",
          data: window.WIND_CHART.values,
          borderColor: "#2563eb",
          backgroundColor: "rgba(37,99,235,0.12)",
          tension: 0.3,
          fill: true,
          pointBackgroundColor: pointColors,
          pointBorderColor: pointColors,
          pointRadius: pointRadii,
        }],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.parsed.y} m/s${labels[ctx.dataIndex] === selectedYear ? " (selected)" : ""}`,
            },
          },
        },
        scales: { y: { title: { display: true, text: "m/s" } } },
      },
    });
  }

  // ---- Wind model comparison chart (RMSE bar chart) ----
  const windModelCtx = document.getElementById("windModelChart");
  if (windModelCtx && window.WIND_MODEL_CHART) {
    const labels = window.WIND_MODEL_CHART.labels;
    const best = window.WIND_MODEL_CHART.best;
    const barColors = labels.map((n) => (n === best ? "#16a34a" : "#93a5c9"));
    new Chart(windModelCtx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "RMSE (lower is better)",
          data: window.WIND_MODEL_CHART.rmse,
          backgroundColor: barColors,
          borderRadius: 6,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, title: { display: true, text: "RMSE" } } },
      },
    });
  }

  // ---- Solar forecast chart ----
  const solarCtx = document.getElementById("solarChart");
  if (solarCtx && window.SOLAR_CHART) {
    new Chart(solarCtx, {
      type: "line",
      data: {
        labels: window.SOLAR_CHART.labels,
        datasets: [
          {
            label: "Irradiance (W/m²)",
            data: window.SOLAR_CHART.irradiance,
            borderColor: "#ea8c00",
            backgroundColor: "rgba(234,140,0,0.12)",
            yAxisID: "y",
            tension: 0.3,
          },
          {
            label: "Output (kW/kWp)",
            data: window.SOLAR_CHART.output,
            borderColor: "#16a34a",
            backgroundColor: "rgba(22,163,74,0.12)",
            yAxisID: "y1",
            tension: 0.3,
          },
        ],
      },
      options: {
        responsive: true,
        interaction: { mode: "index", intersect: false },
        scales: {
          y: { type: "linear", position: "left", title: { display: true, text: "W/m²" } },
          y1: { type: "linear", position: "right", title: { display: true, text: "kW/kWp" }, grid: { drawOnChartArea: false } },
        },
      },
    });
  }
});
