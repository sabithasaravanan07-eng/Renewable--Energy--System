document.addEventListener('DOMContentLoaded', function () {
  const button = document.getElementById('loadYearBtn');
  const input = document.getElementById('yearInput');
  const dashboardResult = document.getElementById('dashboardResult');

  if (!button || !input || !dashboardResult) return;

  async function loadYearData() {
    const year = input.value || 2025;
    const response = await fetch(`/api/year?year=${year}`);
    const data = await response.json();

    dashboardResult.innerHTML = `
      <section class="panel summary">
        <h2>Analysis Summary</h2>
        <div class="grid">
          <div><strong>Year:</strong> ${data.year}</div>
          <div><strong>Status:</strong> ${data.status}</div>
          <div><strong>Wind Speed:</strong> ${data.wind_speed === null ? 'N/A' : `${data.wind_speed.toFixed(2)} m/s`}</div>
          <div><strong>Predicted Wind Speed:</strong> ${data.predicted_wind_speed === null ? 'N/A' : `${data.predicted_wind_speed.toFixed(2)} m/s`}</div>
          <div><strong>Wind Power:</strong> ${data.wind_power.toFixed(2)} W</div>
          <div><strong>Energy Output:</strong> ${data.energy_output.toFixed(2)} kWh</div>
          <div><strong>Selected Model:</strong> ${data.selected_model}</div>
          <div><strong>MAE:</strong> ${data.mae.toFixed(4)}</div>
          <div><strong>RMSE:</strong> ${data.rmse.toFixed(4)}</div>
          <div><strong>R²:</strong> ${data.r2.toFixed(4)}</div>
        </div>
      </section>
    `;
  }

  button.addEventListener('click', loadYearData);
  loadYearData();
});
