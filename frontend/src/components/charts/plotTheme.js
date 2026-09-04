export const PLOT_COLORS = [
  "#5b8ff9",
  "#5ad8a6",
  "#f6bd16",
  "#e8684a",
  "#6dc8ec",
  "#9270ca",
  "#ff9d4d",
  "#269a99",
  "#ff99c3",
  "#bda29a",
];

export function plotLayout(title, extra = {}) {
  return {
    title: { text: title, font: { size: 14, color: "#e8eef7" } },
    paper_bgcolor: "#1a2332",
    plot_bgcolor: "#1a2332",
    font: { color: "#8b9bb4", size: 11 },
    margin: { t: 40, r: 24, b: 56, l: 52 },
    legend: { orientation: "h", y: -0.22 },
    xaxis: { gridcolor: "#2c3a4f" },
    yaxis: { gridcolor: "#2c3a4f" },
    ...extra,
  };
}
