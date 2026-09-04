export function fmt(value) {
  if (value == null) return "—";
  if (typeof value === "string") return value;
  if (Number.isNaN(value)) return "—";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    return Math.abs(value) >= 0.01 ? value.toFixed(4) : value.toExponential(2);
  }
  return String(value);
}

export function icirValue(ic, icir) {
  if (icir?.scalars?.icir != null) return icir.scalars.icir;
  const scalars = ic?.scalars;
  if (scalars?.mean != null && scalars.std) return scalars.mean / scalars.std;
  return null;
}

export function nestedScalar(item, path) {
  let current = item.scalars || {};
  for (const part of path.split(".")) {
    current = current && current[part];
  }
  return current;
}

export function seriesXY(series) {
  return { x: series?.dates || [], y: series?.values || [] };
}
