#!/usr/bin/env python3
import os, re, csv, argparse
from pathlib import Path

PHASE_RE   = re.compile(r"^\s*Profiling phase:\s*(.+?)\s*$")
LAT_RE     = re.compile(r"^\s*-\s*Latency:\s*([0-9.eE+-]+)\s*$")
POW_RE     = re.compile(r"^\s*-\s*Active Power:\s*([0-9.eE+-]+)\s*W\s*$")
TOTAL_RE   = re.compile(r"^\s*Total Latency.*?:\s*([0-9.eE+-]+)\s*$")
PARAMS_RE  = re.compile(
    r"B\s*=\s*(\d+).*?"
    r"P\s*=\s*(\d+).*?"
    r"O\s*=\s*(\d+).*?"
    r"D\s*=\s*(\d+).*?"
    r"H\s*=\s*(\d+).*?"
    r"E\s*=\s*(\d+).*?"
    r"I\s*=\s*(\d+)",
    re.DOTALL
)

def parse_file(path: Path):
    """Parse one results file -> (row dict, phases dict)"""
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Params (optional)
    all_text = "\n".join(text)
    params = {"B": None, "P": None, "O": None, "D": None, "H": None, "E": None, "I": None}
    m = PARAMS_RE.search(all_text)
    if m:
        keys = ["B","P","O","D","H","E","I"]
        for k, v in zip(keys, m.groups()):
            params[k] = int(v)

    # Phases
    phases = {}  # name -> {"latency_s": float, "power_w": float, "energy_j": float}
    i = 0
    current_phase = None
    while i < len(text):
        line = text[i]
        m_phase = PHASE_RE.match(line)
        if m_phase:
            current_phase = m_phase.group(1).strip()
            # Expect following two lines to contain latency and power (order-insensitive)
            lat = pow_ = None
            j = i + 1
            # look ahead a few lines to be resilient
            for k in range(j, min(j + 6, len(text))):
                m_lat = LAT_RE.match(text[k])
                if m_lat: lat = float(m_lat.group(1))
                m_pow = POW_RE.match(text[k])
                if m_pow: pow_ = float(m_pow.group(1))
            if lat is not None and pow_ is not None:
                phases[current_phase] = {
                    "latency_s": lat,
                    "power_w": pow_,
                    "energy_j": lat * pow_,
                }
            else:
                # record partials if present
                phases[current_phase] = {
                    "latency_s": lat,
                    "power_w": pow_,
                    "energy_j": (lat * pow_) if (lat is not None and pow_ is not None) else None,
                }
        i += 1

    # Total latency (optional)
    total_latency = None
    for line in text[::-1]:
        m_tot = TOTAL_RE.match(line)
        if m_tot:
            total_latency = float(m_tot.group(1))
            break

    # Total energy = sum of phase energies when available
    total_energy = None
    energies = [v["energy_j"] for v in phases.values() if v.get("energy_j") is not None]
    if energies:
        total_energy = sum(energies)

    row = {
        "label": path.stem,
        **params,
        "total_latency_s": total_latency,
        "total_energy_j": total_energy,
    }
    return row, phases

def main():
    ap = argparse.ArgumentParser(description="Combine per-phase benchmark printouts into a CSV.")
    ap.add_argument("folder", help="Folder containing text result files.")
    ap.add_argument("-o", "--out", default="combined_results.csv", help="Output CSV path.")
    ap.add_argument("--ext", default=".txt", help="File extension filter (default: .txt). Use '' for all files.")
    args = ap.parse_args()

    folder = Path(args.folder)
    files = sorted([p for p in folder.iterdir() if p.is_file() and (args.ext == "" or p.suffix == args.ext)])
    if not files:
        raise SystemExit(f"No files found in {folder} with ext '{args.ext}'")

    rows = []
    all_phase_names = set()

    parsed = []
    for f in files:
        row, phases = parse_file(f)
        parsed.append((row, phases))
        rows.append(row)
        all_phase_names.update(phases.keys())

    # Build header: label, params, totals, then per-phase triplets (sorted by name)
    phase_cols = []
    for name in sorted(all_phase_names):
        slug = name.replace(",", "_").replace(" ", "_")
        phase_cols += [
            f"{slug}_latency_s",
            f"{slug}_power_w",
            f"{slug}_energy_j",
        ]

    base_cols = ["label", "B", "P", "O", "D", "H", "E", "I", "total_latency_s", "total_energy_j"]
    header = base_cols + phase_cols

    # Write CSV
    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=header)
        writer.writeheader()
        for (row, phases) in parsed:
            out_row = dict(row)  # shallow copy
            for name in all_phase_names:
                slug = name.replace(",", "_").replace(" ", "_")
                vals = phases.get(name, {})
                out_row.setdefault(f"{slug}_latency_s", vals.get("latency_s"))
                out_row.setdefault(f"{slug}_power_w",   vals.get("power_w"))
                out_row.setdefault(f"{slug}_energy_j",  vals.get("energy_j"))
            writer.writerow(out_row)

    print(f"Wrote {out_path} with {len(rows)} rows and {len(header)} columns.")

if __name__ == "__main__":
    main()
