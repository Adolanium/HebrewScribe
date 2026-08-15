#!/usr/bin/env python3
"""Generate the transitive-dependency licence table for THIRD-PARTY-LICENSES.md.

Why this exists
---------------
The hand-maintained "Additional Bundled Python Dependencies" table drifts every
time a dependency gains a transitive dep. A 2026-08-15 audit found ~40 packages
present in the project's venvs that the table did not name. Hand-maintaining a
60-row table across three build targets is not a thing a solo dev should do.

This reads licence metadata from the packages that ACTUALLY ship and emits the
table. Point it at the frozen bundle for the authoritative answer, or at a venv
for a close approximation.

Usage
-----
    # Authoritative — run on the Windows build box AFTER PyInstaller freezes:
    python scripts/gen_third_party_licenses.py dist/HebrewScribe

    # Approximation from a venv:
    python scripts/gen_third_party_licenses.py .venv

    # Diff against what the doc currently claims:
    python scripts/gen_third_party_licenses.py .venv --check THIRD-PARTY-LICENSES.md

Exit codes: 0 clean, 1 undeclared licences found, 2 doc is missing packages.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

# Build-time only: present in the venv, never shipped to a user.
BUILD_ONLY = {
    "altgraph", "macholib", "pip", "setuptools-scm", "wheel",
    "pyinstaller", "pyinstaller-hooks-contrib", "pefile", "pywin32-ctypes",
    "pytest", "iniconfig", "pluggy", "detect-secrets", "hebrewscribe",
}

# Excluded in HebrewScribe.spec -> not in the frozen bundle even if installed.
SPEC_EXCLUDED = {
    "torchcodec", "matplotlib", "ipython", "jupyter", "tensorboard", "triton",
    # matplotlib's own dependency cone, pulled only by matplotlib:
    "contourpy", "cycler", "fonttools", "kiwisolver",
}

NORMALISE = re.compile(r"[-_.]+")


def canon(name: str) -> str:
    return NORMALISE.sub("-", name).strip().lower()


def read_licence(dist_info: pathlib.Path) -> tuple[str, str]:
    """Return (Name, licence-string) for one *.dist-info directory."""
    meta = dist_info / "METADATA"
    if not meta.exists():
        return ("", "")
    text = meta.read_text(encoding="utf-8", errors="replace")

    name_m = re.search(r"^Name: (.+)$", text, re.M)
    if not name_m:
        return ("", "")
    name = name_m.group(1).strip()

    # PEP 639 License-Expression wins; then a SHORT free-text License; then
    # the trove classifier; then a shipped licence file.
    expr = re.search(r"^License-Expression: (.+)$", text, re.M)
    if expr:
        return (name, expr.group(1).strip())

    free = re.search(r"^License: ([^\n]{1,60})$", text, re.M)
    if free and not free.group(1).strip().startswith("="):
        val = free.group(1).strip()
        # "UNKNOWN" is setuptools' legacy default for an unset field, NOT a
        # statement that the licence is unknown. primePy 1.3 carries
        # `License: UNKNOWN` alongside an explicit MIT classifier; preferring
        # the free-text field there reports a false undeclared.
        placeholder = val.upper() in {"UNKNOWN", "UNLICENSED", "NONE", "N/A", ""}
        # Reject prose masquerading as a licence name.
        if not placeholder and len(val.split()) <= 6 and "copyright" not in val.lower():
            return (name, val)

    classifiers = re.findall(r"^Classifier: License :: (.+)$", text, re.M)
    if classifiers:
        return (name, classifiers[0].split("::")[-1].strip())

    # A wheel can ship LICENSE without exposing it in JSON metadata — this is
    # exactly how pyannoteai-sdk read as "undeclared" until the file was opened.
    lic_dir = dist_info / "licenses"
    files = sorted(lic_dir.rglob("*")) if lic_dir.is_dir() else []
    files = [f for f in files if f.is_file()]
    if not files and re.search(r"^License-File: ", text, re.M):
        files = [p for p in dist_info.glob("LICENSE*") if p.is_file()]
    for f in files:
        head = f.read_text(encoding="utf-8", errors="replace")[:400]
        first = next((ln.strip() for ln in head.splitlines() if ln.strip()), "")
        if first:
            return (name, f"{first}  [from {f.name}]")

    return (name, "UNDECLARED")


def collect(root: pathlib.Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for di in root.rglob("*.dist-info"):
        if not di.is_dir():
            continue
        name, lic = read_licence(di)
        if not name:
            continue
        key = canon(name)
        if key in BUILD_ONLY or key in SPEC_EXCLUDED:
            continue
        found[key] = f"{name}\t{lic}"
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="frozen bundle dir (authoritative) or venv dir")
    ap.add_argument("--check", metavar="DOC",
                    help="compare against THIRD-PARTY-LICENSES.md and report gaps")
    args = ap.parse_args()

    root = pathlib.Path(args.root).expanduser()
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        return 2

    found = collect(root)
    if not found:
        print(f"error: no *.dist-info found under {root} — wrong directory?",
              file=sys.stderr)
        return 2

    rows = [tuple(v.split("\t", 1)) for v in found.values()]
    rows.sort(key=lambda r: r[0].lower())

    undeclared = [n for n, lic in rows if lic == "UNDECLARED"]

    print(f"# {len(rows)} shipped packages found under {root}\n")
    print("| Package | License |")
    print("|---|---|")
    for name, lic in rows:
        print(f"| {name} | {lic} |")

    if undeclared:
        print(f"\n## UNDECLARED ({len(undeclared)}) — resolve each before shipping",
              file=sys.stderr)
        for n in undeclared:
            print(f"  - {n}", file=sys.stderr)
        print("\n  Check the wheel for a LICENSE file and the upstream repo before\n"
              "  assuming there is no grant. A package with genuinely no licence\n"
              "  carries no redistribution right at all.", file=sys.stderr)

    exit_code = 1 if undeclared else 0

    if args.check:
        doc_path = pathlib.Path(args.check).expanduser()
        if not doc_path.exists():
            print(f"error: {doc_path} not found", file=sys.stderr)
            return 2
        doc = canon(doc_path.read_text(encoding="utf-8", errors="replace").lower())
        missing = [n for n, _ in rows if canon(n) not in doc]
        print(f"\n## Not named in {doc_path.name}: {len(missing)}", file=sys.stderr)
        for n in missing:
            print(f"  - {n}", file=sys.stderr)
        if missing:
            exit_code = max(exit_code, 2)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
