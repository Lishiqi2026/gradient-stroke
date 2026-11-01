"""Prepare a private runnable copy without modifying publication source files.

This command restores study-specific cohort keys using an authorized private
mapping. No imaging data or analysis is loaded. The output is git-ignored.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-keys", type=Path, required=True,
                        help="Private JSON: PRIVATE_SUBJECT_001 -> original subject key")
    parser.add_argument("--output", type=Path, default=ROOT / "local_work" / "analysis")
    parser.add_argument("--private-values", type=Path,
                        help="Private JSON for study-specific non-ID placeholders")
    parser.add_argument("--text-replacements", type=Path,
                        help="Optional private JSON of exact source-text path replacements")
    args = parser.parse_args()
    mapping = json.loads(args.subject_keys.read_text(encoding="utf-8-sig"))
    if not isinstance(mapping, dict):
        raise ValueError("Subject-key JSON must be an object.")
    if args.private_values:
        mapping.update(json.loads(args.private_values.read_text(encoding="utf-8-sig")))
    replacements = {}
    if args.text_replacements:
        replacements = json.loads(args.text_replacements.read_text(encoding="utf-8-sig"))
    output = args.output.resolve()
    source = ROOT / "analysis"
    if source.resolve() == output or source.resolve() in output.parents:
        raise ValueError("Output must be outside the publication analysis directory.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output is not empty; use a new private output folder.")
    pending = []
    token = re.compile(r"PRIVATE_SUBJECT_\d{3}|__PRIVATE_[A-Z0-9_]+__")
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8-sig")
        needed = set(token.findall(content))
        missing = needed.difference(mapping)
        if missing:
            raise KeyError(f"{path.relative_to(ROOT)}: missing {sorted(missing)}")
        content = token.sub(lambda match: str(mapping[match.group()]), content)
        for before, after in replacements.items():
            content = content.replace(before, after)
        if path.suffix == ".py":
            ast.parse(content, filename=str(path.relative_to(ROOT)))
        pending.append((path.relative_to(source), content))
    # Validate every source before writing. Outputs may contain private IDs.
    for relative, content in pending:
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    print(f"Prepared {len(pending)} source/configuration files in {output}")
    print("This output can contain private study identifiers; do not publish it.")
    print("Configure input paths and provide authorized datasets before running analyses.")

if __name__ == "__main__":
    main()

