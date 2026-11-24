# Decoupage des docs Markdown en chunks citables
import re, json, pathlib, statistics

DOCS = pathlib.Path("data/docs")
OUT = pathlib.Path("index/chunks.jsonl")
MAX_WORDS = 200

HEADING = re.compile(r"^(#{1,3})\s+(.*)")
FENCE = re.compile(r"^\s*(```|~~~)")


def parse_sections(md):
    """Retourne (titre, [(chemin_section, lignes), ...]). Ignore les # dans le code."""
    title, path, lines, sections, in_code = None, [], [], [], False
    for line in md.splitlines():
        if FENCE.match(line):
            in_code = not in_code
            lines.append(line)
            continue
        m = None if in_code else HEADING.match(line)
        if m:
            level, text = len(m.group(1)), m.group(2).strip()
            if level == 1 and title is None:
                title = text
                continue
            if any(l.strip() for l in lines):
                sections.append((" › ".join(path) or "Introduction", lines))
            path = path[:max(level - 2, 0)] + [text]
            lines = []
        else:
            lines.append(line)
    if any(l.strip() for l in lines):
        sections.append((" › ".join(path) or "Introduction", lines))
    return title, sections


def to_blocks(lines):
    """Découpe une section en blocs : paragraphe/liste, tableau, bloc de code."""
    blocks, buf, kind, in_code = [], [], None, False

    def flush():
        nonlocal buf, kind
        if buf and any(l.strip() for l in buf):
            blocks.append((kind or "text", "\n".join(buf).strip()))
        buf, kind = [], None

    for line in lines:
        if in_code:
            buf.append(line)
            if FENCE.match(line):
                in_code = False
                flush()
            continue
        if FENCE.match(line):
            flush()
            in_code, kind, buf = True, "code", [line]
            continue
        s = line.strip()
        if not s or s == "---":
            flush()
            continue
        is_table = s.startswith("|")
        if is_table and kind != "table":
            flush()
            kind = "table"
        elif not is_table and kind == "table":
            flush()
        buf.append(line)
    flush()
    return blocks


def n_words(text):
    return len(text.split())


def split_big(kind, text):
    """Coupe un bloc trop long ; un tableau garde son en-tête dans chaque morceau."""
    if n_words(text) <= MAX_WORDS:
        return [text]
    lines = text.splitlines()
    header, rows = (lines[:2], lines[2:]) if kind == "table" and len(lines) > 2 else ([], lines)
    out, cur = [], []
    for row in rows:
        if cur and n_words("\n".join(header + cur + [row])) > MAX_WORDS:
            out.append("\n".join(header + cur))
            cur = []
        cur.append(row)
    if cur:
        out.append("\n".join(header + cur))
    return out


def pack(blocks):
    """Regroupe des blocs entiers jusqu'à MAX_WORDS, sans jamais couper un bloc."""
    chunks, cur = [], []
    for kind, text in blocks:
        for piece in split_big(kind, text):
            if cur and n_words("\n\n".join(cur + [piece])) > MAX_WORDS:
                chunks.append("\n\n".join(cur))
                cur = []
            cur.append(piece)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def main():
    OUT.parent.mkdir(exist_ok=True)
    rows = []
    for p in sorted(DOCS.glob("*.md")):
        title, sections = parse_sections(p.read_text(encoding="utf-8"))
        title = title or p.stem
        for s_i, (section, lines) in enumerate(sections):
            for c_i, text in enumerate(pack(to_blocks(lines))):
                rows.append({
                    "chunk_id": f"{p.stem}#{s_i}.{c_i}",
                    "doc_id": p.stem,
                    "title": title,
                    "section": section,
                    "source": str(p),
                    "n_words": n_words(text),
                    "text": text,
                })

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(rows)} chunks écrits dans {OUT}\n")
    for doc in sorted({r["doc_id"] for r in rows}):
        sizes = [r["n_words"] for r in rows if r["doc_id"] == doc]
        print(f"{doc}: {len(sizes)} chunks | mots min/médiane/max = "
              f"{min(sizes)}/{int(statistics.median(sizes))}/{max(sizes)}")

    sample = next((r for r in rows if "|" in r["text"]), rows[0])
    print(f"\n--- Exemple : {sample['chunk_id']} ---\n[{sample['section']}]\n{sample['text'][:600]}")


if __name__ == "__main__":
    main()