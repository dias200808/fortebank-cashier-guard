"""Markdown for README sections «Типы нарушений» and «Нормативные источники и принцип сопоставления», built from
app/rules.yaml + app/norms.yaml (the same data as GET /api/v1/rules).

    .venv/bin/python -m scripts.catalog            # print the markdown
    .venv/bin/python -m scripts.catalog --write    # replace the block between the catalog markers in README.md
    .venv/bin/python -m scripts.catalog --check    # exit 1 if README.md is out of date
"""
import argparse
import sys
from pathlib import Path

from app.api import load_catalog

README = Path(__file__).resolve().parents[1] / "README.md"
START, END = "<!-- catalog:start -->", "<!-- catalog:end -->"


def cell(text: str) -> str:
    return " ".join(str(text or "").split()).replace("|", "\\|")


def plural(n: int, one: str, few: str, many: str) -> str:
    a, b = n % 10, n % 100
    return one if a == 1 and b != 11 else few if 2 <= a <= 4 and not 12 <= b <= 14 else many


def render(cat: dict) -> str:
    short = {s["url"]: s["short"] for s in cat["sources"]}
    n = len(cat["rules"])
    out = [
        "## 5. Типы нарушений",
        "",
        f"Сервис распознаёт {n} {plural(n, 'тип', 'типа', 'типов')} нарушений. Таблица собрана скриптом "
        "`scripts/catalog.py` из `app/rules.yaml` (после изменения правил: `.venv/bin/python -m scripts.catalog --write`). "
        "Тот же каталог открывается в веб-интерфейсе по ссылке «Нарушения и источники» и отдаётся по `GET /api/v1/rules`. "
        "Цитаты норм дословные, из `docs/normative_research.md`.",
        "",
        "| id | Тип | Описание | Как видно на видео | Норма | Сила |",
        "|---|---|---|---|---|---|",
    ]
    for r in cat["rules"]:
        nm = r["norm"]
        act = f"[{short.get(nm['url'], nm['act'])}]({nm['url']})" if nm["url"] else nm["act"]
        norm = f"{act}, {nm['point']}: «{nm['quote']}»"
        out.append(f"| {r['id']} | {cell(r['type'])} | {cell(r['summary'])} | {cell(r['signals'])} | {cell(norm)} | {r['strength']} |")
    out += ["", "Сила связи:", ""]
    out += [f"- **{k}** — {v['label']}: {v['text']}." for k, v in cat["strengths"].items()]
    out += [
        "",
        "## 6. Нормативные источники и принцип сопоставления",
        "",
        f"Тексты актов сверены с adilet.zan.kz {cat['checked']} (`docs/normative_research.md`, обозначения актов — как там). "
        "Утратившие силу акты не цитируются.",
        "",
        "| Код | Акт | Для чего используется |",
        "|---|---|---|",
    ]
    for s in cat["sources"]:
        cited = ", ".join(f"{r['id']} ({r['point']})" for r in s.get("rules", []))
        used = s["used_for"] + (f" **Правила каталога:** {cited}." if cited else "")
        out.append(f"| {s['code']} | [{cell(s['act'])}]({s['url']}) | {cell(used)} |")
    out += ["", "### Принцип сопоставления", ""]
    for p in cat["principle"].split("\n\n"):
        out += [p.strip(), ""]
    out += ["### Что не проверяется по видео", "", "| Тип | Почему не проверяется |", "|---|---|"]
    out += [f"| {cell(x['item'])} | {cell(x['why'])} |" for x in cat["not_checkable"]]
    return "\n".join(out)


def splice(readme: str, block: str) -> str:
    a, b = readme.index(START) + len(START), readme.index(END)
    return readme[:a] + "\n" + block + "\n" + readme[b:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="update README.md in place")
    mode.add_argument("--check", action="store_true", help="exit 1 if README.md is out of date")
    args = ap.parse_args()
    block = render(load_catalog())
    if not (args.write or args.check):
        print(block)
        return 0
    readme = README.read_text(encoding="utf-8")
    fresh = splice(readme, block)
    if args.check:
        if fresh != readme:
            print("README.md catalog is out of date: run .venv/bin/python -m scripts.catalog --write", file=sys.stderr)
            return 1
        return 0
    README.write_text(fresh, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
