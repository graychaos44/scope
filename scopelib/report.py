"""[7] 보고 — 실행 기록을 사람이 읽는 형태로 내보낸다 (2026-08-13).

설계 정본: `restructure/SCOPE_DESIGN_0813.md` §5

★**원본을 안 바꾼다.** 항상 새 파일로 낸다.
★보고서 문장은 **수치를 옮긴 것**이어야 하고 **출처(run_id)를 단다** —
  언어모델이 새 수치를 만들어 넣지 않는다(설계 §10, 08-13 사용자 원칙).
"""
import json
import os

from .record import RUNS, load

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def _rows(run_ids):
    return [load(r) for r in run_ids]


def to_md(run_ids, title="SCOPE 보고"):
    ds = _rows(run_ids)
    keys = []
    for d in ds:
        for k in d["metrics"]:
            if k not in keys:
                keys.append(k)
    L = [f"# {title}", "",
         f"실행 {len(ds)}건 · 생성 {ds[0]['ended'] if ds else '-'}", "",
         "## 조건", ""]
    for d in ds:
        L.append(f"- `{d['run_id']}` · {d['kind']} · 출처 **{d['source']}** · {d['seconds']}초 · "
                 f"{json.dumps(d['params'], ensure_ascii=False)}")
    L += ["", "## 지표", "", "| 지표 | " + " | ".join(d["run_id"][-6:] for d in ds) + " |",
          "|---|" + "---|" * len(ds)]
    for k in keys:
        L.append(f"| `{k}` | " + " | ".join(
            (f"{d['metrics'][k]:g}" if k in d["metrics"] else "—") for d in ds) + " |")
    tags = [d for d in ds if d.get("tags")]
    if tags:
        L += ["", "## 꼬리표", ""]
        for d in tags:
            L.append(f"- `{d['run_id'][-6:]}` {json.dumps(d['tags'], ensure_ascii=False)}")
    errs = [d for d in ds if d.get("error")]
    if errs:
        L += ["", "## ★오류", ""] + [f"- `{d['run_id'][-6:]}` {d['error']}" for d in errs]
    L += ["", "---", "",
          "★이 표의 수치는 실행 기록에서 그대로 옮긴 것이다. 출처는 위 run_id.",
          "★`출처` 칸이 `simulated` 인 것은 **시뮬 값**이라 판정 근거로 쓰지 않는다(규약 §9-1)."]
    return "\n".join(L)


def to_tsv(run_ids):
    ds = _rows(run_ids)
    keys = sorted({k for d in ds for k in d["metrics"]})
    L = ["run_id\t종류\t출처\t초\t" + "\t".join(keys)]
    for d in ds:
        L.append("\t".join([d["run_id"], d["kind"], d["source"], str(d["seconds"])]
                           + [str(d["metrics"].get(k, "")) for k in keys]))
    return "\n".join(L)


def export(run_ids, fmt="md", out=None, title="SCOPE 보고"):
    os.makedirs(OUTDIR, exist_ok=True)
    if fmt == "md":
        body, ext = to_md(run_ids, title), "md"
    elif fmt == "tsv":
        body, ext = to_tsv(run_ids), "tsv"
    elif fmt == "json":
        body, ext = json.dumps(_rows(run_ids), ensure_ascii=False, indent=1), "json"
    else:
        raise ValueError(f"모르는 형식: {fmt} (md·tsv·json)")
    path = out or os.path.join(OUTDIR, f"scope_{run_ids[-1]}.{ext}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(body + "\n")
    return path
