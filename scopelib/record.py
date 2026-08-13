"""기록 형식 실물 — 실행 하나를 남기고 읽는다 (2026-08-13 신설).

규격 정본: `restructure/DATA_API_SPEC_0813.md` · `restructure/DATA_FORMAT_SPEC_0813.md`

## 무엇을 하나

실행(run) 하나 = **설정(params) + 지표(metrics) + 산출물(artifacts) + 꼬리표(tags)**.
MLflow 방식을 따른다 — 외부 도구가 이미 이 모양을 읽는다.

    from scopelib.record import Run
    with Run("어댑터검증", params={"env":"turn", "episodes":1000}) as r:
        r.metric("result.winrate", 100.0, unit="percent_0_100")
        r.artifact("checkpoints_kernel/retrain_0813/turn_adapter.pt")
    # 끝나면 runs/<run_id>/run.json 과 실행 대장 한 줄이 자동으로 남는다

## 규격 (여기서 강제한다)

  · UTF-8 · LF · 한글 그대로 저장(ensure_ascii=False)
  · 시각은 ISO 8601 + 타임존
  · 지표 이름은 `카테고리.하위.항목` · **단위 필수**
  · `source` 필수 — measured / simulated / predicted 를 절대 섞지 않는다
  · `derived_from` 으로 파생 기록을 표현한다(여러 자료를 합쳐 만든 것)

★상한을 두지 않는다 — 카테고리·항목은 필요하면 는다(어댑터 차원 상한과는 다른 이야기).
  대신 **등록제**로 난립을 막는다: 처음 보는 이름은 사전에 자동 등록되고, 단위가 없으면 거부한다.
"""
import datetime
import json
import os
import platform
import socket
import subprocess
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
LEDGER = os.path.join(ROOT, "SCOPE_LEDGER.tsv")          # 실행 대장(한 줄 요약)
DICT = os.path.join(ROOT, "scope_metrics.json")          # 항목 사전(이름·단위·의미·방향)
SCHEMA = 1

SOURCES = ("measured", "simulated", "predicted")


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _load_dict():
    if os.path.exists(DICT):
        with open(DICT, encoding="utf-8") as f:
            return json.load(f)
    return {"_설명": "지표 사전 — 이름·단위·의미·방향. 단위 없는 지표는 받지 않는다(07-27 단위 사고).",
            "항목": {}}


def _save_dict(d):
    with open(DICT, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


def _git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


class Run:
    """실행 하나. `with` 로 쓰면 끝날 때 자동으로 저장된다."""

    def __init__(self, kind, params=None, tags=None, derived_from=None, source="measured"):
        if source not in SOURCES:
            raise ValueError(f"source 는 {SOURCES} 중 하나여야 한다 — 시뮬값이 실측에 섞이면 판정이 오염된다")
        self.id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.kind = kind
        self.source = source
        self.params = dict(params or {})
        self.tags = dict(tags or {})
        self.derived_from = list(derived_from or [])
        self.metrics = {}
        self.artifacts = []
        self.t0 = time.time()
        self.started = now()
        self.error = None

    # ── 기록 ────────────────────────────────────────────
    def metric(self, name, value, unit=None, meaning=None, direction=None):
        """지표 하나. ★단위는 필수다 — 07-27 에 비율(0~1)과 퍼센트(0~100)를 섞어
        관문 26건이 허위 발화하고 진짜 결함은 구조적으로 0건이 된 사고가 있었다."""
        d = _load_dict()
        item = d["항목"].get(name)
        if item is None:
            if not unit:
                raise ValueError(f"처음 보는 지표 '{name}' 는 **단위**를 함께 줘야 한다 (예 unit='percent_0_100')")
            d["항목"][name] = {"단위": unit, "의미": meaning or "", "방향": direction or "",
                               "처음등록": now(), "등록한실행": self.id}
            _save_dict(d)
        elif unit and item.get("단위") != unit:
            raise ValueError(f"지표 '{name}' 의 단위가 사전({item['단위']})과 다르다: {unit}")
        self.metrics[name] = float(value)
        return self

    def artifact(self, path, note=""):
        self.artifacts.append({"경로": os.path.relpath(path, ROOT) if os.path.isabs(path) else path,
                               "설명": note})
        return self

    def tag(self, k, v):
        self.tags[k] = v
        return self

    # ── 저장 ────────────────────────────────────────────
    def save(self):
        d = os.path.join(RUNS, self.id)
        os.makedirs(d, exist_ok=True)
        rec = {
            "run_id": self.id, "schema_version": SCHEMA, "kind": self.kind,
            "source": self.source,
            "started": self.started, "ended": now(), "seconds": round(time.time() - self.t0, 1),
            "host": socket.gethostname(), "arch": platform.machine(), "git": _git_rev(),
            "params": self.params, "metrics": self.metrics,
            "artifacts": self.artifacts, "tags": self.tags,
            "derived_from": self.derived_from,
            "error": self.error,
        }
        with open(os.path.join(d, "run.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)

        # 실행 대장 한 줄 — 나중에 대리 모델의 재료가 된다(설정 → 결과)
        new = not os.path.exists(LEDGER)
        with open(LEDGER, "a", encoding="utf-8") as f:
            if new:
                f.write("run_id\t시각\t종류\t출처\t기기\t초\t설정\t지표\t상태\n")
            f.write("\t".join([
                self.id, self.started, self.kind, self.source, socket.gethostname(),
                str(rec["seconds"]),
                json.dumps(self.params, ensure_ascii=False)[:300],
                json.dumps(self.metrics, ensure_ascii=False)[:300],
                "실패:" + self.error if self.error else "정상",
            ]) + "\n")
        return rec

    # ── with 문 ─────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if ev is not None:
            self.error = f"{et.__name__}: {str(ev)[:120]}"
        self.save()
        return False        # 예외를 삼키지 않는다 — 조용히 넘어가면 실패가 숨는다


# ── 읽기 ────────────────────────────────────────────────
def load(run_id):
    with open(os.path.join(RUNS, run_id, "run.json"), encoding="utf-8") as f:
        return json.load(f)


def find(kind=None, since=None, limit=50):
    """대장에서 찾는다. 파일을 다 열지 않는다."""
    if not os.path.exists(LEDGER):
        return []
    out = []
    for line in open(LEDGER, encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if len(c) < 9 or c[0] == "run_id":
            continue
        if kind and c[2] != kind:
            continue
        if since and c[1] < since:
            continue
        out.append({"run_id": c[0], "시각": c[1], "종류": c[2], "출처": c[3],
                    "기기": c[4], "초": c[5], "설정": c[6], "지표": c[7], "상태": c[8]})
    return out[-limit:]
