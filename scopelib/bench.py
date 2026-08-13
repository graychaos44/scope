"""[6] 작업대 — 검사 목록·배치 실행·**교정 기한** (2026-08-13).

설계 정본: `restructure/SCOPE_DESIGN_0813.md` §4

## 왜 있나

사용자가 보여준 실험실 프로그램처럼, **무엇을 검사해야 하는지 목록**이 있고 **한 번에 돌리고**
결과가 쌓여야 한다. 그런데 더 중요한 것이 하나 있었다:

  ★**교정 기한** — 실험실은 저울을 정기 교정하고, **기한 지난 장비의 값은 안 쓴다.**

08-13 에 정본 검사 3종이 심은 결함 10개 중 **3개만** 잡는 것이 드러났는데, 그전까지 아무도 몰랐다.
검측기도 정기적으로 **정답지에 대조**해야 하고, 안 한 검측기의 결과는 판정에 쓰면 안 된다.

★그리고 08-13 에 이런 일도 있었다 — 작업대에 게임을 **등록해 놓고 한 번도 안 돌려서**
  "상수로 이겨지는 게임" 을 못 잡고 판정이 뒤집혔다(V702). 등록과 실행은 다르다.
"""
import datetime
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKLIST = os.path.join(ROOT, "scope_worklist.tsv")
CAL = os.path.join(ROOT, "scope_calibration.json")
CAL_DAYS = int(os.environ.get("CAL_DAYS", "14"))
PY = os.path.join(ROOT, "venv/bin/python")
SCOPE = os.path.join(ROOT, "scope.py")


def today():
    return datetime.date.today().isoformat()


# ── 검사 목록 ───────────────────────────────────────────
def load_list():
    if not os.path.exists(WORKLIST):
        with open(WORKLIST, "w", encoding="utf-8") as f:
            f.write("# SCOPE 검사 대기 목록 — 지우지 말고 상태만 바꾼다\n")
            f.write("대상\t종류\t등록일\t이유\t상태\n")
        return []
    rows = []
    for line in open(WORKLIST, encoding="utf-8"):
        if line.startswith("#") or line.startswith("대상"):
            continue
        c = line.rstrip("\n").split("\t")
        if len(c) >= 5:
            rows.append(dict(zip(("대상", "종류", "등록일", "이유", "상태"), c)))
    return rows


def add(target, kind, reason):
    load_list()
    with open(WORKLIST, "a", encoding="utf-8") as f:
        f.write(f"{target}\t{kind}\t{today()}\t{reason}\t대기\n")
    return f"등록: {target}({kind}) — {reason}"


# ── 교정 기한 ───────────────────────────────────────────
def load_cal():
    if os.path.exists(CAL):
        return json.load(open(CAL, encoding="utf-8"))
    return {"_설명": "검사별 정답지 대조일 = 교정일. 기한이 지나면 그 검사 결과를 판정에 쓰지 않는다.",
            "_유효기간_일": CAL_DAYS, "검사": {}}


def mark_calibrated(check_names, note=""):
    c = load_cal()
    for n in check_names:
        c["검사"][n] = {"교정일": today(), "비고": note}
    json.dump(c, open(CAL, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return c


def cal_status():
    c = load_cal()
    out = []
    for name, v in sorted(c.get("검사", {}).items()):
        d = v.get("교정일")
        if not d:
            out.append((name, "—", "★교정 안 됨", False))
            continue
        days = (datetime.date.today() - datetime.date.fromisoformat(d)).days
        ok = days <= CAL_DAYS
        out.append((name, d, f"{'유효' if ok else '★만료'}({days}일)", ok))
    return out


def usable(check_name):
    """이 검사의 결과를 판정에 써도 되나 — 교정 기한을 지났으면 안 된다."""
    for n, _d, _s, ok in cal_status():
        if n == check_name:
            return ok
    return False        # 등록조차 안 된 검사는 못 믿는다


# ── 정답지 회귀 (교정) ──────────────────────────────────
CORPUS_EXPECT = {
    # 정답지 게임 → 이 검사가 잡아야 한다(비면 아무 판정도 없어야 한다 = 오탐 시험)
    "gcorp_n1_normal_env": [],
    "gcorp_n2_normal_short_env": [],
    "gcorp_d03_deadcell_env": ["DEADCELL"],
    "gcorp_d04_range_env": ["RANGE"],
    "gcorp_d05_earlyterm_env": ["EPLEN"],
    "gcorp_d06_nowon_env": ["NOWON"],
    "gcorp_d08_notimeout_env": ["NOTERM"],
}


def calibrate(episodes=60, seeds="20000,30000,40000"):
    """정답지로 검측기를 교정한다 — **답을 내가 만든 것으로 잰다**(규약 §9-2).

    잡아야 할 것을 잡고, 정상에서 오탐이 없으면 교정 성공으로 기록한다.
    """
    from . import analyze as AN
    from . import sandbox as SB
    rows, ok_all = [], True
    for mod, expect in CORPUS_EXPECT.items():
        try:
            rec = SB.rollout(mod, None, [int(x) for x in seeds.split(",")], episodes,
                             SB.Intervention("무작위"))
            got = [n for n, _w, _v in AN.analyze(rec)]
        except Exception as e:
            rows.append((mod, expect, [f"실패:{type(e).__name__}"], False))
            ok_all = False
            continue
        miss = [e for e in expect if e not in got]
        false_pos = (not expect) and got
        good = (not miss) and (not false_pos)
        ok_all = ok_all and good
        rows.append((mod, expect, got, good))
    if ok_all:
        mark_calibrated(["NOWON", "NOWIN?", "DEADCELL", "RANGE", "EPLEN", "NOTERM"],
                        note=f"정답지 {len(CORPUS_EXPECT)}종 전건 통과")
    return rows, ok_all


# ── 배치 실행 ───────────────────────────────────────────
def run_batch(only=None, cmds=("검사", "기준선")):
    """목록에 있는 것을 한 번에 돌린다. ★등록만 하고 안 돌리는 것을 막는 자리다."""
    out = []
    for r in load_list():
        if r["상태"] == "보류" or (only and r["대상"] != only):
            continue
        for c in cmds:
            p = subprocess.run([PY, SCOPE, c, r["대상"]], cwd=ROOT,
                               capture_output=True, text=True, timeout=3600,
                               env=dict(os.environ, OMP_NUM_THREADS="1"))
            head = [ln for ln in (p.stdout or "").splitlines()
                    if ln[:4].strip() in ("OK", "FAIL") or ln.startswith("★이 게임")]
            out.append((r["대상"], c, p.returncode, head[0][:90] if head else "(출력 없음)"))
    return out
