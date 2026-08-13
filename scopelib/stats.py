"""[5] 판정·통계 — 수치에 오차범위를 붙이고, 비교가 성립하는지부터 본다 (2026-08-13).

설계 정본: `restructure/SCOPE_DESIGN_0813.md` §3

## 왜 있나

지금까지는 **평균만** 봤다. 08-12 하루에 소표본이 방향을 **세 번** 거꾸로 보여줬고,
08-12 에 같은 사안에 판정이 **세 번 뒤집혔는데** 원인은 데이터도 모델도 아니라
**매번 조건이 달랐던 것**이었다(DEFECT P21).

  → 그래서 이 모듈은 두 가지를 강제한다:
     ①모든 수치에 **신뢰구간**을 붙인다
     ②**조건 지문이 다르면 비교를 거부**한다

## 표준

IQM(사분위 평균) + 층화 부트스트랩 — rliable(Agarwal 2021). 규약 §9-2 가 이름까지 적어둔 표준.
여기서는 의존성 없이 같은 계산을 numpy 로 한다(venv 를 늘리지 않는다).
"""
import numpy as np

MARGIN = 5.0            # 기준선 대비 이만큼은 넘어야 '배웠다'
BOOT = 2000


def iqm(x):
    """사분위 평균 — 위아래 25% 를 버리고 평균. 이상치 한 판에 안 흔들린다."""
    x = np.sort(np.asarray(x, dtype=float))
    if len(x) < 4:
        return float(np.mean(x)) if len(x) else float("nan")
    k = len(x) // 4
    return float(np.mean(x[k:len(x) - k]))


def boot_ci(x, alpha=0.05, n=BOOT, seed=0):
    """부트스트랩 신뢰구간. 시드를 고정해 **같은 입력이면 같은 구간**이 나오게 한다."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    means = [float(np.mean(rng.choice(x, len(x), replace=True))) for _ in range(n)]
    lo, hi = np.percentile(means, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return (float(lo), float(hi))


def summarize(values, seed=0):
    """시드별 값 목록 → 요약. ★시드별 값을 **항상 같이** 남긴다(평균만 보면 한 시드가 끈 것을 못 본다)."""
    v = np.asarray(values, dtype=float)
    lo, hi = boot_ci(v, seed=seed)
    return {"평균": float(np.mean(v)), "IQM": iqm(v), "표준편차": float(np.std(v)),
            "최소": float(np.min(v)), "최대": float(np.max(v)),
            "CI95": [round(lo, 2), round(hi, 2)], "시드별": [round(float(x), 2) for x in v],
            "n": int(len(v))}


def overlap(a, b):
    """두 신뢰구간이 겹치나 — 겹치면 '다르다' 고 말할 수 없다."""
    return not (a[1] < b[0] or b[1] < a[0])


# ── 조건 지문 ────────────────────────────────────────────
COMPARE_KEYS = ("env", "episodes", "seeds", "decision_rule", "개입", "policy")


def fingerprint(params):
    """비교에 쓰이는 조건만 뽑는다. 여기가 다르면 비교가 성립하지 않는다."""
    return {k: params.get(k) for k in COMPARE_KEYS if k in params}


def comparable(p1, p2):
    """비교해도 되는가. ★다르면 **거부**한다 — 08-12 에 조건이 달라 판정이 세 번 뒤집혔다."""
    f1, f2 = fingerprint(p1), fingerprint(p2)
    diff = {k: (f1.get(k), f2.get(k)) for k in set(f1) | set(f2) if f1.get(k) != f2.get(k)}
    # 개입·정책이 다른 것은 **일부러** 비교하는 것이므로 허용한다
    blocking = {k: v for k, v in diff.items() if k in ("env", "episodes", "seeds", "decision_rule")}
    return (not blocking), blocking


# ── 판정 ────────────────────────────────────────────────
def verdict(adapter_vals, random_vals, const_best, margin=MARGIN, seed=0):
    """어댑터가 **무작위와 상수를 둘 다** 넘었나.

    ★08-13 에 상수를 빼고 판정했다가 뒤집혔다(V702) — 어댑터 88.9% 인데 상수가 93.3% 였다.
    ★`판정불가` 를 실패와 **분리**한다. 천장에 걸려 차이가 안 나는 것은 실패가 아니다(V701).
    """
    a = summarize(adapter_vals, seed=seed)
    r = summarize(random_vals, seed=seed)
    er, ec = a["평균"] - r["평균"], a["평균"] - const_best

    if overlap(a["CI95"], r["CI95"]) and er < margin:
        v = "판정불가"
        why = f"어댑터 CI{a['CI95']} 와 무작위 CI{r['CI95']} 가 겹친다 — 표본을 늘려라"
    elif ec < margin:
        v = "★상수이하"
        why = f"상수 최고 {const_best:.1f}% 대비 {ec:+.1f}%p — 버튼 하나로도 이 정도 나온다"
    elif er < margin:
        v = "★무작위이하"
        why = f"무작위 {r['평균']:.1f}% 대비 {er:+.1f}%p"
    else:
        v = "통과"
        why = f"무작위 {er:+.1f}%p · 상수 {ec:+.1f}%p — 둘 다 넘었다"
    return {"판정": v, "이유": why, "어댑터": a, "무작위": r,
            "상수최고": round(float(const_best), 2),
            "무작위대비": round(er, 2), "상수대비": round(ec, 2),
            "신뢰도": "CANDIDATE"}       # ★확정은 팀장 몫(규약 §9-2)


def compare(run_a, run_b, key="result.winrate"):
    """두 실행을 같은 자로 비교한다. 조건이 다르면 **거부**."""
    ok, blocking = comparable(run_a["params"], run_b["params"])
    if not ok:
        return {"비교": "거부", "이유": "조건이 다르다 — 비교가 성립하지 않는다", "다른조건": blocking}
    va, vb = run_a["metrics"].get(key), run_b["metrics"].get(key)
    if va is None or vb is None:
        return {"비교": "거부", "이유": f"지표 {key} 가 한쪽에 없다"}
    return {"비교": "성립", "지표": key, "A": va, "B": vb, "차이": round(vb - va, 2),
            "조건": fingerprint(run_a["params"])}
