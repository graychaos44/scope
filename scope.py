"""SCOPE — 검측·분석 프로그램 (한 진입점) (2026-08-13 신설).

하루 종일 세운 뼈대(`restructure/GATE_SKELETON_0813.md`)의 실물이다.
흩어진 스크립트가 아니라 **하나의 프로그램**으로 부른다.

    venv/bin/python scope.py 검사 <게임>        굴리고·기록하고·판정한다(한 번 굴려 여섯 검사)
    venv/bin/python scope.py 기준선 <게임>      무작위·상수를 재서 "이 게임이 쉬운가" 를 본다
    venv/bin/python scope.py 능력 <게임> <체크포인트>  어댑터가 무슨 능력을 쓰는지 (개입 실험)
    venv/bin/python scope.py 해부 <체크포인트> [이전체크포인트]   안에 무엇이 들었나 [9]
    venv/bin/python scope.py 작업대 [목록|등록|실행|교정]          한 번에 여러 개 · 교정 기한 [6]
    venv/bin/python scope.py 비교 <run_id> <run_id>               같은 자로 비교(조건 다르면 거부) [5]
    venv/bin/python scope.py 보고 <run_id...> [--형식 md]          사람이 읽는 표로 [7]
    venv/bin/python scope.py 현황               최근 실행들
    venv/bin/python scope.py 기록 <run_id>      그 실행의 내용

## 무엇이 달라지나

| | 예전 | 지금 |
|---|---|---|
| 게임 굴리기 | 검사마다 따로 (5번) | **한 번** |
| 남는 것 | 도구가 볼 것만 | **매 수 전부** — 나중에 새 검사를 옛 기록에 돌릴 수 있다 |
| 조건·결과 | 로그에 흩어짐 | **실행 대장 한 줄** (나중에 대리 모델 재료) |
| 상수 기준선 | 빠뜨리기 쉬움 | **`기준선` 명령이 항상 같이 잰다**(V702 사고 방지) |

★기록은 `runs/<run_id>/` · 대장은 `SCOPE_LEDGER.tsv` · 지표 사전은 `scope_metrics.json`.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scopelib import analyze as AN          # noqa: E402
from scopelib import sandbox as SB          # noqa: E402
from scopelib.record import Run, find, load  # noqa: E402

EP = int(os.environ.get("EP", "100"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "20000,30000,40000").split(",")]


def _module(game):
    return game if game.endswith("_env") else f"{game}_env"


def cmd_check(game):
    """굴리고 → 기록하고 → 기록만 읽어 판정한다."""
    mod = _module(game)
    print(f"[검사] {mod} · {EP}판 × 시드{len(SEEDS)} · 무작위 정책으로 굴린다\n")
    with Run("검사", params={"env": mod, "episodes": EP, "seeds": SEEDS, "policy": "random"},
             tags={"도구": "scope.py 검사"}) as r:
        rec = SB.rollout(mod, None, SEEDS, EP, SB.Intervention("무작위"))
        for k, v in AN.metrics(rec).items():
            r.metric(k, v, unit=AN.UNITS.get(k, "unknown"))
        finds = AN.analyze(rec)
        r.metric("gate.finding_count", float(len(finds)), unit="count")
        r.tag("판정", "FAIL" if finds else "OK")

        S = rec["요약"]
        print(f"{'FAIL' if finds else 'OK':5s} 판정 {len(finds)}건 | 무작위승률 {S['승률']:5.1f}% "
              f"평균 {S['평균길이']:6.1f}수 미종료 {S['미종료']} | 관측 {S['관측칸']}칸 버튼 {S['버튼수']}")
        if S.get("예약칸"):
            print(f"      (예약 칸 {S['예약칸']} 은 죽은 칸 검사에서 뺐다)")
        for name, why, _v in finds:
            print(f"      [{name}] {why}")
        print(f"\n[기록] runs/{r.id}/run.json")
    return 1 if finds else 0


def cmd_baseline(game):
    """★이 게임이 **그냥 쉬운 것**인지 본다 — 무작위·상수·무관측을 같이 잰다.

    08-13 에 이것을 빠뜨려 판정이 뒤집혔다(V702): 어댑터 88.9% 가 대단해 보였는데
    **버튼 하나만 눌러도 93.3%** 였다.
    """
    mod = _module(game)
    C = SB.env_class(mod)
    n_act = int(C(seed=0).n_act)
    print(f"[기준선] {mod} · {EP}판 × 시드{len(SEEDS)}\n")
    with Run("기준선", params={"env": mod, "episodes": EP, "seeds": SEEDS},
             tags={"도구": "scope.py 기준선"}) as r:
        rnd = SB.rollout(mod, None, SEEDS, EP, SB.Intervention("무작위"), record=False)
        print(f"  무작위          {rnd['요약']['승률']:6.1f}%")
        r.metric("baseline.random_winrate", rnd["요약"]["승률"], unit=AN.UNITS["baseline.random_winrate"])

        best, best_a = -1.0, None
        for a in range(n_act):
            c = SB.rollout(mod, None, SEEDS[:1], EP, SB.Intervention("상수행동", action=a), record=False)
            w = c["요약"]["승률"]
            print(f"  상수 버튼{a}      {w:6.1f}%")
            if w > best:
                best, best_a = w, a
        r.metric("baseline.const_best", best, unit=AN.UNITS["baseline.const_best"])
        r.tag("상수최고버튼", best_a)

        # 화면을 가려도 이기나 — 무관측 상한의 싼 근사
        blind = SB.rollout(mod, None, SEEDS[:1], EP, SB.Intervention("화면가림"), record=False)
        print(f"  화면가림+무작위  {blind['요약']['승률']:6.1f}%")
        r.metric("baseline.blind_winrate", blind["요약"]["승률"], unit=AN.UNITS["baseline.blind_winrate"])

        print(f"\n★이 게임의 기준선 = 무작위 {rnd['요약']['승률']:.1f}% · **상수 최고 {best:.1f}%(버튼{best_a})**")
        print("  어떤 정책이든 이 둘을 **모두** 넘어야 '배웠다' 고 말할 수 있다")
        print(f"\n[기록] runs/{r.id}/run.json")


def cmd_probe(game, ckpt):
    """[4] 개입 — 게임을 한 가지씩 건드려 **무슨 능력을 쓰는지** 잰다."""
    from scopelib import probe as PB
    mod = _module(game)
    print(f"[능력] {mod} · {EP}판 × 시드{len(SEEDS)} · 표집\n")
    with Run("능력", params={"env": mod, "ckpt": ckpt, "episodes": EP, "seeds": SEEDS,
                            "decision_rule": "sample"}, tags={"도구": "scope.py 능력"}) as r:
        res = PB.probe(mod, ckpt, SEEDS, EP)
        r.metric("result.winrate", res["그대로"], unit="percent_0_100")
        r.metric("baseline.random_winrate", res["무작위"], unit="percent_0_100")
        r.metric("baseline.const_best", res["상수최고"], unit="percent_0_100")
        r.metric("edge.vs_baseline", res["기준선대비"], unit="percent_point",
                 meaning="어댑터 − max(무작위, 상수최고)", direction="클수록 좋음")
        for k, v in res["능력"].items():
            r.metric(f"capability.{k}_drop", v["하락"], unit="percent_point",
                     meaning=f"{k} 개입 시 승률 하락")

        print(f"{'그대로':12s} {res['그대로']:6.1f}%")
        for k, v in res["능력"].items():
            print(f"{k:12s} {v['승률']:6.1f}%  하락 {v['하락']:+6.1f}%p")
        print(f"\n기준선: 무작위 {res['무작위']:.1f}% · 상수최고 {res['상수최고']:.1f}% "
              f"→ 어댑터가 {res['기준선대비']:+.1f}%p")
        if not res["믿을수있나"]:
            print("★★기준선을 못 넘는다 — 위 하락폭은 **능력의 증거가 아니다**")
            print("   (기준선으로도 같은 승률이 나오므로, 떨어진다는 것은 '쓴다' 이지 '필요하다' 가 아니다)")
        if res["칸별중요도"]:
            top = res["칸별중요도"][:5]
            print("칸별 중요도 상위: " + " · ".join(f"칸{c['칸']}={c['하락']:+.1f}" for c in top))
        r.tag("믿을수있나", res["믿을수있나"])
        print(f"\n[기록] runs/{r.id}/run.json")


def cmd_dissect(ckpt, prev=None):
    """[9] 해부 — 읽기만 한다. 고치지 않는다."""
    from scopelib import dissect as DS
    print(f"[해부] {ckpt}" + (f"  (이전판 {prev} 와 비교)" if prev else "") + "\n")
    with Run("해부", params={"ckpt": ckpt, "prev": prev}, tags={"도구": "scope.py 해부"}) as r:
        h = DS.health(ckpt, prev)
        print("【학습 건강】")
        for k, v in h.items():
            if k in ("비교", "★경보"):
                continue
            print(f"  {k:20s} {v}")
            if isinstance(v, (int, float)):
                r.metric(f"core.health.{k}", float(v), unit="count_or_ratio")
        if h.get("비교"):
            print("\n【이전 판과 비교】")
            for k, v in h["비교"].items():
                print(f"  {k:20s} {v}")
            r.metric("core.health.changed_keys", float(h["비교"]["바뀐키"]), unit="count")
            r.metric("core.health.lost_keys", float(h["비교"]["★사라진키"]), unit="count")
        if h.get("★경보"):
            print(f"\n★경보: {h['★경보']}")
            r.tag("경보", h["★경보"])
        try:
            c = DS.capability(ckpt)
            print(f"\n【능력 보유】 종목 {c['종목수']}개 · 공용 코어 {c['공용코어파라미터']:,} "
                  f"({c['공용비율']*100:.0f}%)")
            r.metric("core.capability.games", float(c["종목수"]), unit="count")
            r.metric("core.capability.shared_params", float(c["공용코어파라미터"]), unit="count")
        except Exception:
            pass
        print("\n★내부 지표는 **표시만** 한다 — 결과와의 상관이 확인된 것만 판정에 쓴다(V675)")
        print(f"[기록] runs/{r.id}/run.json")


def cmd_bench(sub, *args):
    """[6] 작업대."""
    from scopelib import bench as BN
    if sub in ("목록", "list", None):
        rows = BN.load_list()
        print(f"검사 목록 {len(rows)}건\n")
        for x in rows:
            print(f"  {x['대상']:28s} {x['종류']:8s} {x['상태']:6s} {x['이유'][:34]}")
        print("\n【검측기 교정】 정답지로 마지막에 대조한 날")
        for n, d, st, ok in BN.cal_status():
            print(f"  {'' if ok else '★'}{n:12s} {d:12s} {st}")
        if not BN.cal_status():
            print("  ★한 번도 교정 안 됨 — `작업대 교정` 을 돌려라")
    elif sub in ("등록", "add"):
        print(BN.add(args[0], args[1] if len(args) > 1 else "게임", " ".join(args[2:]) or "-"))
    elif sub in ("교정", "calibrate"):
        print("[교정] 정답지로 검측기를 대조한다 — 답을 내가 만든 것으로 잰다\n")
        rows, ok = BN.calibrate()
        for mod, exp, got, good in rows:
            print(f"  {'OK ' if good else '★틀림'} {mod:28s} 기대 {exp} → 나옴 {got}")
        print(f"\n{'교정 성공 — 검사들에 오늘 날짜를 찍었다' if ok else '★교정 실패 — 검측기를 고쳐야 한다'}")
    elif sub in ("실행", "run"):
        for t, c, rc, head in BN.run_batch(only=args[0] if args else None):
            print(f"  {t:26s} {c:6s} rc={rc}  {head}")
    else:
        print("작업대 [목록|등록|실행|교정]")


def cmd_compare(a, b):
    from scopelib import stats as ST
    ra, rb = load(a), load(b)
    res = ST.compare(ra, rb)
    if res["비교"] == "거부":
        print(f"★비교 거부 — {res['이유']}")
        for k, v in res.get("다른조건", {}).items():
            print(f"   {k}: {v[0]}  vs  {v[1]}")
        print("   같은 조건으로 다시 재야 비교가 성립한다(08-12 에 이것 때문에 판정이 세 번 뒤집혔다)")
        return
    print(f"비교 성립 · 지표 {res['지표']}\n  A {res['A']}  →  B {res['B']}  차이 {res['차이']:+}")
    print(f"  같은 조건: {res['조건']}")


def cmd_report(run_ids, fmt="md"):
    from scopelib import report as RP
    p = RP.export(run_ids, fmt=fmt)
    print(f"[보고] {p}")


def cmd_status():
    rows = find(limit=15)
    if not rows:
        return print("아직 실행 기록이 없다")
    print(f"최근 실행 {len(rows)}건  (대장 SCOPE_LEDGER.tsv)\n")
    print(f"{'시각':22s} {'종류':10s} {'출처':10s} {'초':>6s}  설정")
    print("-" * 96)
    for r in rows:
        print(f"{r['시각'][:19]:22s} {r['종류']:10s} {r['출처']:10s} {r['초']:>6s}  {r['설정'][:44]}")


def cmd_show(run_id):
    d = load(run_id)
    print(f"실행 {d['run_id']} · {d['kind']} · 출처 {d['source']} · {d['seconds']}초 · {d['host']}")
    print(f"\n설정: {d['params']}")
    print("\n지표:")
    for k, v in d["metrics"].items():
        print(f"  {k:28s} {v}")
    if d["artifacts"]:
        print("\n산출물:", [a["경로"] for a in d["artifacts"]])
    if d["tags"]:
        print("꼬리표:", d["tags"])
    if d.get("error"):
        print("★오류:", d["error"])


def main(argv):
    cmd = argv[0] if argv else "현황"
    if cmd in ("검사", "check"):
        if len(argv) < 2:
            return print("게임을 적어라: 검사 chase")
        sys.exit(cmd_check(argv[1]))
    elif cmd in ("기준선", "baseline"):
        if len(argv) < 2:
            return print("게임을 적어라: 기준선 chase")
        cmd_baseline(argv[1])
    elif cmd in ("능력", "probe"):
        if len(argv) < 3:
            return print("게임과 체크포인트를 적어라: 능력 chase checkpoints_kernel/chase_adapter.pt")
        cmd_probe(argv[1], argv[2])
    elif cmd in ("해부", "dissect"):
        if len(argv) < 2:
            return print("체크포인트를 적어라: 해부 checkpoints_kernel/reflex_core.pt")
        cmd_dissect(argv[1], argv[2] if len(argv) > 2 else None)
    elif cmd in ("작업대", "bench"):
        cmd_bench(argv[1] if len(argv) > 1 else "목록", *argv[2:])
    elif cmd in ("비교", "compare"):
        if len(argv) < 3:
            return print("run_id 두 개를 적어라")
        cmd_compare(argv[1], argv[2])
    elif cmd in ("보고", "report"):
        ids = [a for a in argv[1:] if not a.startswith("--")]
        fmt = argv[argv.index("--형식") + 1] if "--형식" in argv else "md"
        if not ids:
            return print("run_id 를 적어라")
        cmd_report(ids, fmt)
    elif cmd in ("현황", "status"):
        cmd_status()
    elif cmd in ("기록", "show"):
        if len(argv) < 2:
            return print("run_id 를 적어라")
        cmd_show(argv[1])
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
