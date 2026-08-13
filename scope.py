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


def cmd_bench_speed(args):
    """[11] 벤치마크 — 어디가 얼마나 빠른지 한 자로 (08-14 신설).

    사용: scope.py 벤치 [게임 …]
    ★병목을 고치기 전에 **어디가 병목인지부터** 잰다.
      08-13 에 게임을 2.5배 빠르게 고쳤는데 전체는 안 변했다 — 진짜 병목은 신경망(96%)이었다.
    """
    from scopelib import benchmark as BM
    games = [a for a in args if not a.startswith("--")] or ["resource_defense_env"]
    m = BM.machine()
    print(f"[벤치마크] {m['기기']} · CPU {m['cpu']} · GPU {m['cuda']}\n")
    rows = []
    for g in games:
        try:
            rows.append(BM.bench_env(g))
        except Exception as ex:
            print(f"  ★{g} 실패: {type(ex).__name__}: {ex}")
    try:
        rows.append(BM.bench_net(device="cpu"))
        import torch
        if torch.cuda.is_available():
            rows.append(BM.bench_net(device="cuda"))
            rows.append(BM.bench_net(device="cuda", image=(12, 120, 160), batch=12, iters=200))
    except Exception as ex:
        print(f"  ★신경망 재기 실패: {type(ex).__name__}: {ex}")
    rows += BM.bench_record()

    print(f"{'항목':30s} {'초당 수':>12s} {'한 수':>10s}")
    print("-" * 56)
    for r in rows:
        print(f"{r['항목']:30s} {r['초당수']:12,.0f} {r['한수us']:9.2f}us")
    print()
    print(BM.verdict(rows))
    with Run("벤치마크", source="measured",
             params={"env": ",".join(games), "episodes": 0,
                     "decision_rule": "speed_bench", "기기": m["기기"]}) as run:
        for r in rows:
            key = r["항목"].replace(":", ".").replace("/", "_")
            run.metric(f"speed.{key}", r["초당수"], unit="steps_per_sec")
        run.table("벤치", rows)
        run.tag("판정", BM.verdict(rows))


def cmd_register(game, ckpt=None):
    """[S1] 등록 한 번으로 **검사 → 기준선 → (있으면)능력 → 격차** 를 다 돌린다 (08-14 신설).

    설계 완료조건 S1: *"게임 하나를 등록하면 **사람 손 없이** 검사→기준선→능력→격차가 다 나온다."*
    지금까지는 명령을 하나씩 쳐야 했다 — 그래서 **빠뜨리는 일이 생겼다**
    (08-13: 기준선을 안 재고 '전이 성립' 이라 판정했다가 뒤집힘, V702).

    ★앞 단계가 실패하면 뒤를 **안 돌린다**(관문 규율 — 규약 §9-2).
    """
    from scopelib import bench as BN
    print(f"[등록] {game} — 검사부터 차례로 돌린다. 앞이 막히면 뒤는 안 돌린다\n")
    steps = []

    print("① 검사 — 과제로 성립하는가")
    rc = cmd_check(game)
    steps.append(("검사", rc == 0))
    if rc != 0:
        print("\n★검사에서 막혔다 — 기준선을 안 돌린다. 게임을 먼저 고쳐라")
        _register_summary(game, steps)
        return 1

    print("\n② 기준선 — 그냥 쉬운 게임은 아닌가")
    try:
        cmd_baseline(game)
        steps.append(("기준선", True))
    except Exception as ex:
        print(f"★기준선 실패: {type(ex).__name__}: {ex}")
        steps.append(("기준선", False))
        _register_summary(game, steps)
        return 1

    if ckpt and os.path.exists(ckpt):
        print("\n③ 능력 — 무엇을 보고 푸는가")
        try:
            cmd_probe(game, ckpt)
            steps.append(("능력", True))
        except Exception as ex:
            print(f"★능력 실패: {type(ex).__name__}: {ex}")
            steps.append(("능력", False))
    else:
        print("\n③ 능력 — 건너뜀(체크포인트를 안 줬다. `등록 <게임> <체크포인트>` 로 주면 잰다)")

    print("\n④ 작업대 등록 — 다음 배치 실행에 들어간다")
    print("  " + BN.add(game, "게임", "등록 명령으로 자동 등록"))
    steps.append(("작업대등록", True))
    _register_summary(game, steps)
    return 0


def _register_summary(game, steps):
    print(f"\n{'='*54}")
    print(f"[등록 결과] {game}")
    for name, ok in steps:
        print(f"  {'통과' if ok else '★막힘'}  {name}")
    print("=" * 54)


def cmd_compose(args):
    """부품을 분해해 조합하고 **짧게 진짜로 굴려** 수치를 낸다 (08-14 신설).

    사용:
        scope.py 조합 목록                       # 코어에 뭐가 있나(분해)
        scope.py 조합 부품                       # 켜고 끌 수 있는 것들
        scope.py 조합 실행 [회차] [--부품 이름,이름]
    """
    from scopelib import compose as CP
    sub = args[0] if args else "목록"

    if sub in ("목록", "분해"):
        rows = CP.inventory()
        print(f"{'부품':14s} {'상태':10s} {'파라미터':>14s}  경로")
        print("-" * 84)
        tot = 0
        for r in rows:
            tot += r["파라미터"]
            n = f"{r['파라미터']:,}" if r["파라미터"] else "-"
            print(f"{r['부품']:14s} {r['상태']:10s} {n:>14s}  {r['경로']}")
        print(f"\n합계 {tot:,} 파라미터 ({tot/1e6:.2f}M)")
        with Run("조합분해", params={"env": "core", "episodes": 0,
                                   "decision_rule": "inventory"}) as r:
            r.metric("core.total_params", tot, unit="count")
            r.table("부품", rows)
        return

    if sub == "부품":
        print(f"{'부품':12s} {'환경변수':14s} 설명")
        print("-" * 76)
        for name, var, off, on, desc in CP.PARTS:
            print(f"{name:12s} {var:14s} {desc}")
        print(f"\n조합 수 = 2^{len(CP.PARTS)} = {2**len(CP.PARTS)}가지")
        return

    if sub in ("실행", "run"):
        upd = int(args[1]) if len(args) > 1 and args[1].isdigit() else 60
        mx = int(args[args.index("--판길이") + 1]) if "--판길이" in args else 1200
        only = None
        if "--부품" in args:
            only = args[args.index("--부품") + 1].split(",")
        cs = list(CP.combos(only=only))
        print(f"[조합] {len(cs)}가지 · 각 {upd}회차 · 판 최대 {mx}수 — **짧게 진짜로 굴린다**")
        print("★이 값은 **어느 조합부터 길게 돌릴지 정렬하는 용도**다.")
        print("  판정은 정본 조건(1000판×시드3)으로 다시 한다 — 규약 §9-1 '시뮬 30%'\n")
        rows = []
        for i, c in enumerate(cs, 1):
            tag = " ".join(f"{k}{'켬' if v else '끔'}" for k, v in c.items())
            print(f"  [{i}/{len(cs)}] {tag} …", end=" ", flush=True)
            r = CP.run_one(c, updates=upd, max_step=mx)
            rows.append(r)
            print(f"판{r['판']} 클리어{r['클리어']} 출구{r['출구까지최소']} "
                  f"{r['초당수']}수/초 ({r['초']}초)", flush=True)
        print()
        keys = [p[0] for p in CP.PARTS]
        hdr = keys + ["판", "클리어", "출구까지최소", "평균보상", "초당수"]
        print(" | ".join(f"{h:>8s}" for h in hdr))
        print("-" * (11 * len(hdr)))
        for r in sorted(rows, key=lambda x: (-(x["클리어"] or 0),
                                             x["출구까지최소"] if x["출구까지최소"] is not None else 9e9)):
            print(" | ".join(f"{str(r.get(h, '-')):>8s}" for h in hdr))
        with Run("조합실행", params={"env": "e1m1", "episodes": upd,
                                   "decision_rule": "short_screen"}) as run:
            run.metric("compose.combos", len(rows), unit="count")
            run.metric("compose.updates_each", upd, unit="count")
            best = min(rows, key=lambda x: x["출구까지최소"] if x["출구까지최소"] is not None else 9e9)
            if best["출구까지최소"] is not None:
                run.metric("compose.best_exit_dist", best["출구까지최소"], unit="cells")
            run.table("조합결과", rows)
        return

    print(cmd_compose.__doc__)


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
    elif cmd in ("등록", "register"):
        if len(argv) < 2:
            return print("게임을 적어라: 등록 resource_defense")
        sys.exit(cmd_register(argv[1], argv[2] if len(argv) > 2 else None))
    elif cmd in ("벤치", "benchmark"):
        cmd_bench_speed(argv[1:] )
    elif cmd in ("조합", "compose"):
        cmd_compose(argv[1:] if len(argv) > 1 else [])
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
