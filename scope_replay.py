"""SCOPE 기록 → 증류·회복학습 재료 (2026-08-13 신설).

> 사용자 08-13: *"스코프로 학습데이터까지 로그로 남는다면 **나중에 증류할 때 재학습용으로**
>  돌리면 되지 않나?"*

**맞다.** 그리고 이건 규약이 짚어온 구멍을 정확히 메운다:

  규약 §6 — *"**증류 → 재학습(회복)이 한 세트**다. 증류만 하고 끝내지 말 것."*
  그런데 회복학습은 08-12 에 **사상 처음** 돌았다. 재료를 매번 새로 굴려야 했기 때문이다.

이제는 굴릴 필요가 없다 — **재기만 해도 재료가 쌓인다.**

## 무엇을 하나

`runs/<run_id>/` 에 쌓인 SCOPE 기록을, 증류·회복학습이 이미 읽는 형식
(`reflex_distill_data/*.npz`, schema `v2_0804`)으로 바꾼다.

  obs · teacher_logits · action · reward · done · env_id · next_obs · game_feat

★소비자를 새로 만들지 않는다 — `core_recover.py` · `reflex_distill_eval.py` ·
  `central_train_loop.py` 등 **이미 이 형식을 읽는 곳이 10군데**다(규약 §9-0).

## 쓰는 법

    # 어떤 기록이 재료가 되나 (logits 가 있어야 증류에 쓴다)
    venv/bin/python scope_replay.py 목록

    # 특정 게임의 기록을 모아 npz 로
    venv/bin/python scope_replay.py 만들기 resource_defense_env

★`logits` 가 없는 기록도 **회복학습·세계모델 재료로는 쓸 수 있다**(obs·action·reward·next_obs).
  증류(로짓 추종)에만 logits 가 필요하다. 그래서 둘을 갈라 표시한다.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(ROOT, "reflex_distill_data_scope"))


def _load(run_id):
    p = os.path.join(RUNS, run_id, "steps.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return None


def scan():
    """기록마다 무엇이 들어 있는지 훑는다."""
    rows = []
    for f in sorted(glob.glob(os.path.join(RUNS, "*", "run.json"))):
        d = json.load(open(f, encoding="utf-8"))
        rid = d["run_id"]
        env = (d.get("params") or {}).get("env", "?")
        raw = os.path.join(RUNS, rid, "궤적.npz")
        rows.append({"run_id": rid, "종류": d["kind"], "게임": env,
                     "궤적파일": os.path.exists(raw)})
    return rows


def save_trajectory(run_id, rec):
    """샌드박스 기록(rollout 결과)을 궤적 npz 로 그 자리에 저장한다.

    ★JSON 으로 두면 크고 느리다. 넘파이로 눌러 담는다.
    """
    OBS, ACT, REW, DONE, EID, NOB, LG, VAL = [], [], [], [], [], [], [], []
    have_lg = have_val = True
    for ep in rec["episodes"]:
        for s in ep["steps"]:
            OBS.append(s["obs"]); ACT.append(s["a"]); REW.append(s["r"])
            DONE.append(s["done"]); EID.append(s.get("ep_id", 0))
            NOB.append(s.get("next_obs", s["obs"]))
            if "logits" in s:
                LG.append(s["logits"])
            else:
                have_lg = False
            if "value" in s:
                VAL.append(s["value"])
            else:
                have_val = False
    if not OBS:
        return None
    d = os.path.join(RUNS, run_id)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "궤적.npz")
    kw = dict(obs=np.asarray(OBS, dtype=np.float32),
              action=np.asarray(ACT, dtype=np.int32),
              reward=np.asarray(REW, dtype=np.float32),
              done=np.asarray(DONE, dtype=np.bool_),
              env_id=np.asarray(EID, dtype=np.int16),
              next_obs=np.asarray(NOB, dtype=np.float32),
              schema=np.str_("v2_0804"))
    if have_lg:
        kw["teacher_logits"] = np.asarray(LG, dtype=np.float32)
    if have_val:
        kw["value"] = np.asarray(VAL, dtype=np.float32)
    np.savez_compressed(out, **kw)
    return out


def build(env_module, out=None, limit=None):
    """그 게임의 궤적 기록을 전부 모아 증류·회복학습 형식 npz 하나로."""
    parts = []
    for f in sorted(glob.glob(os.path.join(RUNS, "*", "run.json"))):
        d = json.load(open(f, encoding="utf-8"))
        if (d.get("params") or {}).get("env") != env_module:
            continue
        p = os.path.join(RUNS, d["run_id"], "궤적.npz")
        if os.path.exists(p):
            parts.append(p)
    if limit:
        parts = parts[-limit:]
    if not parts:
        print(f"★{env_module} 의 궤적 기록이 없다. 먼저 SCOPE 로 굴려야 한다"
              f"(scope.py 검사/기준선 또는 scope_replay.save_trajectory)")
        return None

    acc = {}
    for p in parts:
        z = np.load(p, allow_pickle=True)
        for k in z.files:
            if k == "schema":
                continue
            acc.setdefault(k, []).append(z[k])
    merged = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    n = len(merged["obs"])
    # game_feat — 소비자가 기대하는 칸(없으면 0으로 채운다)
    if "game_feat" not in merged:
        merged["game_feat"] = np.zeros((n, 8), dtype=np.float32)
    merged["schema"] = np.str_("v2_0804")
    merged["n_envs"] = np.int32(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = out or os.path.join(OUT_DIR, f"{env_module}.npz")
    np.savez_compressed(out, **merged)
    eps = int(merged["done"].sum())
    print(f"[모음] {len(parts)}개 기록 → {out}")
    print(f"  표본 {n:,} · 에피소드 {eps:,} · 관측 {merged['obs'].shape[1:]}")
    print(f"  들어있는 항목: {sorted(k for k in merged if k not in ('schema', 'n_envs'))}")
    print(f"  증류(로짓추종) 가능: {'예' if 'teacher_logits' in merged else '아니오 — 회복학습·세계모델용으로는 가능'}")
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "목록"
    if cmd == "목록":
        rows = scan()
        print(f"{'run_id':24s} {'종류':12s} {'게임':24s} 궤적")
        print("-" * 72)
        for r in rows[-25:]:
            print(f"{r['run_id']:24s} {r['종류']:12s} {str(r['게임'])[:24]:24s} "
                  f"{'있음' if r['궤적파일'] else '-'}")
        n = sum(1 for r in rows if r["궤적파일"])
        print(f"\n총 {len(rows)}건 중 궤적 있는 기록 {n}건")
    elif cmd == "만들기":
        if len(sys.argv) < 3:
            raise SystemExit("사용: scope_replay.py 만들기 <env 모듈명>")
        build(sys.argv[2])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
