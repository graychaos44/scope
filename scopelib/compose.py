"""[10] 조합 — 코어 부품을 **분해해서 조합하고, 실제로 굴려 수치를 낸다** (2026-08-14 신설).

> 사용자 08-14: *"시뮬레이션 모델도 준비하고 그걸 스코프에 연결해서 **코어에 있는 것들 분해해서
>  학습하게 해봐**. 그다음 실제로 작동했을 때 효율을 보면 되잖아. **수치화**시키면 보기 편하고."*

## 왜 필요한가 — 같은 부품인데 배치로 21.5%p 가 갈렸다

08-12 실측: 같은 추론 모델을 **대체로 쓰면 −8.73%p**, **보정으로 쓰면 +12.73%p**(V676).
즉 **무엇을 넣느냐보다 어떻게 잇느냐**가 더 크게 갈렸다. 그런데 조합을 손으로 하나씩
돌리면 수십 가지 × 몇 시간이라 사실상 못 해본다.

## ★시뮬레이션이 아니라 '짧게 진짜로 굴린다'

규약 §9-1: *"**시뮬 수치는 30%만 믿는다.** 시뮬로 읽어도 되는 것은 딱 하나 —
'실제 학습에서 0에 고정되는 비율이 좀 덜하다', 즉 **0부터가 아닌 시작을 해볼 수 있다** 그 정도."*

그래서 가짜 모델로 예측하지 않는다. **각 조합을 짧게 실제로 굴려** 잰다.
짧게 잰 값은 **순서를 정하는 데까지만** 쓰고, 판정은 정본 조건(1000판×시드3)으로 다시 한다.

  · 짧게 재기(screen)  = 어느 조합부터 길게 돌릴지 **정렬**하는 용도
  · 길게 재기(final)   = 판정. `autobot_measure.py` 프로파일을 따른다

## 부품 (자동으로 찾는다)

| 부품 | 켜고 끄는 법 | 무엇을 바꾸나 |
|---|---|---|
| 보상담당(RND) | `RND` | 중간 점수를 코어가 스스로 만드나 |
| 심판이 계수 결정 | `JUDGE` | 계수를 사람이 박나 심판이 정하나 |
| **추론 다리** | `REASON_WIRE` | 추론이 심판에게 해석을 올리나 |
| 길잡이 버튼 | 게임 쪽 | 길찾기를 주나 |

★부품이 늘면 `PARTS` 에 한 줄 추가하면 된다. 상한을 두지 않는다(규약 §6 여유).
"""
import itertools
import json
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 부품 정의 — (이름, 환경변수, 꺼짐값, 켜짐값, 설명)
PARTS = [
    ("보상담당", "RND", "0", "1", "중간 점수를 코어(RND)가 스스로 만든다"),
    ("심판결정", "JUDGE", "0", "1", "보상 계수를 심판이 정한다(끄면 사람이 박은 고정값)"),
    ("추론다리", "REASON_WIRE", "0", "1", "추론 11.63M 이 심판에게 해석을 올린다"),
]


def inventory():
    """지금 코어에 무엇이 있나 — 분해해서 목록을 낸다."""
    import torch
    out = []
    cands = [
        ("반사 코어", "checkpoints_kernel/reflex_core.pt"),
        ("L1 코어", "checkpoints_kernel/L1_core.pt"),
        ("심판", "checkpoints_kernel/judge_allocator.pt"),
        ("추론 베이스", "/home/gray/corpora/ckpt/base_A_x10.best.pt"),
        ("둠 비전 정책", "checkpoints_kernel/doom_e1m1_real.pt"),
    ]
    for name, p in cands:
        path = p if os.path.isabs(p) else os.path.join(ROOT, p)
        if not os.path.exists(path):
            out.append({"부품": name, "상태": "없음", "파라미터": 0, "경로": p})
            continue
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)
            sd = d
            for k in ("model", "kernel", "student", "policy", "state_dict"):
                if isinstance(d, dict) and k in d and isinstance(d[k], dict):
                    sd = d[k]
                    break
            n = sum(v.numel() for v in sd.values() if hasattr(v, "numel")) if isinstance(sd, dict) else 0
            out.append({"부품": name, "상태": "있음", "파라미터": int(n), "경로": p})
        except Exception as ex:
            out.append({"부품": name, "상태": f"못읽음({type(ex).__name__})", "파라미터": 0, "경로": p})
    return out


def combos(parts=None, only=None):
    """켜고 끄는 모든 조합. only 를 주면 그 부품만 흔든다(나머지는 켠 채)."""
    ps = parts or PARTS
    if only:
        ps = [p for p in ps if p[0] in only]
    for bits in itertools.product([0, 1], repeat=len(ps)):
        yield {p[0]: b for p, b in zip(ps, bits)}


def _env_for(combo, parts=None):
    ps = {p[0]: p for p in (parts or PARTS)}
    env = dict(os.environ)
    for name, on in combo.items():
        _, var, off_v, on_v, _d = ps[name]
        env[var] = on_v if on else off_v
    return env


def run_one(combo, updates=60, n_envs=6, steps_per_update=256, device="cuda",
            env_name="e1m1", extra_env=None, timeout=1800, max_step=1200):
    """조합 하나를 **짧게 진짜로 굴린다.** 돌려주는 것: 지표 딕셔너리.

    ★거르기용은 **판도 짧게** 잡아야 한다(08-14 실측).
      판 하나가 최대 10,500수인데 25회차(6,400수)로 재니 **8조합 중 6개가 판 0개**로
      아무것도 못 갈랐다. 판이 안 끝나면 클리어도 출구거리도 안 나온다.
      → `max_step` 을 줄여 판이 실제로 끝나게 한다. 판정은 정본 조건으로 다시 한다.
    """
    env = _env_for(combo)
    env["DOOM_MAXSTEP"] = str(max_step)
    env.update(extra_env or {})
    tag = "_".join(f"{k}{v}" for k, v in combo.items())
    ck = f"/tmp/compose_{tag}.pt"
    py = os.path.join(ROOT, "venv_cuda/bin/python") if device == "cuda" \
        else os.path.join(ROOT, "venv/bin/python")
    cmd = [py, "-u", os.path.join(ROOT, "doom_core_train.py"),
           "--env", env_name, "--n_envs", str(n_envs), "--device", device,
           "--steps_per_update", str(steps_per_update),
           "--total_updates", str(updates), "--resume", "0", "--ckpt", ck]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True,
                           text=True, timeout=timeout)
        out = r.stdout
    except subprocess.TimeoutExpired as ex:
        out = (ex.stdout or b"").decode(errors="ignore") if isinstance(ex.stdout, bytes) else (ex.stdout or "")
    wall = time.time() - t0

    # 마지막 진행 줄에서 수치를 뽑는다
    last = [ln for ln in out.splitlines() if "upd=" in ln]
    res = {"조합": tag, "초": round(wall, 1), "회차": 0, "판": 0, "클리어": 0,
           "출구까지최소": None, "평균보상": None}
    if last:
        ln = last[-1]
        import re
        def g(pat, cast=float, d=None):
            m = re.search(pat, ln)
            return cast(m.group(1)) if m else d
        res["회차"] = g(r"upd=(\d+)", int, 0)
        res["판"] = g(r"n_eps=(\d+)", int, 0)
        res["클리어"] = g(r"클리어=(\d+)", int, 0)
        res["출구까지최소"] = g(r"출구까지최소=([\d.]+)", float, None)
        res["평균보상"] = g(r"avg_ep_ret\(20\)=(-?[\d.na]+)", str, None)
    res["초당수"] = round(res["회차"] * steps_per_update / max(wall, 1e-6), 1)
    res["판최대수"] = max_step
    for k, v in combo.items():
        res[k] = "켬" if v else "끔"
    try:
        for p in (ck, ck + ".latest"):
            if os.path.exists(p):
                os.remove(p)
    except Exception:
        pass
    return res
