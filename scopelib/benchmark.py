"""[11] 벤치마크 — **어디가 얼마나 빠른지 한 자로 잰다** (2026-08-14 신설).

> 사용자 08-13: *"작은 벤치마크"* → 08-14: *"스코프에 벤치마크 비스무리한 거 달아 달라고 했었는데."*

## 왜 필요한가 — 08-13 에 병목을 두 번 잘못 짚었다

- "병렬 8개" 라고 적혀 있어서 병렬인 줄 알았다. 실제로는 게임이 **한 줄로** 돌고 있었다
- 게임을 2.5배 빠르게 고쳤는데 **전체는 하나도 안 변했다** — 진짜 병목은 신경망(96%)이었다
- CPU→GPU 로 바꾸니 **17배**. 규약(§8 "학습은 GPU 우선")을 지켰으면 처음부터 안 겪었다

★**병목을 고치기 전에 어디가 병목인지부터 재야 한다.** 이 도구가 그 자리다.

## 무엇을 재나

| 항목 | 무엇을 알려주나 |
|---|---|
| 게임 | 초당 몇 수를 굴릴 수 있나 (게임마다) |
| 신경망 | 초당 몇 수를 처리하나 (CPU / GPU) |
| 기록 | SCOPE 가 남기는 비용 |
| **판정** | 셋 중 **누가 병목인가** — 이것이 결론이다 |

★값은 기기마다 다르다. **같은 기기에서 잰 값끼리만** 비교한다(조건 지문에 기기 이름이 들어간다).
"""
import os
import platform
import socket
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bench_env(module, steps=20000, seed=1):
    """게임 하나를 굴려 초당 수를 잰다."""
    from scopelib.sandbox import env_class
    C = env_class(module)
    e = C(seed=seed)
    o = e.reset()
    rng = np.random.RandomState(0)
    n_act = int(e.n_act)
    t0 = time.time()
    n = 0
    for _ in range(steps):
        o, _r, dn, _i = e.step(int(rng.randint(n_act)))
        n += 1
        if dn:
            o = e.reset()
    dt = time.time() - t0
    if hasattr(e, "close"):
        try:
            e.close()
        except Exception:
            pass
    return {"항목": f"게임:{module}", "초당수": round(n / dt, 1),
            "한수us": round(dt / n * 1e6, 2), "표본": n}


def bench_net(obs_dim=15, n_act=5, batch=16, device="cpu", iters=2000, image=None):
    """신경망 앞으로 계산 속도. image 를 주면 (ch,h,w) 화면 입력으로 잰다."""
    import torch
    import torch.nn as nn
    dev = torch.device(device)
    if device == "cpu":
        torch.set_num_threads(1)
    if image:
        import core_kernel as ck
        net = ck.FPSInputDriver(in_ch=image[0]).to(dev)
        x = torch.randn(batch, *image, device=dev)
        label = f"신경망:화면{image[0]}x{image[1]}x{image[2]}"
    else:
        net = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(),
                            nn.Linear(256, 256), nn.ReLU(),
                            nn.Linear(256, n_act)).to(dev)
        x = torch.randn(batch, obs_dim, device=dev)
        label = f"신경망:숫자{obs_dim}칸"
    with torch.no_grad():
        for _ in range(5):
            net(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            net(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
    per = dt / iters / batch
    return {"항목": f"{label}({device})", "초당수": round(1 / per, 1),
            "한수us": round(per * 1e6, 2), "표본": iters * batch}


def bench_record(dim=15, n=20000):
    """SCOPE 가 매 수를 남기는 비용."""
    from scopelib.sandbox import obs_digest
    o = np.random.rand(dim).astype(np.float32)
    t0 = time.time()
    for _ in range(n):
        np.asarray(o, dtype=np.float32).tolist()
    dt1 = time.time() - t0
    img = np.random.rand(4, 84, 84).astype(np.float32)
    m = max(200, n // 20)
    t0 = time.time()
    for _ in range(m):
        obs_digest(img)
    dt2 = time.time() - t0
    return [
        {"항목": "기록:숫자관측", "초당수": round(n / dt1, 1),
         "한수us": round(dt1 / n * 1e6, 2), "표본": n},
        {"항목": "기록:화면요약", "초당수": round(m / dt2, 1),
         "한수us": round(dt2 / m * 1e6, 2), "표본": m},
    ]


def verdict(rows):
    """★결론 — 누가 병목인가. 한 수당 시간이 가장 큰 것이 병목이다."""
    cand = [r for r in rows if r["항목"].startswith(("게임", "신경망"))]
    if not cand:
        return "판정불가(잰 것이 없다)"
    slow = max(cand, key=lambda r: r["한수us"])
    others = [r for r in cand if r is not slow]
    if not others:
        return f"병목 = {slow['항목']} ({slow['한수us']}us/수)"
    second = max(others, key=lambda r: r["한수us"])
    ratio = slow["한수us"] / max(second["한수us"], 1e-9)
    if ratio < 1.3:
        return (f"병목이 뚜렷하지 않다 — {slow['항목']} {slow['한수us']}us 와 "
                f"{second['항목']} {second['한수us']}us 가 비슷하다(비 {ratio:.2f})")
    return (f"★병목 = **{slow['항목']}** ({slow['한수us']}us/수) · "
            f"다음은 {second['항목']} ({second['한수us']}us) · **{ratio:.1f}배 차이**")


def machine():
    import torch
    return {"기기": socket.gethostname(), "cpu": platform.processor() or platform.machine(),
            "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음"}
