"""[1] 샌드박스 — 게임을 굴리며 **모든 것을 기록**한다 (2026-08-13 신설).

뼈대 설계: `restructure/GATE_SKELETON_0813.md`

## 왜 있나

지금까지는 **검사마다 게임을 따로 굴렸다.** 검사 5종이면 같은 게임을 5번 플레이했고,
각 도구가 자기가 볼 것만 남기고 나머지를 버렸다. 그래서 새 검사를 만들면 처음부터 다시 돌려야 했다.

  한 번 굴리고 → 매 수를 전부 남기고 → 분석기들이 그 기록을 나눠 읽는다.

## 개입(가정형)도 여기서 한다

*"무엇이 일어났나"* 는 기록으로 답하지만, *"만약 이랬으면?"* 은 **다시 굴려야** 답한다.
그래서 샌드박스가 개입 훅을 같이 갖는다 — 조건 지문이 한 곳에 있어야 비교가 성립한다.

  없음 · 화면가림 · 화면잡음 · 버튼씹힘 · 한수지연 · 칸섞기 · 상수행동 · 무작위

★08-13 에 상수 기준선을 빠뜨려 판정이 뒤집혔다(V702). 개입을 **한 곳에 모아** 그것을 막는다.
"""
import copy
import importlib
import os

import numpy as np

MAXSTEP = int(os.environ.get("MAXSTEP", "400"))


def env_class(module):
    m = importlib.import_module(module)
    names = [n for n in dir(m) if n.endswith("Env")]
    if not names:
        raise ValueError(f"{module} 에 이름이 Env 로 끝나는 클래스가 없다")
    return getattr(m, names[-1])


class Intervention:
    """게임을 한 가지만 건드린다. 이름과 설정이 그대로 조건 지문에 들어간다."""

    def __init__(self, kind="none", **kw):
        self.kind = kind
        self.kw = kw
        self.rng = np.random.RandomState(kw.get("seed", 12345))
        self.prev_obs = None
        self.prev_act = 0

    def on_reset(self, obs):
        self.prev_obs = None
        self.prev_act = 0
        return self.obs(obs)

    def obs(self, o):
        o = np.asarray(o, dtype=np.float32).copy()
        k = self.kind
        if k == "화면가림":
            return np.zeros_like(o)
        if k == "화면잡음":
            return o + self.rng.randn(*o.shape).astype(np.float32) * self.kw.get("sigma", 0.1)
        if k == "한수지연":
            out = self.prev_obs if self.prev_obs is not None else np.zeros_like(o)
            self.prev_obs = o
            return out
        if k == "칸섞기":
            i = self.kw["cell"]
            pool = self.kw.get("pool")
            o[i] = float(self.rng.choice(pool)) if pool is not None else 0.0
            return o
        return o

    def act(self, a, n_act):
        k = self.kind
        if k == "버튼씹힘" and self.rng.rand() < self.kw.get("p", 0.25):
            a = self.prev_act
        elif k == "상수행동":
            a = self.kw["action"]
        elif k == "무작위":
            a = int(self.rng.randint(n_act))
        self.prev_act = a
        return a

    def fingerprint(self):
        return {"개입": self.kind, **{f"개입.{k}": v for k, v in self.kw.items()
                                     if k != "pool" and not isinstance(v, np.ndarray)}}


def rollout(module, policy, seeds, episodes, intervention=None, record=True, max_step=MAXSTEP):
    """굴리면서 **매 수를 남긴다.**

    policy: (obs, n_act) -> action.  None 이면 개입이 행동을 정해야 한다(상수·무작위).
    돌려주는 것: {"episodes":[...], "요약":{...}}  — episodes[i] 는 한 판의 기록
    """
    C = env_class(module)
    probe = C(seed=0)
    n_act = int(probe.n_act)
    iv = intervention or Intervention("none")

    eps, wins, lens, truncs = [], 0, [], 0
    for s in seeds:
        for i in range(episodes):
            e = C(seed=int(s) * 1000 + i)
            iv.rng = np.random.RandomState(int(s) * 7919 + i)   # 판마다 다르게, 시드로 재현 가능
            o = iv.on_reset(e.reset())
            steps, cut, info = [], True, {}
            for t in range(max_step):
                a = iv.act(policy(o, n_act) if policy else 0, n_act)
                o2, r, dn, info = e.step(a)
                o2 = iv.obs(o2)
                if record:
                    steps.append({"t": t, "obs": np.asarray(o, dtype=np.float32).tolist(),
                                  "a": int(a), "r": float(r), "done": bool(dn)})
                o = o2
                if dn:
                    cut = False
                    break
            won = bool(info.get("won")) if isinstance(info, dict) else False
            has_won_key = isinstance(info, dict) and "won" in info
            eps.append({"seed": int(s), "i": i, "steps": steps, "won": won,
                        "won_key": has_won_key, "cut": cut, "len": len(steps),
                        "score": (info.get("score") if isinstance(info, dict) else None)})
            wins += 1 if won else 0
            lens.append(len(steps))
            truncs += 1 if cut else 0

    n = len(eps)
    return {
        "episodes": eps,
        "요약": {"판수": n, "승률": wins / n * 100 if n else 0.0,
                "평균길이": float(np.mean(lens)) if lens else 0.0,
                "미종료": truncs, "관측칸": len(np.asarray(probe.reset(), dtype=np.float32)),
                "버튼수": n_act,
                "예약칸": [int(x) for x in (getattr(probe, "obs_reserved", []) or [])]},
        "조건": {"모듈": module, "시드": list(seeds), "판수/시드": episodes,
                "최대수": max_step, **iv.fingerprint()},
    }


def cell_pool(module, n=300, seed=7):
    """칸섞기에 쓸 '다른 판의 값' — 아무 값이나 넣으면 범위 밖이 된다."""
    C = env_class(module)
    rng = np.random.RandomState(seed)
    probe = C(seed=0)
    dim = len(np.asarray(probe.reset(), dtype=np.float32))
    pool = [[] for _ in range(dim)]
    for i in range(30):
        e = C(seed=90000 + i)
        o = e.reset()
        for _ in range(max(1, n // 30)):
            v = np.asarray(o, dtype=np.float32).ravel()
            for c in range(dim):
                pool[c].append(float(v[c]))
            o, _r, dn, _ = e.step(rng.randint(int(e.n_act)))
            if dn:
                break
    return [np.array(p if p else [0.0]) for p in pool]
