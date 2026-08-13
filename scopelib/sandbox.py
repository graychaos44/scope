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

## 화면(이미지) 게임도 받는다 (08-13 추가 — 규약 §9-6 "둠은 무조건 비전")

환경이 `is_image = True` 를 달면 관측이 **화면 묶음**이다. 그때는 두 가지가 달라진다.

  · **매 수의 화면을 통째로 안 남긴다.** 84x84x4 한 장이 113KB 라 1000판이면 수십 GB 다.
    대신 **화면 요약**(평균·표준편차·밝은칸비율·앞뒤 프레임 차이)만 남긴다.
    ★분석기는 그 요약으로 판정한다 — 픽셀 하나하나를 '죽은 칸' 으로 세는 것은 뜻이 없다.
  · **판이 끝나면 환경을 닫는다.** 둠은 판마다 게임 프로세스를 띄우므로
    안 닫으면 프로세스가 쌓인다(리뷰B 지적).

## 강화학습 재료로도 쓸 수 있다 (08-13 추가 — 사용자 질문)

> *"스코프가 나중에 강화학습도 호환되게 가능하나? 코어 내부 모델들이 신경망으로 게임하면서
>  거기서 나오는 로그나 실시간 플레이를 하면서 강화학습 돼?"*

**된다.** 다만 그러려면 기록에 **네 가지가 더** 있어야 한다 —
학습용 데이터 정본(`ADAPTER_DATA_SPEC_0805.md`)이 요구하는 것 중 빠져 있던 항목이다.

  · `logits`   행동 확률 → 증류(로짓 추종). 가중치 평균이 아니라 이게 정본(규약 §6)
  · `next_obs` 다음 관측 → **세계모델**(행동 → 다음 상태) · 오프라인 강화학습
  · `value`    정책이 낸 상태가치 → 심판 대조군
  · `ep_id`    판 번호 → 시퀀스 복원(스텝 단위로 쪼개면 못 쓴다)

정책이 `(행동, {"logits":..., "value":...})` 형태로 답하면 그대로 받아 적는다.
행동만 답해도 예전처럼 돌아간다 — **옛 호출을 안 깨뜨린다.**

★화면 게임은 관측이 커서 기본은 요약만 남긴다. `FULL_OBS=1` 이면 원본을 남긴다
  (84x84x4 한 장이 113KB — 1000판이면 수십 GB 라 켤 때 용량을 먼저 보라).
"""
import copy
import importlib
import os

import numpy as np

MAXSTEP = int(os.environ.get("MAXSTEP", "400"))
FULL_OBS = os.environ.get("FULL_OBS", "0") == "1"   # 화면 게임에서 원본 관측을 남길까


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
            if o.ndim > 1:
                # 화면에서는 '칸' 이 픽셀 하나가 아니라 **프레임 한 장**이다
                i = int(self.kw["cell"]) % o.shape[0]
                o[i] = 0.0
                return o
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


# ★뜨거운 부분만 컴파일판을 쓴다. 없으면 파이썬판으로 떨어진다(08-14).
#   실측: 화면 요약은 1.45배 빨라지고, 관측 변환은 **오히려 0.82배로 느려서 안 쓴다** —
#   이미 numpy(C)를 부르는 코드는 컴파일해도 바꿀 게 없다.
try:
    from scopelib import fastcore as _FC
except Exception:
    _FC = None


def obs_digest(o):
    """화면 한 묶음을 **몇 개의 숫자**로 줄인다 — 기록에 남길 것.

    ★왜 요약인가: 84x84x4 화면 하나가 113KB 다. 1000판×100수면 11GB 가 된다.
      분석기가 실제로 보는 것은 '변하나 · 범위가 맞나 · 움직임이 있나' 뿐이라
      아래 넷이면 충분하다.
    """
    if _FC is not None:
        return _FC.obs_digest(o)
    a = np.asarray(o, dtype=np.float32)
    flat = a.reshape(a.shape[0], -1) if a.ndim > 1 else a.reshape(1, -1)
    d = [float(a.mean()), float(a.std()), float(a.min()), float(a.max()),
         float((a > 0.05).mean())]                       # 밝은 칸 비율
    if flat.shape[0] > 1:                                # 프레임끼리 얼마나 다른가 = 움직임
        d.append(float(np.abs(flat[-1] - flat[0]).mean()))
    else:
        d.append(0.0)
    return d


def save_trajectory(rec, run_id):
    """굴린 기록을 **증류·회복학습이 읽는 형식**으로 그 자리에 저장한다 (08-13 신설).

    > 사용자: *"스코프로 학습데이터까지 로그로 남는다면 나중에 증류할 때 재학습용으로 돌리면 되지 않나?"*

    맞다. 그래서 굴릴 때마다 궤적을 남긴다 — **재기만 해도 재료가 쌓인다.**
    규약 §6 의 "증류 → 재학습(회복)이 한 세트" 가 재료 부족으로 08-12 에야 처음 돌았다.
    """
    from scope_replay import save_trajectory as _st
    return _st(run_id, rec)


def rollout(module, policy, seeds, episodes, intervention=None, record=True, max_step=MAXSTEP):
    """굴리면서 **매 수를 남긴다.**

    policy: (obs, n_act) -> action.  None 이면 개입이 행동을 정해야 한다(상수·무작위).
    돌려주는 것: {"episodes":[...], "요약":{...}}  — episodes[i] 는 한 판의 기록
    """
    C = env_class(module)
    probe = C(seed=0)
    n_act = int(probe.n_act)
    is_image = bool(getattr(C, "is_image", False))
    iv = intervention or Intervention("none")

    eps, wins, lens, truncs = [], 0, [], 0
    for s in seeds:
        for i in range(episodes):
            e = C(seed=int(s) * 1000 + i)
            iv.rng = np.random.RandomState(int(s) * 7919 + i)   # 판마다 다르게, 시드로 재현 가능
            o = iv.on_reset(e.reset())
            steps, cut, info = [], True, {}
            for t in range(max_step):
                raw = policy(o, n_act) if policy else 0
                # ★정책이 (행동, 부가정보) 로 답하면 학습 재료를 같이 받아 적는다.
                #   행동만 답해도 그대로 돈다 — 옛 호출을 안 깨뜨린다.
                extra = {}
                if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict):
                    raw, extra = raw
                a = iv.act(raw, n_act)
                o2, r, dn, info = e.step(a)
                o2 = iv.obs(o2)
                if record:
                    # ★화면 게임은 기본이 요약이다. FULL_OBS=1 이면 원본을 남긴다.
                    def _enc(x):
                        if is_image and not FULL_OBS:
                            return obs_digest(x)
                        return np.asarray(x, dtype=np.float32).tolist()
                    row = {"ep_id": len(eps), "t": t, "obs": _enc(o),
                           "a": int(a), "r": float(r), "done": bool(dn),
                           "next_obs": _enc(o2)}
                    for k in ("logits", "value"):
                        if k in extra:
                            v = extra[k]
                            row[k] = (np.asarray(v, dtype=np.float32).tolist()
                                      if hasattr(v, "__len__") else float(v))
                    steps.append(row)
                o = o2
                if dn:
                    cut = False
                    break
            won = bool(info.get("won")) if isinstance(info, dict) else False
            has_won_key = isinstance(info, dict) and "won" in info
            eps.append({"ep_id": len(eps), "seed": int(s), "i": i, "steps": steps, "won": won,
                        "end_reason": ("시간초과" if cut else ("승" if won else "패")),
                        "won_key": has_won_key, "cut": cut, "len": len(steps),
                        "score": (info.get("score") if isinstance(info, dict) else None)})
            wins += 1 if won else 0
            lens.append(len(steps))
            truncs += 1 if cut else 0
            # ★판마다 닫는다 — 둠은 판마다 게임 프로세스가 뜬다(리뷰B 지적)
            if hasattr(e, "close"):
                try:
                    e.close()
                except Exception:
                    pass

    n = len(eps)
    obs_cells = (len(obs_digest(probe.reset())) if is_image
                 else len(np.asarray(probe.reset(), dtype=np.float32)))
    if hasattr(probe, "close"):
        try:
            probe.close()
        except Exception:
            pass
    return {
        "episodes": eps,
        "요약": {"판수": n, "승률": wins / n * 100 if n else 0.0,
                "평균길이": float(np.mean(lens)) if lens else 0.0,
                "미종료": truncs, "관측칸": obs_cells, "화면게임": is_image,
                "버튼수": n_act,
                # ★화면 게임은 요약칸이라 예약칸 개념이 없다
                "예약칸": ([] if is_image
                          else [int(x) for x in (getattr(probe, "obs_reserved", []) or [])])},
        "조건": {"모듈": module, "시드": list(seeds), "판수/시드": episodes,
                "최대수": max_step, "원본관측": FULL_OBS, **iv.fingerprint()},
    }


def cell_pool(module, n=300, seed=7):
    """칸섞기에 쓸 '다른 판의 값' — 아무 값이나 넣으면 범위 밖이 된다."""
    C = env_class(module)
    if bool(getattr(C, "is_image", False)):
        return None          # 화면 게임은 프레임 한 장을 지우는 방식이라 값 풀이 필요 없다
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
