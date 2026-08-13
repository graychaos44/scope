"""[4] 개입 실험 — *"만약 이랬으면?"* 을 답한다 (2026-08-13, 뼈대로 옮김).

[3] 분석기는 **기록만 읽는다**. 그러나 *"화면을 가렸으면 어땠을까"* 는 기록으로 못 답한다 —
**다시 굴려야** 한다. 그 자리가 여기다.

옮겨온 것: `adapter_capability_probe.py`(08-12) 의 알맹이.
달라진 것: 굴리는 일을 **샌드박스에 맡긴다**(조건 지문이 한 곳에 모인다).

## 재는 것

| 건드림 | 많이 떨어지면 |
|---|---|
| 화면가림 | 화면을 본다(지각) |
| 버튼씹힘 | 타이밍이 중요하다 (Machado 2018 sticky actions) |
| 한수지연 | 반응 속도로 푼다 |
| 화면잡음 | 정확한 값이 필요하다 |
| 칸섞기 | 그 칸을 실제로 본다 (permutation importance) |

## ★★두 가지를 같이 봐야 한다

08-12 에 하마터면 거꾸로 읽을 뻔했다 — '안 보고도 이기는 게임' 에 어댑터를 대면
화면 가릴 때 96.7%p 나 떨어진다. 그런데 그 게임은 **상수 행동으로도 100%** 라 지각이 필요 없다.

  능력 = ①하락이 크다 **그리고** ②어댑터가 기준선(무작위·상수)보다 낫다

②가 아니면 하락폭은 능력의 증거가 아니다.
"""
import numpy as np
import torch

from . import sandbox as SB


def load_policy(ckpt, obs_dim, n_act):
    """체크포인트에서 정책을 만든다. 표집(sample)로 고른다 — 결정규칙을 조건에 적는다(M25)."""
    import policy_filter as PF
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("policy", raw.get("kernel", raw))
    net = PF.build_network(sd, obs_dim, n_act)

    def pol(o, _n):
        t = torch.from_numpy(np.asarray(o, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            lg, _ = net(t)
        return int(torch.multinomial(torch.softmax(lg[0], -1), 1).item())
    return pol


KINDS = [("화면가림", {}), ("버튼씹힘", {"p": 0.25}), ("한수지연", {}), ("화면잡음", {"sigma": 0.1})]


def probe(module, ckpt, seeds, episodes, cells=True):
    """어댑터의 능력 프로파일. 기준선(무작위·상수)도 같이 잰다."""
    C = SB.env_class(module)
    e = C(seed=0)
    obs_dim = len(np.asarray(e.reset(), dtype=np.float32))
    n_act = int(e.n_act)
    pol = load_policy(ckpt, obs_dim, n_act)

    base = SB.rollout(module, pol, seeds, episodes, None, record=False)["요약"]["승률"]
    rnd = SB.rollout(module, None, seeds, episodes, SB.Intervention("무작위"), record=False)["요약"]["승률"]
    const = max(
        SB.rollout(module, None, seeds[:1], episodes, SB.Intervention("상수행동", action=a),
                   record=False)["요약"]["승률"]
        for a in range(n_act))
    edge = base - max(rnd, const)

    caps = {}
    for kind, kw in KINDS:
        w = SB.rollout(module, pol, seeds, episodes, SB.Intervention(kind, **kw),
                       record=False)["요약"]["승률"]
        caps[kind] = {"승률": round(w, 2), "하락": round(base - w, 2)}

    cellimp = []
    if cells:
        pool = SB.cell_pool(module)
        for c in range(obs_dim):
            w = SB.rollout(module, pol, seeds[:1], episodes,
                           SB.Intervention("칸섞기", cell=c, pool=pool[c]),
                           record=False)["요약"]["승률"]
            cellimp.append({"칸": c, "하락": round(base - w, 2)})
        cellimp.sort(key=lambda x: -x["하락"])

    return {"그대로": round(base, 2), "무작위": round(rnd, 2), "상수최고": round(const, 2),
            "기준선대비": round(edge, 2), "능력": caps, "칸별중요도": cellimp,
            "조건": {"모듈": module, "체크포인트": ckpt, "판수": episodes,
                    "시드": list(seeds), "결정규칙": "sample"},
            "믿을수있나": edge > 5}
