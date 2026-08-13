"""[9] 코어 해부 — **읽기만** 한다 (2026-08-13).

설계 정본: `restructure/SCOPE_DESIGN_0813.md` §6

> 사용자 08-13: *"내가 말한 코어 분해는 그냥 **읽어 오라는 거야.** 실시간으로 수정하는 게 아니라,
>  코어의 지금 현재 **뭐가 부족하고 지금 파워가 얼마나 나오는지** 정확하게 수치화해서,
>  이런 게임엔 이런 식으로 해야 한다 …그걸로 **학습 데이터를 어떤 식으로 만들지** 체크하고 예측하는 거지."*

두 가지를 **둘 다** 낸다 — 사용자 정정: *"학습이 건강한가 그것도 있어야지. 다 있어야 하는 거야."*

  (가) 학습 건강  — 망가지고 있나 (죽은 뉴런·가중치 변화·정규화 통계)
  (나) 능력 보유  — 뭘 할 줄 아나 (종목별 성능·표현 분해능)

★★함정(V675): 내부 지표는 결과와 직결되지 않을 수 있다. 타깃망이 성능을 8.5%p 올렸는데
  당시 보던 내부 지표는 그걸 못 잡았다. → **결과와 상관이 확인된 것만 판정에 쓴다.**
  확인 전에는 **표시만** 한다(합격·불합격을 매기지 않는다).
"""
import numpy as np
import torch


def _tensors(sd, prefix=""):
    return {k: v for k, v in sd.items()
            if hasattr(v, "numel") and (not prefix or k.startswith(prefix))}


def health(ckpt_path, prev_path=None):
    """(가) 학습 건강 — 가중치만 보고 알 수 있는 것. 굴리지 않는다."""
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = d.get("student", d.get("policy", d.get("kernel", d)))
    if not isinstance(sd, dict):
        return {"오류": "가중치 dict 를 못 찾음"}
    T = _tensors(sd)
    out = {"파라미터": int(sum(v.numel() for v in T.values())), "키수": len(T)}

    # 죽은 유닛 — 출력이 항상 0 이 되는 행(가중치가 통째로 0)
    dead, rows = 0, 0
    for k, v in T.items():
        if v.dim() == 2:
            rows += v.shape[0]
            dead += int((v.abs().sum(dim=1) < 1e-8).sum())
    out["죽은유닛비율"] = round(dead / rows, 4) if rows else 0.0

    # 층별 크기 — 폭주·수축을 본다
    norms = {k: float(v.float().norm()) for k, v in T.items() if v.dim() >= 2}
    if norms:
        vals = np.array(list(norms.values()))
        out["층norm_중앙값"] = round(float(np.median(vals)), 4)
        out["층norm_최대"] = round(float(vals.max()), 4)
        out["층norm_최소"] = round(float(vals.min()), 6)

    # 정규화 통계가 들어 있나(덮이면 눈금이 바뀐다 — 08-13 에 우연히 발견한 것)
    nk = [k for k in sd if ".norm." in k]
    out["정규화키수"] = len(nk)

    # 이전 판과 비교 — **증류·학습이 어디를 바꿨나**
    if prev_path:
        p = torch.load(prev_path, map_location="cpu", weights_only=False)
        psd = p.get("student", p.get("policy", p))
        pT = _tensors(psd)
        changed, same, missing = [], 0, []
        for k, v in T.items():
            if k not in pT:
                missing.append(k)
                continue
            if pT[k].shape != v.shape:
                changed.append((k, float("inf")))
                continue
            dlt = float((v.float() - pT[k].float()).abs().max())
            if dlt > 1e-9:
                changed.append((k, dlt))
            else:
                same += 1
        gone = [k for k in pT if k not in T]
        changed.sort(key=lambda x: -x[1])
        out["비교"] = {
            "바뀐키": len(changed), "그대로": same,
            "새로생긴키": len(missing), "★사라진키": len(gone),
            "사라진키목록": gone[:8],
            "가장크게바뀐": [(k, round(v, 5)) for k, v in changed[:5]],
        }
        # ★사라진 키가 있으면 그것만으로 경보다 — V655(증류가 가치헤드를 지웠다)가 이 형태였다
        out["★경보"] = ("이전 판에 있던 가중치가 사라졌다 — 증류·저장이 지운 것일 수 있다(V655 형태)"
                       if gone else None)
    return out


def capability(core_ckpt, games=None):
    """(나) 능력 보유 — 코어가 아는 종목과 각 종목의 부품 크기."""
    d = torch.load(core_ckpt, map_location="cpu", weights_only=False)
    sd = d.get("student", d)
    heads = sorted({k.split(".")[1] for k in sd if k.startswith("heads.")})
    drivers = sorted({k.split(".")[1] for k in sd if k.startswith("drivers.")})
    per = {}
    for g in heads:
        n = sum(int(v.numel()) for k, v in sd.items()
                if hasattr(v, "numel") and (k.startswith(f"heads.{g}.") or k.startswith(f"drivers.{g}.")))
        w = [v for k, v in sd.items() if k == f"heads.{g}.weight"]
        per[g] = {"고유파라미터": n, "버튼수": int(w[0].shape[0]) if w else None}
    core_n = sum(int(v.numel()) for k, v in sd.items()
                 if hasattr(v, "numel") and k.startswith("core."))
    return {"종목수": len(heads), "종목": heads, "드라이버있는종목": len(drivers),
            "공용코어파라미터": core_n, "종목별": per,
            "공용비율": round(core_n / max(1, sum(int(v.numel()) for v in sd.values()
                                              if hasattr(v, "numel"))), 3)}


def gap(demand, have):
    """(다) 격차 = 게임이 요구하는 것 − 코어가 가진 것.

    demand/have 는 {능력이름: 수치}. **격차가 큰 곳이 다음에 학습 자료를 부어야 할 곳**이다.
    """
    keys = sorted(set(demand) | set(have))
    out = {}
    for k in keys:
        d, h = float(demand.get(k, 0.0)), float(have.get(k, 0.0))
        out[k] = {"요구": round(d, 2), "보유": round(h, 2), "격차": round(d - h, 2)}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["격차"]))
