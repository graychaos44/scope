"""[3] 분석기 — **기록만 읽고** 판정한다 (2026-08-13 신설).

뼈대 설계: `restructure/GATE_SKELETON_0813.md`

★핵심: 여기 있는 검사들은 **게임을 굴리지 않는다.** 샌드박스가 남긴 기록을 읽을 뿐이다.
  그래서 한 번 굴린 것으로 여섯 가지를 다 본다(예전에는 검사마다 따로 굴렸다).
★게임이 무엇인지 몰라도 된다 — 그래서 **새 장르가 와도 그대로 걸린다**(M7: 사람이 100번 손대는 구조 금지).

  NOWON     이겼는지 안 알려준다        승률이 조용히 0 이 된다
  NOWIN?    won 은 있는데 늘 False      어려운 것인지 못 재는 것인지 안 갈린다
  DEADCELL  관측 칸이 내내 안 변한다     학습이 그 칸을 못 쓴다
  BLANKSCREEN 화면이 내내 한 색이다      (화면 게임 전용) 그림이 안 들어온다
  NOMOTION  쌓은 프레임이 다 똑같다      (화면 게임 전용) 움직임이 관측에 안 담긴다
  RANGE     관측 값이 규격을 크게 넘는다  교사가 못 배우는 원인이었다(V645)
  EPLEN     판이 너무 짧다             첫 수부터 끝낼 수 있다는 뜻
  NOTERM    판이 안 끝난다             학습이 멈춘다

개입이 필요한 검사(상수·무관측·버튼영향)는 여기가 아니라 **[4] 개입**이 맡는다 —
*"만약 이랬으면?"* 은 다시 굴려야 답하기 때문이다.
"""
import numpy as np

RANGE_CAP = 10.0        # 정상 게임(chase) 실측 최대가 1.0 이라 넉넉한 여유
EPLEN_MIN = 5.0         # 규약 §3


def _obs_matrix(rec):
    rows = [s["obs"] for e in rec["episodes"] for s in e["steps"]]
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 1), dtype=np.float32)


def analyze(rec):
    """기록 하나를 읽어 판정 목록을 낸다. 각 판정은 (이름, 설명, 수치)."""
    out = []
    eps = rec["episodes"]
    S = rec["요약"]
    O = _obs_matrix(rec)

    # ── 승패 보고 ────────────────────────────────────────
    ended = [e for e in eps if not e["cut"]]
    no_key = [e for e in ended if not e["won_key"]]
    if no_key:
        out.append(("NOWON", f"제대로 끝난 판 {len(ended)}개 중 {len(no_key)}개가 won 키를 안 준다 "
                             f"→ 승률이 조용히 0 이 된다", len(no_key)))
    elif ended and not any(e["won"] for e in ended):
        out.append(("NOWIN?", f"제대로 끝난 판 {len(ended)}개가 전부 won=False — "
                              f"어려운 것인지 못 재는 것인지 갈리지 않는다(규약 §9-3)", 0))

    # ── 판 길이 ─────────────────────────────────────────
    if S["평균길이"] < EPLEN_MIN:
        out.append(("EPLEN", f"한 판이 평균 {S['평균길이']:.1f}수 — 첫 수부터 끝낼 수 있다는 뜻(규약 §3)",
                    S["평균길이"]))
    if S["미종료"]:
        out.append(("NOTERM", f"{S['판수']}판 중 {S['미종료']}판이 최대 수를 다 쓰도록 안 끝났다",
                    S["미종료"]))

    # ── 행동 쏠림(정책 붕괴) ─────────────────────────────
    # ★08-03 에 둠 정책이 **제자리에서 회전만** 했다(80스텝 이동 0, 회전 23회). 그때는 손으로 셌다.
    #   한 버튼이 대부분을 차지하면 정책이 무너진 것이다 — 무작위 정책에는 안 나온다.
    acts = [s2["a"] for e in eps for s2 in e["steps"]]
    if acts:
        import collections as _c
        cnt = _c.Counter(acts)
        n = len(acts)
        top_a, top_n = cnt.most_common(1)[0]
        share = top_n / n
        used = len(cnt)
        # 골고루 쓰면 1/버튼수 근처. 0.8 넘게 한쪽이면 쏠린 것
        if share > 0.8 and used > 1:
            out.append(("ACTBIAS", f"버튼 {top_a} 하나가 전체의 {share*100:.0f}% — 정책이 무너진 형태다"
                                   f"(쓴 버튼 {used}/{S['버튼수']})", share))
        elif used == 1:
            out.append(("ACTBIAS", f"버튼 {top_a} **하나만** 쓴다 — 완전히 굳었다", 1.0))

    # ── 화면 게임은 칸 단위 검사를 안 한다 (08-13, 규약 §9-6) ───────
    #   기록에 남은 것은 픽셀이 아니라 **화면 요약** 여섯 칸이다
    #   [평균, 표준편차, 최소, 최대, 밝은칸비율, 프레임차이].
    #   픽셀 하나를 '죽은 칸' 으로 세는 것은 뜻이 없으므로, 대신 화면이 살아 있는지만 본다.
    if S.get("화면게임") and len(O):
        if O.shape[1] >= 6:
            # ★평균이 아니라 **비율**로 본다.
            #   평균으로 재면 정상인 첫 수 몇 개가 섞여 희석돼 못 잡는다(08-13 교정에서 발각).
            blank = float((O[:, 1] < 1e-4).mean())
            still = float((O[:, 5] < 1e-5).mean())
            if blank > 0.9:
                out.append(("BLANKSCREEN", f"수의 {blank*100:.0f}% 에서 화면이 한 가지 색이다 "
                                           f"— 그림이 안 들어오고 있다", blank))
            if still > 0.9:
                out.append(("NOMOTION", f"수의 {still*100:.0f}% 에서 쌓은 프레임이 서로 똑같다 "
                                        f"— 움직임이 관측에 안 담긴다"
                                        f"(프레임 쌓기가 고장났거나 게임이 안 진행된다)", still))
            if float(O[:, 3].max()) > 1.5:
                out.append(("RANGE", f"화면 값이 {float(O[:, 3].max()):.2f} — 0~1 로 정규화가 안 됐다",
                            float(O[:, 3].max())))
        return out

    # ── 죽은 칸 ─────────────────────────────────────────
    if len(O):
        sd = O.std(axis=0)
        reserved = set(S.get("예약칸", []))
        dead = [int(i) for i in np.where(sd < 1e-6)[0] if int(i) not in reserved]
        if dead:
            head, more = dead[:8], (f" 외 {len(dead)-8}칸" if len(dead) > 8 else "")
            note = " ★단 판이 짧아 변할 틈이 없던 것일 수 있다" if S["평균길이"] < 10 else ""
            out.append(("DEADCELL", f"칸 {head}{more} 이(가) {len(O):,}수 내내 안 변한다{note}", len(dead)))

        # ── 값 범위 ──────────────────────────────────────
        amax = float(np.abs(O).max())
        if amax > RANGE_CAP:
            bad = [int(i) for i in np.where(np.abs(O).max(axis=0) > RANGE_CAP)[0]]
            out.append(("RANGE", f"관측 최대 절대값 {amax:.1f} > 상한 {RANGE_CAP:.0f} · 칸 {bad[:8]}", amax))
    return out


def metrics(rec):
    """실행 대장에 남길 지표 — 이름은 `카테고리.항목`, 단위는 부르는 쪽이 준다."""
    S = rec["요약"]
    O = _obs_matrix(rec)
    m = {
        "env.obs_dim": float(S["관측칸"]),
        "env.n_act": float(S["버튼수"]),
        "env.ep_len_mean": float(S["평균길이"]),
        "env.timeout_count": float(S["미종료"]),
        "result.winrate": float(S["승률"]),
    }
    if len(O):
        sd = O.std(axis=0)
        reserved = set(S.get("예약칸", []))
        m["env.dead_cells"] = float(sum(1 for i in range(len(sd))
                                        if sd[i] < 1e-6 and i not in reserved))
        m["env.obs_absmax"] = float(np.abs(O).max())
    acts = [s2["a"] for e in rec["episodes"] for s2 in e["steps"]]
    if acts:
        import collections as _c
        cnt = _c.Counter(acts)
        m["policy.top_action_share"] = float(cnt.most_common(1)[0][1] / len(acts))
        m["policy.actions_used"] = float(len(cnt))
    scores = [e["score"] for e in rec["episodes"] if e.get("score") is not None]
    if scores:
        m["result.score_mean"] = float(np.mean(scores))
    return m


UNITS = {
    "env.obs_dim": "count", "env.n_act": "count", "env.ep_len_mean": "steps",
    "env.timeout_count": "count", "result.winrate": "percent_0_100",
    "env.dead_cells": "count", "env.obs_absmax": "abs_value",
    "result.score_mean": "points",
    "policy.top_action_share": "ratio_0_1", "policy.actions_used": "count",
    "baseline.random_winrate": "percent_0_100",
    "baseline.const_best": "percent_0_100",
    "baseline.blind_winrate": "percent_0_100",
    "edge.vs_random": "percent_point", "edge.vs_const": "percent_point",
}
