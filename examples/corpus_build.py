"""[SCOPE 흡수 안내 — 2026-08-13]
SCOPE(`scope.py` · `scopelib/`)가 같은 일을 한다. 새 작업은 SCOPE 를 쓸 것.
이 파일은 **회귀 대조용으로 남겨둔다**(규약 §7 — 잘 돌아가는 것은 건드리지 않는다).
SCOPE 가 같은 값을 내는 것이 확인되면 archive/ 로 내린다. 대응표는 restructure/SCOPE_DESIGN_0813.md §7.
"""
"""검사 도구 비교용 **정답지** 만들기 (2026-08-12).

규약 §9-2 *"관문을 검증하려면 답을 내가 만든 것으로 잰다"*.
정상 게임(chase) 하나를 상속해 **한 군데만** 망가뜨린 게임 10종과, 안 망가뜨린 정상 2종을 만든다.
한 군데만 바꾸는 것이 핵심이다 — 두 군데를 바꾸면 어느 것을 잡았는지 갈리지 않는다(M30).

★도구마다 게임을 찾는 방식이 다르다(모듈명으로 import 해서 이름이 Env 로 끝나는 마지막 클래스를
고르는 도구가 있다). 그래서 **파일 하나에 게임 하나**로 만든다.

실행: OMP_NUM_THREADS=1 venv/bin/python examples/corpus_build.py
결과: gcorp_*.py 12개 + corpus_manifest.tsv(무엇을 심었는지 정답지)
"""
import os

ROOT = "/home/gray/rl_server"
HDR = '''"""검사 도구 비교용 시험 게임 — {desc}
자동 생성: examples/corpus_build.py (2026-08-12). 손으로 고치지 말 것."""
import numpy as np
from chase_env import ChaseEnv


'''

# (파일키, 클래스명, 한 줄 설명, 정답 라벨, 본문)
SPECS = [
    ("n1_normal", "GcorpN1NormalEnv", "정상 — 아무것도 안 바꿈", "정상", """
class {cls}(ChaseEnv):
    pass
"""),
    ("n2_normal_short", "GcorpN2NormalShortEnv", "정상 — 판만 짧다(60수). 결함 아님", "정상", """
class {cls}(ChaseEnv):
    def __init__(self, horizon=60, seed=None):
        super().__init__(horizon=horizon, seed=seed)
"""),
    ("d01_constwin", "GcorpD01ConstwinEnv", "버튼 4번만 누르면 바로 이긴다", "상수행동승리", """
class {cls}(ChaseEnv):
    def step(self, a):
        o, r, dn, inf = super().step(a)
        if a == 4:                      # ★심은 결함: 특정 버튼 하나가 즉시 승리
            return o, r + 10.0, True, {{"won": True}}
        return o, r, dn, inf
"""),
    ("d02_blindwin", "GcorpD02BlindwinEnv", "게임을 안 보고 고정 순서만으로 이긴다", "무관측승리", """
class {cls}(ChaseEnv):
    def reset(self):
        o = super().reset()
        # ★심은 결함: 시작 자리와 목표가 늘 같다 → 관측을 볼 필요가 없다
        self.pos = np.array([0.0, 0.0], dtype=np.float32)
        self.prey = np.array([3.0, 0.0], dtype=np.float32)
        self.prev_dist = float(np.linalg.norm(self.prey - self.pos))
        return self._obs()

    def _prey_move(self):
        return                          # 목표가 안 움직인다
"""),
    ("d03_deadcell", "GcorpD03DeadcellEnv", "관측 마지막 칸이 늘 0", "죽은칸", """
class {cls}(ChaseEnv):
    def _obs(self):
        o = np.array(super()._obs(), dtype=np.float32)
        o[-1] = 0.0                     # ★심은 결함: 한 칸이 항상 0
        return o
"""),
    ("d04_range", "GcorpD04RangeEnv", "관측 값이 0~1 범위를 크게 넘는다", "범위깨짐", """
class {cls}(ChaseEnv):
    def _obs(self):
        return np.array(super()._obs(), dtype=np.float32) * 50.0   # ★심은 결함
"""),
    ("d05_earlyterm", "GcorpD05EarlytermEnv", "두 수만에 끝난다", "에피소드붕괴", """
class {cls}(ChaseEnv):
    def step(self, a):
        o, r, dn, inf = super().step(a)
        if self.t >= 2:                 # ★심은 결함: 첫 수부터 끝낼 수 있다
            return o, r, True, inf
        return o, r, dn, inf
"""),
    ("d06_nowon", "GcorpD06NowonEnv", "이겼는지 안 알려준다(won 키 없음)", "승패미보고", """
class {cls}(ChaseEnv):
    def step(self, a):
        o, r, dn, inf = super().step(a)
        return o, r, dn, {{}}           # ★심은 결함: won 키가 사라진다
"""),
    ("d07_dimover", "GcorpD07DimoverEnv", "관측 칸이 109개(상한 100 초과)", "차원초과", """
class {cls}(ChaseEnv):
    def __init__(self, horizon=120, seed=None):
        super().__init__(horizon=horizon, seed=seed)
        self.obs_dim = 109              # ★심은 결함

    def _obs(self):
        o = np.array(super()._obs(), dtype=np.float32)
        return np.concatenate([o, np.zeros(109 - len(o), dtype=np.float32)])
"""),
    ("d08_notimeout", "GcorpD08NotimeoutEnv", "시간이 다 돼도 안 끝난다", "미종료", """
class {cls}(ChaseEnv):
    def step(self, a):
        o, r, dn, inf = super().step(a)
        if not inf.get("won"):
            return o, r, False, inf     # ★심은 결함: 시간초과로 끝나지 않는다
        return o, r, dn, inf
"""),
    ("d09_revreward", "GcorpD09RevrewardEnv", "보상이 승리와 반대", "보상역정렬", """
class {cls}(ChaseEnv):
    def step(self, a):
        o, r, dn, inf = super().step(a)
        return o, -r, dn, inf           # ★심은 결함: 이길수록 손해
"""),
    ("d10_actionfree", "GcorpD10ActionfreeEnv", "무엇을 눌러도 결과가 같다", "행동무관", """
class {cls}(ChaseEnv):
    def step(self, a):
        return super().step(0)          # ★심은 결함: 버튼이 결과를 안 바꾼다
"""),
]


def main():
    man = [("파일", "클래스", "정답라벨", "설명")]
    for key, cls, desc, label, body in SPECS:
        path = os.path.join(ROOT, f"gcorp_{key}_env.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(HDR.format(desc=desc) + body.format(cls=cls).lstrip("\n"))
        man.append((f"gcorp_{key}_env", cls, label, desc))
        print(f"[생성] gcorp_{key}_env.py  ({label})")
    with open(os.path.join(ROOT, "corpus_manifest.tsv"), "w", encoding="utf-8") as f:
        for row in man:
            f.write("\t".join(row) + "\n")
    print(f"\n정답지 corpus_manifest.tsv  — 결함 {len(SPECS)-2}종 + 정상 2종")


if __name__ == "__main__":
    main()
