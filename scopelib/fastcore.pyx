# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""SCOPE 뜨거운 부분만 컴파일 (2026-08-14 신설).

> 사용자 08-13: *"파이썬도 컴파일하는 거 있다고 하던데, **싸이썬**인가?
>  러스트랑 파이썬 속도 차이가 10~20배, **컴파일링 되는 파이썬이 3~7배**."*
> 사용자 08-14: *"파이썬 중에 자주 쓰이는 거 싸이썬, 컴파일링 되는 파이썬으로 구현하기로 했었어."*

## 왜 여기인가 (실측으로 골랐다)

08-14 프로파일(resource_defense 12,196수):

| 자리 | 시간 | 비중 |
|---|---|---|
| 게임 step + _obs | 0.148초 | 74% ← 게임 쪽(여기서 못 건드린다) |
| **기록 변환**(`_enc`·`tolist`·`asarray`) | **0.042초** | **19%** ← ★여기 |
| 굴리기 나머지 | 0.014초 | 7% |

★**전부 컴파일하지 않는다.** 게임과 신경망은 각각 vizdoom·torch 라 파이썬을 벗어나도
  다시 불러야 해서 손해다. **매 수마다 도는 순수 파이썬 계산만** 옮긴다.

## 안전장치

빌드가 없거나 실패하면 `scopelib/sandbox.py` 가 **파이썬 판으로 떨어진다**.
컴파일이 안 돼도 SCOPE 는 그대로 돈다 — 결과도 같아야 한다(같은지 시험으로 확인한다).

빌드: `venv_gate/bin/python scopelib/build_fast.py`
"""
import numpy as np
cimport numpy as cnp
cimport cython

cnp.import_array()


def encode_obs(obj):
    """관측 → 파이썬 실수 목록. `np.asarray(o).tolist()` 를 대신한다.

    매 수마다 불린다. numpy 왕복을 없애는 것이 요점이다.
    """
    cdef cnp.ndarray[cnp.float32_t, ndim=1] a
    cdef Py_ssize_t i, n
    if isinstance(obj, np.ndarray) and obj.dtype == np.float32 and obj.ndim == 1:
        a = obj
    else:
        a = np.ascontiguousarray(obj, dtype=np.float32).ravel()
    n = a.shape[0]
    out = [0.0] * n
    for i in range(n):
        out[i] = a[i]
    return out


def obs_digest(obj):
    """화면 묶음 → 요약 여섯 칸.

    [평균, 표준편차, 최소, 최대, 밝은칸비율, 앞뒤 프레임 차이]
    `scopelib/sandbox.py` 의 같은 이름 함수와 **값이 같아야 한다**(시험으로 확인).
    """
    cdef cnp.ndarray a = np.ascontiguousarray(obj, dtype=np.float32)
    cdef cnp.ndarray flat
    cdef Py_ssize_t nch
    if a.ndim > 1:
        flat = a.reshape(a.shape[0], -1)
        nch = a.shape[0]
    else:
        flat = a.reshape(1, -1)
        nch = 1
    cdef cnp.ndarray[cnp.float32_t, ndim=1] v = a.ravel()
    cdef Py_ssize_t i, n = v.shape[0]
    cdef double s = 0.0, s2 = 0.0, mn = v[0], mx = v[0], bright = 0.0
    cdef double x
    for i in range(n):
        x = v[i]
        s += x
        s2 += x * x
        if x < mn:
            mn = x
        if x > mx:
            mx = x
        if x > 0.05:
            bright += 1.0
    cdef double mean = s / n
    cdef double var = s2 / n - mean * mean
    if var < 0.0:
        var = 0.0
    cdef double motion = 0.0
    if nch > 1:
        motion = float(np.abs(flat[nch - 1] - flat[0]).mean())
    return [mean, var ** 0.5, mn, mx, bright / n, motion]
