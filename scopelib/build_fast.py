"""SCOPE 뜨거운 부분 컴파일 (2026-08-14).

빌드: venv_gate/bin/python scopelib/build_fast.py
★라이브 venv 에 Cython 을 넣지 않는다(규약 §9-2 — 팜 6대가 같이 죽는다).
  venv_gate 에서 만들고, 만들어진 .so 는 같은 파이썬(3.14)이라 라이브가 그대로 쓴다.
"""
import os
import sys

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], "build_ext", "--inplace"]
setup(
    name="scope_fastcore",
    ext_modules=cythonize(os.path.join(HERE, "fastcore.pyx"),
                          compiler_directives={"language_level": "3"}),
    include_dirs=[np.get_include()],
    script_args=["build_ext", "--inplace"],
    options={"build_ext": {"inplace": True, "build_lib": HERE}},
)
