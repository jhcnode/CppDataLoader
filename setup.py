from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

ext_modules = [
    Extension(
        "CppDataLoader",  # 빌드될 모듈 이름
        ["CppDataLoader.cpp"],  # C++ 소스 파일
        include_dirs=[pybind11.get_include(),'/usr/include'],
        define_macros=[("GNU_SOURCE", "1")],  # memfd_create 등 GNU 확장 사용
        language="c++",
        extra_compile_args=["-std=c++14"],
        extra_link_args=[ "-lrt", "-pthread"]
    ),
]

setup(
    name="CppDataLoader",
    version="0.0.1",
    author="Your Name",
    author_email="your.email@example.com",
    description="C++ 멀티프로세싱, 비동기 prefetch 및 글로벌 shared memory 기반 최적화를 적용한 CppDataLoader",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
