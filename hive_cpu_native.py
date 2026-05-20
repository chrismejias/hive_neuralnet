from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

import torch
from torch.utils.cpp_extension import load


_extension = None


def load_extension():
    global _extension
    if _extension is not None:
        return _extension

    root = os.path.dirname(__file__)
    csrc = os.path.join(root, "hive_gpu", "csrc")
    build_dir = os.path.join(root, "build", "hive_cpu_native_ext")
    os.makedirs(build_dir, exist_ok=True)
    _ensure_msvc_environment()
    _ensure_ninja_on_path()

    candidates = [
        os.path.join(root, "hive_cpu_native_ext.pyd"),
        os.path.join(root, "hive_cpu_native_ext.so"),
    ]
    for path in candidates:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("hive_cpu_native_ext", path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules["hive_cpu_native_ext"] = module
                _extension = module
                return _extension

    _extension = load(
        name="hive_cpu_native_ext",
        sources=[os.path.join(csrc, "cpu_native.cpp")],
        build_directory=build_dir,
        extra_include_paths=[csrc],
        extra_cflags=["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"],
        verbose=False,
    )
    return _extension


def _ensure_ninja_on_path() -> None:
    if shutil.which("ninja") is not None:
        return
    scripts_dir = os.path.join(os.path.dirname(__file__), ".venv", "Scripts")
    if os.path.exists(os.path.join(scripts_dir, "ninja.exe")):
        os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")
        return
    try:
        import ninja
    except Exception:
        return
    ninja_dir = getattr(ninja, "BIN_DIR", None)
    if ninja_dir and os.path.isdir(ninja_dir):
        path = os.environ.get("PATH", "")
        if ninja_dir not in path.split(os.pathsep):
            os.environ["PATH"] = ninja_dir + os.pathsep + path


def _ensure_msvc_environment() -> None:
    if os.name != "nt" or shutil.which("cl") is not None:
        return

    candidates = [
        r"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
    ]
    vsdev = next((p for p in candidates if os.path.exists(p)), None)
    if vsdev is None:
        return

    script = f'@call "{vsdev}" -arch=x64\n@set\n'
    script_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".bat",
            delete=False,
            encoding="utf-8",
        ) as fh:
            fh.write(script)
            script_path = fh.name
        output = subprocess.check_output(
            ["cmd.exe", "/d", "/c", script_path],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return
    finally:
        if script_path:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            if key.lower() == "path":
                if key == "PATH" or "VC\\Tools\\MSVC" in value:
                    os.environ["PATH"] = value
                continue
            os.environ[key] = value
