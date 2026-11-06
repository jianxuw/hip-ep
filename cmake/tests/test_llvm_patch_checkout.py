# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

"""Exercise the LLVM patch through a real Git checkout and CMake entry point."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET = Path("external/llvm-project/llvm/lib/Support/Windows/Threading.inc")
PATCH = Path("cmake/llvm-fix-null-thread-handle.patch")
HELPER = Path("cmake/ApplyLLVMPatches.cmake")
SOURCE = """void llvm_thread_detach_impl(HANDLE hThread) {
  if (::CloseHandle(hThread) == FALSE)
    ReportLastErrorFatal("CloseHandle failed");
}

DWORD llvm_thread_get_id_impl(HANDLE hThread) { return ::GetThreadId(hThread); }

DWORD llvm_thread_get_current_id_impl() { return ::GetCurrentThreadId(); }
"""


class LLVMPatchCheckoutTest(unittest.TestCase):
    def test_checkout_and_idempotent_apply(self):
        for autocrlf in ("true", "false"):
            for source_eol in ("\n", "\r\n"):
                with self.subTest(autocrlf=autocrlf, source_eol=repr(source_eol)):
                    with tempfile.TemporaryDirectory(prefix="hip-llvm-patch-") as tmp:
                        root = Path(tmp).resolve()

                        def run(*args):
                            return subprocess.run(
                                args,
                                cwd=root,
                                capture_output=True,
                                text=True,
                                check=True,
                            ).stdout

                        run("git", "init", "-q")
                        run("git", "config", "core.autocrlf", autocrlf)
                        attributes = (REPO / ".gitattributes").read_text()
                        (root / ".gitattributes").write_text(
                            attributes + "\n*.inc -text\n", encoding="utf-8"
                        )
                        (root / TARGET).parent.mkdir(parents=True)
                        (root / TARGET).write_bytes(
                            SOURCE.replace("\n", source_eol).encode()
                        )
                        for relative in (PATCH, HELPER):
                            (root / relative).parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(REPO / relative, root / relative)
                        run("git", "add", ".")
                        # Force a fresh checkout so core.autocrlf actually runs.
                        (root / PATCH).unlink()
                        run("git", "checkout", "--", PATCH.as_posix())
                        self.assertNotIn(b"\r", (root / PATCH).read_bytes())
                        (root / "unrelated.txt").write_text("preserve me\n")
                        args = (
                            "cmake",
                            f"-DHIP_EP_LLVM_PATCH_SOURCE_DIR={root}",
                            "-P",
                            str(root / HELPER),
                        )
                        self.assertIn("Applied LLVM", run(*args))
                        patched = (root / TARGET).read_bytes()
                        self.assertIn(
                            "if (!hThread)\n    return 0;", (root / TARGET).read_text()
                        )
                        self.assertIn("already applied", run(*args))
                        self.assertEqual(patched, (root / TARGET).read_bytes())
                        self.assertEqual(
                            "preserve me\n", (root / "unrelated.txt").read_text()
                        )


if __name__ == "__main__":
    unittest.main()
