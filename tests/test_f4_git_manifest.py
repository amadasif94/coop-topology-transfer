"""F4: RunManifest must not crash when git is missing or unlaunchable, and must not hide other errors."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utility.run_manifest as rm


def done(code, out=""):
    return subprocess.CompletedProcess(["git"], code, out, "")


class GitHandling(unittest.TestCase):
    def manifest(self, side_effect=None, return_value=None):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(rm.subprocess, "run", side_effect=side_effect, return_value=return_value):
            run = rm.RunManifest(Path(tmp) / "run", "gcn", 7, "val", arguments=dict(a=1))
            data = json.loads(run.path.read_text())
            run.finish()
            final = json.loads(run.path.read_text())
        return data, final

    def test_git_success_records_revision(self):
        data, final = self.manifest(return_value=done(0, "abc123\n"))
        self.assertEqual(data["source_revision"], "abc123")
        self.assertEqual(data["source_revision_status"], "ok")
        self.assertEqual(final["status"], "finished")

    def test_git_command_failure(self):
        data, _ = self.manifest(return_value=done(128))
        self.assertIsNone(data["source_revision"])
        self.assertEqual(data["source_revision_status"], "git_failed")

    def test_missing_or_unlaunchable_git(self):
        for exc in (FileNotFoundError(2, "No such file", "git"), PermissionError(13, "denied"),
                    subprocess.TimeoutExpired(["git"], 30)):
            data, final = self.manifest(side_effect=exc)
            self.assertIsNone(data["source_revision"])
            self.assertEqual(data["source_revision_status"], "git_unavailable")
            # Everything else in the manifest is still written.
            self.assertEqual((data["controller"], data["seed"], data["mode"]), ("GCN", 7, "val"))
            self.assertEqual(data["arguments"], {"a": 1})
            self.assertEqual(final["status"], "finished")

    def test_unrelated_errors_still_raise(self):
        with patch.object(rm.subprocess, "run", side_effect=ValueError("not a git problem")):
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                rm.RunManifest(Path(tmp) / "run", "gcn", 7, "val")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            out.mkdir()
            (out / "existing.txt").write_text("x")
            with patch.object(rm.subprocess, "run", side_effect=FileNotFoundError()), \
                    self.assertRaises(FileExistsError):
                rm.RunManifest(out, "gcn", 7, "val")


if __name__ == "__main__":
    unittest.main()
