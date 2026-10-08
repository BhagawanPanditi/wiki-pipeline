import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "run_entire_pipeline.sh"


class PipelineRunnerTests(unittest.TestCase):
    def test_skip_extraction_requires_consolidated_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    str(RUNNER),
                    "biology",
                    "--skip-extraction",
                    "--wiki-base-dir",
                    temp_dir,
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("requires valid, non-empty concepts", result.stderr)

    def test_resume_skips_completed_stages_and_retries_a_failed_stage(self):
        real_python = shutil.which("python3")
        self.assertIsNotNone(real_python)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            call_log = root / "calls.log"
            global_attempts = root / "global_attempts.txt"
            consolidated_input = root / "wiki" / "biology" / "consolidated" / "canon_test" / "concepts.json"
            consolidated_input.parent.mkdir(parents=True)
            consolidated_input.write_text('{"concepts": [{"title": "Mock Concept"}]}', encoding="utf-8")

            fake_python = bin_dir / "python3"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"REAL_PYTHON={real_python!s}\n"
                "if [[ \"${1:-}\" == \"-\" ]]; then exec \"$REAL_PYTHON\" \"$@\"; fi\n"
                "script=$(basename \"$1\")\n"
                "echo \"$script\" >> \"$FAKE_CALL_LOG\"\n"
                "shift\n"
                "domain=biology\n"
                "base=\n"
                "while [[ $# -gt 0 ]]; do\n"
                "  case \"$1\" in\n"
                "    --domain) domain=$2; shift 2 ;;\n"
                "    --wiki_base_dir) base=$2; shift 2 ;;\n"
                "    *) shift ;;\n"
                "  esac\n"
                "done\n"
                "dir=\"$base/$domain\"\n"
                "case \"$script\" in\n"
                "  global_concept_consolidation.py)\n"
                f"    count=$(cat \"{global_attempts}\" 2>/dev/null || echo 0)\n"
                "    count=$((count + 1))\n"
                f"    echo \"$count\" > \"{global_attempts}\"\n"
                "    if ((count == 1)); then exit 1; fi\n"
                "    printf '{\"concepts\": [{\"title\": \"Mock Concept\"}]}' > \"$dir/global_concepts.json\" ;;\n"
                "  stub_resolution.py) printf '{\"concepts\": [{\"title\": \"Mock Concept\"}]}' > \"$dir/global_concepts_stubs_resolved.json\" ;;\n"
                "  analyze_islands.py) printf '{\"summary\": {}, \"islands\": []}' > \"$dir/islands_report.json\" ;;\n"
                "  thin_concept_handler.py|add_concept_briefs.py|link_entity_overlap.py|audit_noise_islands.py|synthesize_bidirectional_relations.py|final_consolidate_and_link.py)\n"
                "    printf '{\"concepts\": [{\"title\": \"Mock Concept\"}]}' > \"$dir/global_concepts_refined.json\"\n"
                "    if [[ \"$script\" == final_consolidate_and_link.py ]]; then mkdir -p \"$dir/concepts_md\"; echo test > \"$dir/concepts_md/test.md\"; fi ;;\n"
                "esac\n"
            )
            fake_python.chmod(0o755)

            fake_sleep = bin_dir / "sleep"
            fake_sleep.write_text("#!/usr/bin/env bash\nexit 0\n")
            fake_sleep.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
            env["FAKE_CALL_LOG"] = str(call_log)

            cmd = [
                str(RUNNER),
                "biology",
                "--skip-extraction",
                "--wiki-base-dir",
                str(root / "wiki"),
                "--max-stage-attempts",
                "2",
            ]
            first = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, msg=first.stdout + first.stderr)
            self.assertEqual(global_attempts.read_text().strip(), "2")

            calls_before_resume = len(call_log.read_text().splitlines())
            resumed_cmd = cmd[:1] + cmd[1:2] + ["--resume"] + cmd[2:]
            resumed = subprocess.run(
                resumed_cmd,
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(resumed.returncode, 0, msg=resumed.stdout + resumed.stderr)
            new_calls = call_log.read_text().splitlines()[calls_before_resume:]
            self.assertEqual(new_calls, ["check_servers.py"])

            # If an upstream artifact is missing, resume reruns that stage and
            # invalidates checkpoints for all downstream stages.
            (root / "wiki" / "biology" / "global_concepts.json").unlink()
            calls_before_rebuild = len(call_log.read_text().splitlines())
            rebuilt = subprocess.run(
                resumed_cmd,
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(rebuilt.returncode, 0, msg=rebuilt.stdout + rebuilt.stderr)
            rebuilt_calls = call_log.read_text().splitlines()[calls_before_rebuild:]
            self.assertEqual(rebuilt_calls[0:2], ["check_servers.py", "global_concept_consolidation.py"])
            self.assertIn("stub_resolution.py", rebuilt_calls)
            self.assertIn("final_consolidate_and_link.py", rebuilt_calls)


if __name__ == "__main__":
    unittest.main()
