"""Greedy Qwen MTP must commit exactly the tokens plain decode commits.

Every verify row (2 rows at depth 2, 3 at depth 3) has to round like the
one-token decode at its position, so the default depth policy, forced depth
2 and forced depth 3 must reproduce the plain greedy continuation byte for
byte.  A near-tied logit
anywhere in the continuation turns a rounding difference into a different
token, so long continuations of several prompts are checked.  Pass
--prompts-dir to check a directory of *.txt prompts instead of the built-in
ones.
"""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

PROMPTS = {
    # diverged at depth 2 while the attention of the verify rows took other
    # kernels and key splits than one-token decode
    "oped": "Write a persuasive op-ed of about 500 words arguing that small towns should invest in "
            "public libraries rather than new parking garages. Use a warm, conversational tone and "
            "one concrete anecdote.",
    "prose": "Write a 600-word short story about a lighthouse keeper on a remote northern island who "
             "discovers that the ships she has been guiding for twenty years were never real.",
    "code": "Write a Python module that parses ISO-8601 durations (like P3Y6M4DT12H30M5S and PT0.5S) "
            "into a dataclass, supports addition of two durations, and includes a thorough pytest test "
            "suite. Output only the code.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=("metal", "cuda"))
    parser.add_argument("--prompts-dir", type=Path)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output or Path(tempfile.mkdtemp(prefix="qwen-mtp-exact-"))
    out.mkdir(parents=True, exist_ok=True)
    prompts = PROMPTS
    if args.prompts_dir:
        prompts = {p.stem: p.read_text().rstrip("\n") for p in sorted(args.prompts_dir.glob("*.txt"))}
    base = [str(root / "ds4"), *(["--" + args.backend] if args.backend else []),
            "-m", str(args.model.resolve()), "--ctx", "4096", "--temp", "0", "--nothink",
            "-n", str(args.tokens)]
    failed = []
    for name, prompt in prompts.items():
        outputs = {}
        for mode, depth, extra in [("plain", None, []), ("auto", None, ["--mtp"]),
                                   ("depth2", "2", ["--mtp"]), ("depth3", "3", ["--mtp"])]:
            env = os.environ.copy()
            env.pop("DS4_QWEN4_SPEC_FORCE_ACCEPT", None)
            env.pop("DS4_QWEN4_MTP_DEPTH", None)
            if depth:
                env["DS4_QWEN4_MTP_DEPTH"] = depth
            result = subprocess.run(base + ["-p", prompt] + extra, cwd=root, env=env,
                                    capture_output=True, timeout=600)
            (out / f"{name}-{mode}.stdout").write_bytes(result.stdout)
            (out / f"{name}-{mode}.stderr").write_bytes(result.stderr)
            assert result.returncode == 0, (name, mode, result.stderr.decode(errors="replace"))
            outputs[mode] = result.stdout
        for mode in ("auto", "depth2", "depth3"):
            if outputs[mode] != outputs["plain"]:
                failed.append(f"{name} {mode}")
                print(f"FAIL {name} {mode}: MTP changed the greedy continuation", flush=True)
            else:
                print(f"PASS {name} {mode}", flush=True)
    assert not failed, f"MTP is not exact for {', '.join(failed)}; see {out}"


if __name__ == "__main__":
    main()
