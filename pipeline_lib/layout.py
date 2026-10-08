"""Where every file of one pipeline run lives.

A run writes to ``<wiki_base_dir>/<domain or 'default'>/``. Inside it:

* **Outputs** sit at the top level: ``global_concepts_refined.json``,
  ``concepts_md/``, ``islands_report.json``, ``doc_mapping.json`` and friends.
* **Resume checkpoints** also sit at the top level (``extraction_log.json``,
  ``pass2_log.json``, ...). They are read back by the runner and by the stages
  themselves, so they are state, not logs, even though some of their names end
  in ``_log``.
* **Logs** all sit in ``log/``, named ``<stage>.<kind>.jsonl``:

  - ``llm``        every LLM call and its response, one line per HTTP request
  - ``decisions``  what the stage changed, e.g. which stubs were absorbed where
  - ``debug``      failures with full evidence: traceback, prompt, raw output
  - ``run.log``    (written by ``run_pipeline.sh``) the stdout of each step

Every stage builds its paths through this module, so the layout is defined in
exactly one place.
"""

from pathlib import Path
from typing import Any

# The one folder that holds human-readable logs.
LOG_DIR_NAME = "log"


def run_dir_for(args: Any) -> Path:
    """Return the output directory for the run described by parsed CLI args.

    Args:
        args: Parsed namespace carrying ``wiki_base_dir`` and ``domain``. The
            parser already maps an omitted domain to ``"default"``.

    Returns:
        ``<wiki_base_dir>/<domain>``. Not created here.
    """
    return Path(args.wiki_base_dir) / args.domain


def log_path(run_dir: Path, stage: str, kind: str) -> Path:
    """Return the path of one stage's log of a given kind.

    Args:
        run_dir: The run's output directory.
        stage: Stage key, e.g. ``"pass1-extraction"`` or ``"stub-resolution"``.
        kind: One of ``"llm"``, ``"decisions"``, ``"debug"``.

    Returns:
        ``<run_dir>/log/<stage>.<kind>.jsonl``. The file is created on the first
        write, not here, so a stage that records nothing leaves no empty file.
    """
    return Path(run_dir) / LOG_DIR_NAME / f"{stage}.{kind}.jsonl"
