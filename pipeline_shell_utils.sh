#!/usr/bin/env bash
# Shared retry/checkpoint helper for the staged Slurm entrypoints.

pipeline_invalidate_downstream() {
    local current="$1"
    local state_dir="$2"
    local found=false
    local step
    for step in "${PIPELINE_STAGE_STEPS[@]}"; do
        if [[ "$found" == true ]]; then
            rm -f -- "$state_dir/${step}.done" "$state_dir/${step}.failed"
        fi
        if [[ "$step" == "$current" ]]; then
            found=true
        fi
    done
}

pipeline_json_has_concepts() {
    local path="$1"
    python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path
try:
    with Path(sys.argv[1]).open(encoding="utf-8") as f:
        data = json.load(f)
    concepts = data.get("concepts")
    raise SystemExit(0 if isinstance(concepts, list) and bool(concepts) else 1)
except Exception:
    raise SystemExit(1)
PY
}

pipeline_outputs_ready() {
    local step="$1"
    local domain_dir="${PIPELINE_DOMAIN_DIR:?PIPELINE_DOMAIN_DIR is required}"
    local markdown_file
    case "$step" in
        stage1-preflight|stage2-preflight)
            return 0
            ;;
        stage1-global-consolidation)
            pipeline_json_has_concepts "$domain_dir/global_concepts.json"
            ;;
        stage1-stub-resolution)
            pipeline_json_has_concepts "$domain_dir/global_concepts_stubs_resolved.json"
            ;;
        stage1-thin-concept-folding|stage1-brief-generation|stage1-entity-overlap|stage2-noise-audit|stage2-bidirectional-relations)
            pipeline_json_has_concepts "$domain_dir/global_concepts_refined.json"
            ;;
        stage2-markdown-export)
            [[ -d "$domain_dir/concepts_md" ]] || return 1
            shopt -s nullglob
            local markdown_files=("$domain_dir"/concepts_md/*.md)
            shopt -u nullglob
            for markdown_file in "${markdown_files[@]}"; do
                [[ -s "$markdown_file" ]] && return 0
            done
            return 1
            ;;
        stage2-final-analysis)
            [[ -s "$domain_dir/islands_report.json" ]]
            ;;
        *)
            return 1
            ;;
    esac
}

pipeline_stage_step() {
    local step="$1"
    local state_dir="$2"
    local log_dir="$3"
    local resume="$4"
    local max_attempts="$5"
    local retry_forever="$6"
    shift 6

    local marker="${state_dir}/${step}.done"
    local failed_marker="${state_dir}/${step}.failed"
    local log_file="${log_dir}/${step}.log"
    local attempt=1
    local delay=15
    local rc=0

    if [[ "$resume" == true && "$step" != *preflight && -s "$marker" ]]; then
        if pipeline_outputs_ready "$step"; then
            echo ">>> [RESUME] Skipping completed step: ${step}"
            return 0
        fi
        echo ">>> [RESUME] Checkpoint for '${step}' has missing or invalid outputs; rerunning it."
        rm -f -- "$marker"
    fi
    if [[ "$step" != *preflight ]]; then
        pipeline_invalidate_downstream "$step" "$state_dir"
    fi

    while true; do
        {
            printf '\n===== %s | attempt %d | %s =====\n' "$step" "$attempt" "$(date -Is)"
            printf 'Command:'
            printf ' %q' "$@"
            printf '\n'
        } | tee -a "$log_file"

        if "$@" 2>&1 | tee -a "$log_file"; then
            if pipeline_outputs_ready "$step"; then
                printf 'completed_at=%s\nattempts=%s\n' "$(date -Is)" "$attempt" > "${marker}.tmp"
                mv -f -- "${marker}.tmp" "$marker"
                rm -f -- "$failed_marker"
                echo ">>> Completed: ${step} (attempt ${attempt})"
                return 0
            fi
            rc=1
            echo ">>> Step '${step}' exited successfully but its required outputs are missing or invalid." >&2
        else
            rc=$?
        fi

        printf 'failed_at=%s\nattempt=%s\nexit_code=%s\n' "$(date -Is)" "$attempt" "$rc" > "${failed_marker}.tmp"
        mv -f -- "${failed_marker}.tmp" "$failed_marker"
        rm -f -- "$marker"
        echo ">>> Step '${step}' failed with exit code ${rc}." >&2

        if [[ "$retry_forever" != true && "$attempt" -ge "$max_attempts" ]]; then
            echo "ERROR: '${step}' failed after ${attempt} attempt(s). Check ${log_file}; rerun with --resume." >&2
            return "$rc"
        fi

        if [[ "$step" != *preflight ]]; then
            echo ">>> Checking configured LLM servers before retry..." | tee -a "$log_file"
            if ! python3 "${PIPELINE_SCRIPT_DIR}/check_servers.py" 2>&1 | tee -a "$log_file"; then
                echo ">>> Server health check still failing; waiting before retry." | tee -a "$log_file"
            fi
        fi

        echo ">>> Retrying '${step}' in ${delay}s..." | tee -a "$log_file"
        sleep "$delay"
        attempt=$((attempt + 1))
        delay=$((delay * 2))
        if ((delay > 300)); then
            delay=300
        fi
    done
}
