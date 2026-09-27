#!/usr/bin/env bash
set -euo pipefail

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: 'hf' CLI not found. Activate GSC_Token and install huggingface_hub." >&2
  exit 1
fi

download_dataset() {
  local repo_id="$1"
  local target="$2"
  local marker="${target}/.download_complete"
  if [[ -f "${marker}" ]]; then
    echo "SKIP ${repo_id}: completion marker exists at ${marker}"
    return
  fi
  mkdir -p "${target}"
  echo "DOWNLOAD ${repo_id} -> ${target} (hf resumes cached/partial transfers)"
  hf download "${repo_id}" --repo-type dataset --local-dir "${target}"
  touch "${marker}"
  echo "DONE ${repo_id}"
}

download_dataset "derek-thomas/ScienceQA" "./data/raw/ScienceQA"
download_dataset "vikhyatk/gqa-val" "./data/raw/GQA"
download_dataset "HuggingFaceM4/A-OKVQA" "./data/raw/A-OKVQA"

