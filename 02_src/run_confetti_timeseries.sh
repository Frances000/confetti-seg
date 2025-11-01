#!/usr/bin/env bash
set -euo pipefail

# ROOT_SERIES_DIR="/Volumes/Lyons_X5/real_confetti_test/Reduced_4NQO_stitched"   # parent folder containing mouse folders
ROOT_SERIES_DIR="/Volumes/Lyons_X5/real_confetti_test/multi_point_test"
OUT_ROOT="/Volumes/Lyons_X5/treated_test_confetti_batch_22_10_25"                        # where outputs go

SPLIT_PY="02_src/split_confetti_channels.py"
STACK_PY="/Users/franceskan/Documents/confetti-seg/02_src/threshold_stack.py"
CSV_PY="02_src/create_csv_stack.py"
PRED_PY="02_src/convert_test_to_image.py"
SEGCLONES_PY="/Users/franceskan/Documents/confetti-seg/02_src/segment_clones.py"
TRACK_VIZ_PY="02_src/track_confetti_timeseries.py"   # merged tracker+visualiser

# MODEL_DISTANCE="/Volumes/Lyons_X5/confetti-seg/gadi_models/no_selection_distance_trained_models/model_distance_rf.joblib"
# MODEL_BASE="/Volumes/Lyons_X5/confetti-seg/gadi_models/no_selection_base_trained_models/model_base_rf.joblib"
MODEL_DISTANCE="/Volumes/Lyons_X5/distance_inclusion_variation/confetti-seg-training-reduced/all_dt_models/reduced_stack_no_selection_distance_trained_models/model_distance_rf.joblib"
MODEL_BASE="/Volumes/Lyons_X5/distance_inclusion_variation/confetti-seg-training-reduced/all_dt_models/reduced_stack_no_selection_base_trained_models/model_base_rf.joblib"

# Crop used end-to-end (HxW). Keep consistent with convert_test_to_image --shape
CROP_H=400
CROP_W=400
CROP_STR="${CROP_H}x${CROP_W}"

# Channel-number fallback 
CHNUM_MAP="1=R,2=G,3=BF,4=C,5=Y"

# Clone metrics params (optional)
MIN_PIXELS=10
PIX_SIZE_UM=0.5
CORE_RADIUS=3

# ---- Helpers ----
stem_noext() { basename "${1%.*}"; }

# our files look like ...-YYYYMMDD-merge.tif
# sort by the date block (3rd dash-separated field).
sort_by_date() {
  LC_ALL=C sort -t- -k3,3
}

process_one_image() {
  local IMG="$1"        # full path to ...-merge.tif
  local MOUSE="$2"      # mouse folder name, e.g. Aa01F-145201-stitched

  local STEM; STEM="$(stem_noext "$IMG")"
  echo "    • $STEM"

  # Destinations per image
  local DEST="${OUT_ROOT}/${MOUSE}/${STEM}"              # split outputs (C/R/G/Y/BF)
  local STACK_DIR="${OUT_ROOT}/${MOUSE}/stack_${STEM}"   # Base_/Distance_ stacks
  local CSV_DIR="${OUT_ROOT}/${MOUSE}/csv_${STEM}"       # CSVs
  local REPORT_DIR="${DEST}/reports"
  mkdir -p "$REPORT_DIR" "$STACK_DIR" "$CSV_DIR"

  # 1) Split composite → single-channel crops (also BF for background)
  python "$SPLIT_PY" \
    --image "$IMG" \
    --dest_root "$DEST" \
    --crop "$CROP_STR" \
    --include_bf \
    --chnum_map "$CHNUM_MAP" \
    --report_json "${REPORT_DIR}/split_report.json" \
    --overwrite

  # 2) Build feature stacks
  python3 "$STACK_PY" \
    --source "$DEST" \
    --dest "$STACK_DIR"

  # 3) Compose CSVs (inference mode ⇒ no class column)
  python "$CSV_PY" \
    --root "$STACK_DIR" \
    --out  "$CSV_DIR" \
    --mode infer

  # 4) Predict with base & distance models (each will write *_preds_labels.tif + legend)
  if [[ -f "${CSV_DIR}/${STEM}.csv" ]]; then
    python3 "$PRED_PY" --model_path "$MODEL_BASE" \
      --csv_path "${CSV_DIR}/${STEM}.csv" --shape "${CROP_H}x${CROP_W}"
  fi
  if [[ -f "${CSV_DIR}/distance_${STEM}.csv" ]]; then
    python3 "$PRED_PY" --model_path "$MODEL_DISTANCE" \
      --csv_path "${CSV_DIR}/distance_${STEM}.csv" --shape "${CROP_H}x${CROP_W}"
  fi

  # Prefer distance model outputs
    local LABEL_TIF=""
    local LEGEND_CSV=""

    if compgen -G "${CSV_DIR}/distance_*__preds_labels.tif" > /dev/null; then
    LABEL_TIF="$(ls "${CSV_DIR}"/distance_*__preds_labels.tif | head -n1)"
    LEGEND_CSV="$(ls "${CSV_DIR}"/distance_*__preds_legend.csv | head -n1)"
    elif compgen -G "${CSV_DIR}/*__preds_labels.tif" > /dev/null; then
    LABEL_TIF="$(ls "${CSV_DIR}"/*__preds_labels.tif | head -n1)"
    LEGEND_CSV="$(ls "${CSV_DIR}"/*__preds_legend.csv | head -n1)"
    fi

  if [[ -n "$LABEL_TIF" && -n "$LEGEND_CSV" ]]; then
    python3 "$SEGCLONES_PY" \
      --label_tiff "$LABEL_TIF" \
      --legend_csv "$LEGEND_CSV" \
      --csv_out     "${OUT_ROOT}/${MOUSE}/metrics_${STEM}.csv" \
      --overlay_out "${OUT_ROOT}/${MOUSE}/overlay_${STEM}.tif" \
      --min_pixels "$MIN_PIXELS" \
      --pix_size_um "$PIX_SIZE_UM" \
      --core_radius "$CORE_RADIUS"
  fi

  # 6) Record paths for time-series tracking of this mouse
  local BF_PATH="${DEST}/BF/${STEM}_BF.tif"
  mkdir -p "${OUT_ROOT}/${MOUSE}/_tracking_inputs"
  if [[ -n "$LABEL_TIF" ]]; then
    echo "$LABEL_TIF" >> "${OUT_ROOT}/${MOUSE}/_tracking_inputs/labels.txt"
    if [[ -f "$BF_PATH" ]]; then
      echo "$BF_PATH" >> "${OUT_ROOT}/${MOUSE}/_tracking_inputs/intensity.txt"
    else
      echo "" >> "${OUT_ROOT}/${MOUSE}/_tracking_inputs/intensity.txt"
    fi
  fi
}

process_one_mouse_folder() {
  local MOUSE_DIR="$1"                                   # e.g. /.../4NQO_stitched/Aa01F-145201-stitched
  local MOUSE; MOUSE="$(basename "$MOUSE_DIR")"
  echo "== Mouse series: $MOUSE =="

  # Find all *-merge.tif, sort by date block
  mapfile -t MERGES < <(find "$MOUSE_DIR" -maxdepth 1 -type f -name "*-merge.tif" | sort_by_date)

  if [[ ${#MERGES[@]} -eq 0 ]]; then
    echo "   (no -merge.tif files)"; return
  fi

  # reset tracking lists
  rm -f "${OUT_ROOT}/${MOUSE}/_tracking_inputs/labels.txt" 2>/dev/null || true
  rm -f "${OUT_ROOT}/${MOUSE}/_tracking_inputs/intensity.txt" 2>/dev/null || true

  # per-image loop
  for IMG in "${MERGES[@]}"; do
    process_one_image "$IMG" "$MOUSE"
  done

  # run tracking for this mouse
  local LFILE="${OUT_ROOT}/${MOUSE}/_tracking_inputs/labels.txt"
  if [[ ! -f "$LFILE" ]]; then
    echo "   (no labels collected; skipping tracking)"; return
  fi

  mapfile -t LABELS < <(awk 'NF' "$LFILE")
  local IFILE="${OUT_ROOT}/${MOUSE}/_tracking_inputs/intensity.txt"
  local INT_ARG=()
  if [[ -f "$IFILE" ]]; then
    mapfile -t INTS < <(awk 'NF' "$IFILE")
    if [[ ${#INTS[@]} -eq ${#LABELS[@]} ]]; then
      INT_ARG=(--intensity "${INTS[@]}")
    fi
  fi

  local OUT_TS="${OUT_ROOT}/${MOUSE}/tracking"
  mkdir -p "$OUT_TS"
  echo "   → Tracking ${#LABELS[@]} timepoints for $MOUSE"
  python "$TRACK_VIZ_PY" \
    --labels "${LABELS[@]}" \
    "${INT_ARG[@]}" \
    --out_dir "$OUT_TS"
}

# Usage:
#   ./run_confetti_timeseries.sh                      # process ALL mouse folders under ROOT_SERIES_DIR
#   ./run_confetti_timeseries.sh Aa01F-145201-stitched Aa01F-145202-stitched   # only these mice
if [[ $# -gt 0 ]]; then
  for M in "$@"; do
    process_one_mouse_folder "${ROOT_SERIES_DIR}/${M}"
  done
else
  # all subfolders (each a mouse series)
  find "$ROOT_SERIES_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r D; do
    process_one_mouse_folder "$D"
  done
fi

echo "Done."
