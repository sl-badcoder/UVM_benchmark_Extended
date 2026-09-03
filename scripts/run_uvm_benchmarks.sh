#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_ROOT="$ROOT_DIR/UVM_benchmarks"

: "${REPEATS:=3}"
: "${OUT_DIR:=$ROOT_DIR/benchmark_runs/$(date +%Y%m%d_%H%M%S)}"
: "${BENCHMARKS:=kmeans,cnn,pathfinder,gramschm}"
: "${GB_SIZES:=}"
: "${KMEANS_SIZES:=1000000}"
: "${CNN_SIZES:=0}"
: "${PATHFINDER_SIZES:=100000:100:20}"
: "${GRAMSCHM_SIZES:=2048:2048}"
: "${KMEANS_K:=32}"
: "${KMEANS_ITERS:=2}"
: "${KMEANS_TILE_GIB:=1}"
: "${PATHFINDER_ROWS:=100}"
: "${PATHFINDER_PYRAMID:=20}"
: "${KEEP_GOING:=0}"
: "${FEATURE_CONFIGS:=all}"

# Maximum runtime of ONE benchmark execution in seconds.
#
# 3600 = one hour.
# 0    = disable timeout.
: "${MAX_RUNTIME:=3600}"


usage() {
  cat <<'EOF'
Usage: scripts/run_uvm_benchmarks.sh [options]

Options:
  -r, --repeats N                 Number of times to run each size.
                                   Default: 3

  -o, --out-dir DIR               Directory for logs.
                                   Default: benchmark_runs/<timestamp>

      --benchmarks LIST           Comma list:
                                   kmeans,cnn,pathfinder,gramschm

      --gb-sizes LIST             Comma list of target GiB sizes for every
                                   benchmark.

                                   Suffixes GB/GiB are accepted.

                                   Example:
                                     24GB,48GB,72GB,96GB

      --kmeans-sizes LIST         Legacy option; not used by
                                   kmeans_standard-only mode.

      --cnn-sizes LIST            Comma list of train[:test] limits.
                                   0 means full loaded dataset.

                                   Example:
                                     1000:100,5000:1000

      --pathfinder-sizes LIST     Comma list of cols:rows:pyramid.

                                   Example:
                                     100000:100:20

      --gramschm-sizes LIST       Comma list of M:N.

                                   Example:
                                     2048:2048,4096:4096

      --feature-configs LIST      Comma list:
                                   none,memadvise,pref,both,all

                                   Default: all

      --configs LIST              Alias for --feature-configs.

      --max-runtime SECONDS       Maximum runtime for ONE benchmark
                                   execution.

                                   Default: 3600 seconds (1 hour).

                                   Set to 0 to disable the timeout.

      --keep-going                Continue after compile failures,
                                   benchmark failures, and timeouts.

  -h, --help                      Show this help.


Environment variables with the same uppercase names can also set defaults:

  REPEATS
  OUT_DIR
  BENCHMARKS
  GB_SIZES
  KMEANS_SIZES
  CNN_SIZES
  PATHFINDER_SIZES
  GRAMSCHM_SIZES
  KMEANS_K
  KMEANS_ITERS
  KMEANS_TILE_GIB
  PATHFINDER_ROWS
  PATHFINDER_PYRAMID
  KEEP_GOING
  FEATURE_CONFIGS
  MAX_RUNTIME


GB mode conversion rules:

  kmeans
    -> ONLY kmeans_standard --random-gib <GiB>
       using x+y managed data size

  cnn
    -> CNN --synthetic-gib <GiB>
       using synthetic managed training data

  pathfinder
    -> cols derived from
       <GiB> / (PATHFINDER_ROWS * sizeof(int))

  gramschm
    -> square M:N derived from
       <GiB> / (3 managed float arrays)


Examples:

  # Default one-hour timeout:
  ./scripts/run_uvm_benchmarks.sh \
    --gb-sizes 24GB,48GB \
    --keep-going

  # Maximum 30 minutes per benchmark:
  ./scripts/run_uvm_benchmarks.sh \
    --gb-sizes 24GB,48GB \
    --max-runtime 1800 \
    --keep-going

  # Maximum one hour per benchmark:
  ./scripts/run_uvm_benchmarks.sh \
    --gb-sizes 24GB,48GB \
    --max-runtime 3600 \
    --keep-going

  # Disable timeout completely:
  ./scripts/run_uvm_benchmarks.sh \
    --max-runtime 0

EOF
}


# ============================================================
# Parse command-line arguments
# ============================================================

while [[ $# -gt 0 ]]; do
  case "$1" in

    -r|--repeats)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi

      REPEATS="$2"
      shift 2
      ;;

    -o|--out-dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi

      OUT_DIR="$2"
      shift 2
      ;;

    --benchmarks)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi

      BENCHMARKS="$2"
      shift 2
      ;;

    --gb-sizes)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi

      GB_SIZES="$2"
      shift 2
      ;;

    --kmeans-sizes)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi

      KMEANS_SIZES="$2"
      shift 2
      ;;

    --cnn-sizes)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi

      CNN_SIZES="$2"
      shift 2
      ;;

    --pathfinder-sizes)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi

      PATHFINDER_SIZES="$2"
      shift 2
      ;;

    --gramschm-sizes)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi

      GRAMSCHM_SIZES="$2"
      shift 2
      ;;

    --feature-configs|--configs)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi

      FEATURE_CONFIGS="$2"
      shift 2
      ;;

    --max-runtime)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi

      MAX_RUNTIME="$2"
      shift 2
      ;;

    --keep-going)
      KEEP_GOING=1
      shift
      ;;

    -h|--help)
      usage
      exit 0
      ;;

    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done


# ============================================================
# Feature configuration handling
# ============================================================

append_feature_config() {
  local candidate="$1"
  local existing

  for existing in "${FEATURE_CONFIG_LIST[@]:-}"; do
    [[ "$existing" == "$candidate" ]] && return 0
  done

  FEATURE_CONFIG_LIST+=("$candidate")
}


parse_feature_configs() {
  local raw token normalized
  local -a requested

  FEATURE_CONFIG_LIST=()

  IFS=',' read -r -a requested <<< "$FEATURE_CONFIGS"

  for raw in "${requested[@]}"; do
    token="${raw//[[:space:]]/}"
    normalized="$(printf "%s" "$token" | tr '[:upper:]' '[:lower:]')"

    case "$normalized" in

      all)
        append_feature_config "none"
        append_feature_config "memadvise"
        append_feature_config "pref"
        append_feature_config "both"
        ;;

      none|off|baseline)
        append_feature_config "none"
        ;;

      memadvise|mem-advice)
        append_feature_config "memadvise"
        ;;

      pref|prefetch)
        append_feature_config "pref"
        ;;

      both|memadvise+pref|pref+memadvise|memadvise_pref|pref_memadvise)
        append_feature_config "both"
        ;;

      "")
        ;;

      *)
        echo \
          "Invalid feature config '$raw'. Expected none, memadvise, pref, both, or all." \
          >&2
        exit 2
        ;;
    esac
  done

  if [[ "${#FEATURE_CONFIG_LIST[@]}" -eq 0 ]]; then
    echo "FEATURE_CONFIGS must contain at least one configuration." >&2
    exit 2
  fi
}


set_feature_config() {
  FEATURE_CONFIG="$1"

  # -U first makes each configuration explicit even if the caller
  # supplied one of these macros elsewhere on the compiler command line.
  case "$FEATURE_CONFIG" in

    none)
      FEATURE_CPPFLAGS="-UMEMADVISE -UPREF"
      ;;

    memadvise)
      FEATURE_CPPFLAGS="-UMEMADVISE -UPREF -DMEMADVISE"
      ;;

    pref)
      FEATURE_CPPFLAGS="-UMEMADVISE -UPREF -DPREF"
      ;;

    both)
      FEATURE_CPPFLAGS="-UMEMADVISE -UPREF -DMEMADVISE -DPREF"
      ;;

    *)
      echo "Internal error: unknown feature config '$FEATURE_CONFIG'" >&2
      exit 2
      ;;
  esac
}


parse_feature_configs


# ============================================================
# Validate parameters
# ============================================================

if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || [[ "$REPEATS" -lt 1 ]]; then
  echo "REPEATS must be a positive integer" >&2
  exit 2
fi

if ! [[ "$MAX_RUNTIME" =~ ^[0-9]+$ ]]; then
  echo "MAX_RUNTIME must be a non-negative integer in seconds." >&2
  exit 2
fi

if ! [[ "$KEEP_GOING" =~ ^[01]$ ]]; then
  echo "KEEP_GOING must be 0 or 1." >&2
  exit 2
fi


# ============================================================
# Find timeout executable
# ============================================================

TIMEOUT_BIN=""

if [[ "$MAX_RUNTIME" -gt 0 ]]; then

  if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_BIN="$(command -v timeout)"

  elif command -v gtimeout >/dev/null 2>&1; then
    # GNU coreutils uses "gtimeout" on some systems such as macOS.
    TIMEOUT_BIN="$(command -v gtimeout)"

  else
    echo "ERROR: MAX_RUNTIME=$MAX_RUNTIME, but no timeout command was found." >&2
    echo >&2
    echo "Install GNU coreutils or set:" >&2
    echo "  --max-runtime 0" >&2
    echo "to disable the timeout." >&2
    exit 2
  fi
fi


# ============================================================
# Prepare output directory
# ============================================================

mkdir -p "$OUT_DIR"

IFS=',' read -r -a BENCHMARK_LIST <<< "$BENCHMARKS"
IFS=',' read -r -a GB_SIZE_LIST <<< "$GB_SIZES"
IFS=',' read -r -a KMEANS_SIZE_LIST <<< "$KMEANS_SIZES"
IFS=',' read -r -a CNN_SIZE_LIST <<< "$CNN_SIZES"
IFS=',' read -r -a PATHFINDER_SIZE_LIST <<< "$PATHFINDER_SIZES"
IFS=',' read -r -a GRAMSCHM_SIZE_LIST <<< "$GRAMSCHM_SIZES"


# ============================================================
# Store run configuration
# ============================================================

CONFIG_FILE="$OUT_DIR/config.txt"

{
  echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "hostname=$(hostname)"
  echo "root_dir=$ROOT_DIR"
  echo "bench_root=$BENCH_ROOT"
  echo "out_dir=$OUT_DIR"
  echo
  echo "repeats=$REPEATS"
  echo "benchmarks=$BENCHMARKS"
  echo "feature_configs=$FEATURE_CONFIGS"
  echo "gb_sizes=$GB_SIZES"
  echo
  echo "kmeans_sizes=$KMEANS_SIZES"
  echo "cnn_sizes=$CNN_SIZES"
  echo "pathfinder_sizes=$PATHFINDER_SIZES"
  echo "gramschm_sizes=$GRAMSCHM_SIZES"
  echo
  echo "kmeans_k=$KMEANS_K"
  echo "kmeans_iters=$KMEANS_ITERS"
  echo "kmeans_tile_gib=$KMEANS_TILE_GIB"
  echo "pathfinder_rows=$PATHFINDER_ROWS"
  echo "pathfinder_pyramid=$PATHFINDER_PYRAMID"
  echo
  echo "keep_going=$KEEP_GOING"
  echo "max_runtime_seconds=$MAX_RUNTIME"

  if [[ "$MAX_RUNTIME" -gt 0 ]]; then
    echo "timeout_enabled=1"
    echo "timeout_binary=$TIMEOUT_BIN"
  else
    echo "timeout_enabled=0"
    echo "timeout_binary=disabled"
  fi

} > "$CONFIG_FILE"


# ============================================================
# Summary file
# ============================================================

status_file="$OUT_DIR/summary.tsv"

printf \
  "benchmark\tconfig\tsize\trepeat\tstatus\tmax_runtime_s\texit_code\tlog\n" \
  > "$status_file"


# ============================================================
# Compilation
# ============================================================

run_step() {
  local benchmark="$1"
  local size="$2"
  local repeat="$3"
  local workdir="$4"

  shift 4

  local log="$OUT_DIR/${benchmark}_${FEATURE_CONFIG}_${size//[:\/]/_}_run${repeat}.log"

  echo \
    "[$benchmark][$FEATURE_CONFIG] size=$size repeat=$repeat compiling ($FEATURE_CPPFLAGS)"

  if ! (
    cd "$workdir" &&
    make clean &&
    "$@"
  ) > "$log" 2>&1; then

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$benchmark" \
      "$FEATURE_CONFIG" \
      "$size" \
      "$repeat" \
      "compile_failed" \
      "$MAX_RUNTIME" \
      "-" \
      "$log" \
      >> "$status_file"

    echo \
      "[$benchmark][$FEATURE_CONFIG] compile failed, see $log" \
      >&2

    [[ "$KEEP_GOING" == "1" ]] && return 1

    exit 1
  fi

  return 0
}


# ============================================================
# Benchmark execution with timeout
# ============================================================

run_command() {
  local benchmark="$1"
  local size="$2"
  local repeat="$3"
  local workdir="$4"

  shift 4

  local log="$OUT_DIR/${benchmark}_${FEATURE_CONFIG}_${size//[:\/]/_}_run${repeat}.log"

  echo \
    "[$benchmark][$FEATURE_CONFIG] size=$size repeat=$repeat running"

  {
    echo
    echo "============================================================"
    echo "Benchmark execution"
    echo "============================================================"
    echo "benchmark=$benchmark"
    echo "config=$FEATURE_CONFIG"
    echo "size=$size"
    echo "repeat=$repeat"
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "max_runtime_seconds=$MAX_RUNTIME"

    if [[ "$MAX_RUNTIME" -gt 0 ]]; then
      echo "timeout_enabled=1"
    else
      echo "timeout_enabled=0"
    fi

    echo "============================================================"
    echo
  } >> "$log"

  local exit_code
  local start_epoch
  local end_epoch
  local elapsed_seconds

  start_epoch="$(date +%s)"

  if [[ "$MAX_RUNTIME" -gt 0 ]]; then

    (
      cd "$workdir" &&
      "$TIMEOUT_BIN" \
        --signal=TERM \
        --kill-after=30s \
        "${MAX_RUNTIME}s" \
        "$@"
    ) >> "$log" 2>&1

    exit_code=$?

  else

    (
      cd "$workdir" &&
      "$@"
    ) >> "$log" 2>&1

    exit_code=$?
  fi

  end_epoch="$(date +%s)"
  elapsed_seconds=$((end_epoch - start_epoch))

  {
    echo
    echo "============================================================"
    echo "Benchmark finished"
    echo "============================================================"
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "elapsed_seconds=$elapsed_seconds"
    echo "exit_code=$exit_code"
  } >> "$log"


  # ----------------------------------------------------------
  # Successful benchmark
  # ----------------------------------------------------------

  if [[ "$exit_code" -eq 0 ]]; then

    echo "status=ok" >> "$log"
    echo "============================================================" >> "$log"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$benchmark" \
      "$FEATURE_CONFIG" \
      "$size" \
      "$repeat" \
      "ok" \
      "$MAX_RUNTIME" \
      "$exit_code" \
      "$log" \
      >> "$status_file"

    echo \
      "[$benchmark][$FEATURE_CONFIG] finished in ${elapsed_seconds}s"

    return 0
  fi


  # ----------------------------------------------------------
  # Timeout
  # ----------------------------------------------------------

  if [[ "$MAX_RUNTIME" -gt 0 && "$exit_code" -eq 124 ]]; then

    {
      echo "status=timeout"
      echo "timeout_after_seconds=$MAX_RUNTIME"
      echo "============================================================"
    } >> "$log"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$benchmark" \
      "$FEATURE_CONFIG" \
      "$size" \
      "$repeat" \
      "timeout" \
      "$MAX_RUNTIME" \
      "$exit_code" \
      "$log" \
      >> "$status_file"

    echo \
      "[$benchmark][$FEATURE_CONFIG] TIMEOUT after ${MAX_RUNTIME}s" \
      >&2

    if [[ "$KEEP_GOING" == "1" ]]; then
      return 0
    fi

    exit 1
  fi


  # ----------------------------------------------------------
  # Other failure
  # ----------------------------------------------------------

  {
    echo "status=run_failed"
    echo "============================================================"
  } >> "$log"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$benchmark" \
    "$FEATURE_CONFIG" \
    "$size" \
    "$repeat" \
    "run_failed" \
    "$MAX_RUNTIME" \
    "$exit_code" \
    "$log" \
    >> "$status_file"

  echo \
    "[$benchmark][$FEATURE_CONFIG] run failed with exit code $exit_code, see $log" \
    >&2

  if [[ "$KEEP_GOING" == "1" ]]; then
    return 0
  fi

  exit 1
}


# ============================================================
# Make helpers
# ============================================================

make_with_feature_flags() {
  local nvcc_append="${NVCC_APPEND_FLAGS:-}"

  if [[ -n "$FEATURE_CPPFLAGS" ]]; then
    nvcc_append="${nvcc_append:+$nvcc_append }$FEATURE_CPPFLAGS"
  fi

  # NVCC_APPEND_FLAGS is honored by nvcc itself, so this does not
  # overwrite optimization/debug flags defined in individual Makefiles.
  NVCC_APPEND_FLAGS="$nvcc_append" make "$@"
}


compile_kmeans() {
  # Build only the implementation used by this runner.
  make_with_feature_flags kmeans_standard
}


compile_cnn() {
  make_with_feature_flags \
    all \
    CC="${CUDA_DIR:-/usr/local/cuda-12.8}/bin/nvcc"
}


compile_pathfinder() {
  make_with_feature_flags release
}


compile_gramschm() {
  make_with_feature_flags all
}


# ============================================================
# Size conversion helpers
# ============================================================

normalize_gib() {
  local raw="$1"
  local upper

  upper="$(printf "%s" "$raw" | tr '[:lower:]' '[:upper:]')"

  upper="${upper%GIB}"
  upper="${upper%GB}"
  upper="${upper%G}"

  if ! [[ "$upper" =~ ^[0-9]+$ ]] || [[ "$upper" -lt 1 ]]; then
    echo \
      "Invalid GB size '$raw'. Use positive integer values like 24GB or 24GiB." \
      >&2
    exit 2
  fi

  printf "%s" "$upper"
}


gib_to_bytes() {
  local gib="$1"

  printf "%s" $((gib * 1024 * 1024 * 1024))
}


gramschm_size_for_gib() {
  local gib="$1"
  local bytes

  bytes="$(gib_to_bytes "$gib")"

  awk \
    -v bytes="$bytes" \
    'BEGIN {
       n = int(sqrt(bytes / 12));
       if (n < 1) n = 1;
       printf "%d:%d", n, n
     }'
}


pathfinder_size_for_gib() {
  local gib="$1"
  local bytes
  local cols

  bytes="$(gib_to_bytes "$gib")"

  cols=$((bytes / 4 / PATHFINDER_ROWS))

  if [[ "$cols" -lt 1 ]]; then
    cols=1
  fi

  printf \
    "%s:%s:%s" \
    "$cols" \
    "$PATHFINDER_ROWS" \
    "$PATHFINDER_PYRAMID"
}


# ============================================================
# K-means
# ============================================================

run_kmeans() {
  local size="$1"
  local repeat="$2"

  : "$repeat"

  echo \
    "kmeans is configured to run ONLY ./kmeans_standard." \
    >&2

  echo \
    "Point-count mode (--kmeans-sizes, requested size: $size) would previously run ./kmeans_cuda and is now disabled." \
    >&2

  echo \
    "Use --gb-sizes <LIST> for kmeans_standard, e.g. --gb-sizes 24GB,48GB." \
    >&2

  exit 2
}


run_kmeans_gib() {
  local gib="$1"
  local repeat="$2"

  local workdir="$BENCH_ROOT/kmeans"
  local size="${gib}GiB"

  run_step \
    "kmeans" \
    "$size" \
    "$repeat" \
    "$workdir" \
    compile_kmeans \
    || return 0

  if [[ ! -x "$workdir/kmeans_standard" ]]; then
    echo \
      "Expected kmeans standard binary not found or not executable: $workdir/kmeans_standard" \
      >&2

    exit 1
  fi

  run_command \
    "kmeans" \
    "$size" \
    "$repeat" \
    "$workdir" \
    ./kmeans_standard \
    --random-gib "$gib" \
    "$KMEANS_K" \
    "$KMEANS_ITERS" \
    "$gib"
}


# ============================================================
# CNN
# ============================================================

run_cnn() {
  local size="$1"
  local repeat="$2"

  local workdir="$BENCH_ROOT/CNN"
  local train_limit="${size%%:*}"
  local test_limit="${size#*:}"

  run_step \
    "cnn" \
    "$size" \
    "$repeat" \
    "$workdir" \
    compile_cnn \
    || return 0

  if [[ "$size" == "$train_limit" ]]; then

    if [[ "$train_limit" == "0" ]]; then

      run_command \
        "cnn" \
        "$size" \
        "$repeat" \
        "$workdir" \
        ./CNN

    else

      run_command \
        "cnn" \
        "$size" \
        "$repeat" \
        "$workdir" \
        ./CNN \
        "$train_limit"
    fi

  else

    run_command \
      "cnn" \
      "$size" \
      "$repeat" \
      "$workdir" \
      ./CNN \
      "$train_limit" \
      "$test_limit"
  fi
}


run_cnn_gib() {
  local gib="$1"
  local repeat="$2"

  local workdir="$BENCH_ROOT/CNN"
  local size="${gib}GiB"

  run_step \
    "cnn" \
    "$size" \
    "$repeat" \
    "$workdir" \
    compile_cnn \
    || return 0

  run_command \
    "cnn" \
    "$size" \
    "$repeat" \
    "$workdir" \
    ./CNN \
    --synthetic-gib "$gib"
}


# ============================================================
# Pathfinder
# ============================================================

run_pathfinder() {
  local size="$1"
  local repeat="$2"

  local workdir="$BENCH_ROOT/rodinia/pathfinder"

  local cols
  local rows
  local pyramid

  IFS=':' read -r cols rows pyramid <<< "$size"

  if [[ -z "${cols:-}" || -z "${rows:-}" || -z "${pyramid:-}" ]]; then
    echo \
      "Invalid pathfinder size '$size'. Expected cols:rows:pyramid" \
      >&2

    exit 2
  fi

  run_step \
    "pathfinder" \
    "$size" \
    "$repeat" \
    "$workdir" \
    compile_pathfinder \
    || return 0

  run_command \
    "pathfinder" \
    "$size" \
    "$repeat" \
    "$workdir" \
    ./pathfinder \
    "$cols" \
    "$rows" \
    "$pyramid"
}


run_pathfinder_gib() {
  local gib="$1"
  local repeat="$2"

  local size

  size="$(pathfinder_size_for_gib "$gib")"

  run_pathfinder \
    "$size" \
    "$repeat"
}


# ============================================================
# Gram-Schmidt
# ============================================================

run_gramschm() {
  local size="$1"
  local repeat="$2"

  local workdir="$BENCH_ROOT/polybench/GRAMSCHM"

  local m
  local n

  IFS=':' read -r m n <<< "$size"

  if [[ -z "${m:-}" || -z "${n:-}" ]]; then
    echo \
      "Invalid gramschm size '$size'. Expected M:N" \
      >&2

    exit 2
  fi

  run_step \
    "gramschm" \
    "$size" \
    "$repeat" \
    "$workdir" \
    compile_gramschm \
    || return 0

  run_command \
    "gramschm" \
    "$size" \
    "$repeat" \
    "$workdir" \
    ./gramschmidt.exe \
    "$m" \
    "$n"
}


run_gramschm_gib() {
  local gib="$1"
  local repeat="$2"

  local size

  size="$(gramschm_size_for_gib "$gib")"

  run_gramschm \
    "$size" \
    "$repeat"
}


# ============================================================
# Print configuration
# ============================================================

echo
echo "============================================================"
echo "UVM benchmark configuration"
echo "============================================================"
echo "Output directory:  $OUT_DIR"
echo "Benchmarks:        $BENCHMARKS"
echo "Feature configs:   $FEATURE_CONFIGS"
echo "Repeats:           $REPEATS"
echo "GB sizes:          ${GB_SIZES:-<not set>}"

if [[ "$MAX_RUNTIME" -gt 0 ]]; then
  echo "Max runtime:       ${MAX_RUNTIME}s per benchmark"
else
  echo "Max runtime:       disabled"
fi

echo "Keep going:        $KEEP_GOING"
echo "Config file:       $CONFIG_FILE"
echo "Summary file:      $status_file"
echo "============================================================"
echo


# ============================================================
# Main benchmark loop
# ============================================================

for feature_config in "${FEATURE_CONFIG_LIST[@]}"; do

  set_feature_config "$feature_config"

  echo \
    "=== feature config: $FEATURE_CONFIG ($FEATURE_CPPFLAGS) ==="

  for benchmark in "${BENCHMARK_LIST[@]}"; do

    case "$benchmark" in

      kmeans)

        if [[ -n "$GB_SIZES" ]]; then

          for raw_gib in "${GB_SIZE_LIST[@]}"; do

            gib="$(normalize_gib "$raw_gib")"

            for ((repeat = 1; repeat <= REPEATS; repeat++)); do
              run_kmeans_gib \
                "$gib" \
                "$repeat"
            done

          done

        else

          for size in "${KMEANS_SIZE_LIST[@]}"; do

            for ((repeat = 1; repeat <= REPEATS; repeat++)); do
              run_kmeans \
                "$size" \
                "$repeat"
            done

          done

        fi
        ;;


      cnn)

        if [[ -n "$GB_SIZES" ]]; then

          for raw_gib in "${GB_SIZE_LIST[@]}"; do

            gib="$(normalize_gib "$raw_gib")"

            for ((repeat = 1; repeat <= REPEATS; repeat++)); do
              run_cnn_gib \
                "$gib" \
                "$repeat"
            done

          done

        else

          for size in "${CNN_SIZE_LIST[@]}"; do

            for ((repeat = 1; repeat <= REPEATS; repeat++)); do
              run_cnn \
                "$size" \
                "$repeat"
            done

          done

        fi
        ;;


      pathfinder)

        if [[ -n "$GB_SIZES" ]]; then

          for raw_gib in "${GB_SIZE_LIST[@]}"; do

            gib="$(normalize_gib "$raw_gib")"

            for ((repeat = 1; repeat <= REPEATS; repeat++)); do
              run_pathfinder_gib \
                "$gib" \
                "$repeat"
            done

          done

        else

          for size in "${PATHFINDER_SIZE_LIST[@]}"; do

            for ((repeat = 1; repeat <= REPEATS; repeat++)); do
              run_pathfinder \
                "$size" \
                "$repeat"
            done

          done

        fi
        ;;


      gramschm)

        if [[ -n "$GB_SIZES" ]]; then

          for raw_gib in "${GB_SIZE_LIST[@]}"; do

            gib="$(normalize_gib "$raw_gib")"

            for ((repeat = 1; repeat <= REPEATS; repeat++)); do
              run_gramschm_gib \
                "$gib" \
                "$repeat"
            done

          done

        else

          for size in "${GRAMSCHM_SIZE_LIST[@]}"; do

            for ((repeat = 1; repeat <= REPEATS; repeat++)); do
              run_gramschm \
                "$size" \
                "$repeat"
            done

          done

        fi
        ;;


      *)

        echo \
          "Unknown benchmark '$benchmark'. Expected kmeans,cnn,pathfinder,gramschm" \
          >&2

        exit 2
        ;;
    esac

  done

done


# ============================================================
# Finished
# ============================================================

echo
echo "============================================================"
echo "All benchmark attempts finished."
echo "============================================================"
echo "Output directory:"
echo "  $OUT_DIR"
echo
echo "Configuration:"
echo "  $CONFIG_FILE"
echo
echo "Summary:"
echo "  $status_file"
echo "============================================================"
