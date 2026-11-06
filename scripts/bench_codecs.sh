#!/usr/bin/env bash
# Time gzip and zstd tools on NIfTI files from the command line alone: no Python in
# the timed path, so nothing about how the toolbox calls a codec can tilt the result.
#
#   scripts/bench_codecs.sh -o codecs.csv -t 6 -r 5 -w /dev/shm/codecs a.nii.gz b.nii.zst ...
#   scripts/bench_codecs.sh -s codecs.csv          # medians from an existing CSV
#
# Each input is first decoded to a plain .nii in the work directory (use a RAM disk:
# /dev/shm on Linux; on macOS make one with hdiutil and pass it with -w). Then, every
# repetition, in a freshly shuffled order:
#   compress   plain .nii -> file in the work dir, one row per tool and level
#   decompress each compressed file -> /dev/null, one row per decoder
# Every decoder is checked once against the plain bytes (cksum) before timing.
#
# Tools used if found: igzip (ISA-L's own CLI; -T threads), pigz, zstd, pzstd,
# rapidgzip. ISA-L's CLI comes from the isal package (apt install isal, brew install
# isa-l). Without it the script times "$PYTHON -m isal.igzip", which cannot use
# threads, and labels those rows igzip-py so they are never mistaken for the real one.
#
# Needs bash 5 for $EPOCHREALTIME (macOS ships 3.2: use Homebrew's bash).
set -euo pipefail

out=codecs.csv threads=$(( $(getconf _NPROCESSORS_ONLN) / 2 )) reps=5 work="" summary_only=""
levels_igzip="1 2 3" levels_zstd="1 3" levels_pigz="1"
usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while getopts "o:t:r:w:s:h" opt; do
    case $opt in
        o) out=$OPTARG ;; t) threads=$OPTARG ;; r) reps=$OPTARG ;; w) work=$OPTARG ;;
        s) summary_only=$OPTARG ;; h) usage ;; *) usage 1 ;;
    esac
done
shift $((OPTIND - 1))

summarize() {
    # Median wall per (file, op, method); size is the compressed size as % of plain.
    awk -F, 'NR > 1 { k = $1 FS $2 FS $3; t[k] = t[k] " " $5; s[k] = $6; p[k] = $7 }
        END { for (k in t) print k "," t[k] "," s[k] "," p[k] }' "$1" |
    while IFS=, read -r file op method times bytes plain; do
        med=$(tr ' ' '\n' <<<"$times" | grep . | sort -g | awk '{ a[NR] = $1 }
            END { print (NR % 2) ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2 }')
        printf '%s,%s,%s,%.3f,%.1f\n' "$file" "$op" "$method" "$med" "$(awk "BEGIN { print 100 * $bytes / $plain }")"
    done | sort -t, -k1,1 -k2,2r -k4,4g |
    awk -F, 'BEGIN { print "file                        op          method                          seconds  size%" }
        { printf "%-27s %-11s %-30s %8.3f  %5.1f\n", $1, $2, $3, $4, ($2 == "compress" ? $5 : 0) }'
}
[[ -n $summary_only ]] && { summarize "$summary_only"; exit 0; }
(( $# )) || usage 1
[[ ${BASH_VERSINFO[0]} -ge 5 ]] || { echo "needs bash 5 (for \$EPOCHREALTIME)" >&2; exit 1; }
if [[ -z $work ]]; then
    [[ -d /dev/shm ]] || { echo "no /dev/shm: pass a RAM disk with -w" >&2; exit 1; }
    work=/dev/shm/bench_codecs
fi
mkdir -p "$work"
work=$(mktemp -d "$work/run.XXXXXX")
trap 'rm -rf "$work"' EXIT
have() { command -v "$1" >/dev/null; }

# name|compress command (reads $src, writes stdout)|extension
encoders=()
if have igzip; then
    for l in $levels_igzip; do encoders+=("igzip-$l.T$threads|igzip -c -$l -T $threads|gz"); done
    igz_dec="igzip -d -c"
else
    py=${PYTHON:-$(command -v python3)}
    for l in $levels_igzip; do encoders+=("igzip-py-$l.T1|$py -m isal.igzip -c -$l|gz"); done
    igz_dec="$py -m isal.igzip -d -c"
    echo "note: no igzip CLI; timing python -m isal.igzip, single-threaded (rows igzip-py)" >&2
fi
have pigz && for l in $levels_pigz; do encoders+=("pigz-$l.p$threads|pigz -c -$l -p $threads|gz"); done
have pzstd && for l in $levels_zstd; do encoders+=("pzstd-$l.p$threads|pzstd -q -c -$l -p $threads|zst"); done
have zstd && for l in $levels_zstd; do encoders+=("zstd-$l.T$threads|zstd -q -c -$l -T$threads|zst"); done

# name|decompress command (reads $src, writes stdout)|extension it reads
decoders=("igzip.1|$igz_dec|gz")
have pigz && decoders+=("pigz|pigz -d -c -p $threads|gz")
have rapidgzip && decoders+=("rapidgzip.P$threads|rapidgzip -d -c -q -P $threads|gz")
have zstd && decoders+=("zstd.1|zstd -q -d -c|zst")
have pzstd && decoders+=("pzstd.p$threads|pzstd -q -d -c -p $threads|zst")

timed() {  # timed <stdout file> <command...>: prints wall seconds
    local t0=$EPOCHREALTIME
    "${@:2}" >"$1"
    awk "BEGIN { printf \"%.4f\", $EPOCHREALTIME - $t0 }"
}
shuffled() { printf '%s\n' "$@" | awk 'BEGIN { srand() } { print rand() "\t" $0 }' | sort -n | cut -f2-; }

{
    echo "# $(uname -sm), $(getconf _NPROCESSORS_ONLN) logical CPUs, threads=$threads, reps=$reps, $(date -u +%FT%TZ)"
    for t in igzip pigz zstd pzstd rapidgzip; do
        have $t && echo "# $t: $($t --version 2>&1 | head -1)"
    done
} >"$out.meta"
echo "file,op,method,rep,seconds,bytes,plain_bytes,load1" >"$out"
load1() { uptime | awk -F'load average[s]*: ' '{ split($2, a, /[, ]+/); print a[1] }'; }

for input in "$@"; do
    label=$(basename "$input"); label=${label%.nii*}
    plain=$work/$label.nii
    case $input in
        *.gz) rapidgzip -d -c -q "$input" >"$plain" 2>/dev/null || gzip -dc "$input" >"$plain" ;;
        *.zst) zstd -q -d -c "$input" >"$plain" ;;
        *) cp "$input" "$plain" ;;
    esac
    plain_bytes=$(wc -c <"$plain" | tr -d ' ')
    plain_sum=$(cksum <"$plain")
    echo "$label: $((plain_bytes / 1000000)) MB" >&2

    # Compress once, untimed, so every decoder can be verified before any timing.
    for e in "${encoders[@]}"; do
        IFS='|' read -r name cmd ext <<<"$e"
        src=$plain; eval "$cmd" '"$src"' >"$work/$label.$name.$ext"
    done
    for d in "${decoders[@]}"; do
        IFS='|' read -r dname dcmd dext <<<"$d"
        for f in "$work/$label".*."$dext"; do
            src=$f
            [[ $(eval "$dcmd" '"$src"' | cksum) == "$plain_sum" ]] || { echo "$dname misread $f" >&2; exit 1; }
        done
    done

    for ((rep = 0; rep < reps; rep++)); do
        while IFS= read -r e; do
            IFS='|' read -r name cmd ext <<<"$e"
            dst=$work/$label.$name.$ext src=$plain
            s=$(timed "$dst" eval "$cmd \"\$src\"")
            echo "$label,compress,$name,$rep,$s,$(wc -c <"$dst" | tr -d ' '),$plain_bytes,$(load1)" >>"$out"
        done < <(shuffled "${encoders[@]}")
        # Each decoder reads every file of its format: who wrote a gzip changes how fast it reads.
        jobs=()
        for d in "${decoders[@]}"; do
            IFS='|' read -r dname dcmd dext <<<"$d"
            for f in "$work/$label".*."$dext"; do jobs+=("$dname|$dcmd|$f"); done
        done
        while IFS= read -r j; do
            IFS='|' read -r dname dcmd f <<<"$j"
            writer=${f#"$work/$label."}; writer=${writer%.*}
            src=$f; s=$(timed /dev/null eval "$dcmd \"\$src\"")
            echo "$label,decompress,$writer>$dname,$rep,$s,$(wc -c <"$f" | tr -d ' '),$plain_bytes,$(load1)" >>"$out"
        done < <(shuffled "${jobs[@]}")
        echo "  rep $((rep + 1))/$reps" >&2
    done
    rm -f "$work/$label".*
done
summarize "$out"
