/* Let torch's built-in MKL use its AVX-512 kernels on AMD CPUs.
 *
 * MKL picks its code path from mkl_serv_cpu_detect(), and on a non-Intel CPU
 * that answers "AVX2" whatever the hardware supports. On a 32-CPU EPYC 9555
 * (Zen 5, full-width AVX-512) that left torch's float64 GEMM at 58 GFLOP/s per
 * core against OpenBLAS's 109 on the same box; with this preloaded it ran 135.
 * The pip torch wheel links MKL into libtorch_cpu.so, so the classic
 * mkl_serv_intel_cpu_true override cannot reach it -- but mkl_serv_cpu_detect is called through the PLT, so an
 * LD_PRELOAD definition wins.
 *
 * 7 is what MKL's own detection returns on an AVX-512 Intel CPU (measured on a
 * Skylake-W Xeon, torch 2.14 / MKL 2024.2): the Skylake-SP path, which needs
 * F/CD/BW/DQ/VL and nothing newer -- every AVX-512 AMD CPU (Zen 4, Zen 5) has
 * those. The forced value is only returned when this CPU reports all five;
 * otherwise MKL's own answer is passed through, so preloading this on a machine
 * without AVX-512 changes nothing. FFS_MKL_CPU_CODE=<n> overrides the value for
 * experiments.
 *
 * The ffs_* tools compile and preload this themselves on AVX-512 AMD CPUs
 * (fastfuncstuff/_mkl_shim.py; FFS_NO_SHIM=1 turns it off). To A/B by hand:
 *   so=$(python -c 'from fastfuncstuff._mkl_shim import built_library as b; print(b())')
 *   python scripts/bench_cpu_host.py --json base.json
 *   LD_PRELOAD=$so python scripts/bench_cpu_host.py --json shim.json
 *   python scripts/bench_cpu_host.py --compare base.json shim.json
 *
 * Results change in the last bits (different kernels, different summation
 * order), as they would between two CPUs.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>

typedef long (*detect_fn)(long, long, long, long);

static long forced = -1;

__attribute__((constructor)) static void choose(void) {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512cd") &&
        __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("avx512vl"))
        forced = 7;
    const char *env = getenv("FFS_MKL_CPU_CODE");
    if (env && *env)
        forced = atol(env);
}

static detect_fn real_detect(void) {
    /* torch loads libtorch_cpu RTLD_LOCAL, so RTLD_NEXT cannot see it; ask
     * the library by name, then fall back to a globally loaded libmkl. */
    void *h = dlopen("libtorch_cpu.so", RTLD_NOLOAD | RTLD_LAZY);
    detect_fn f = h ? (detect_fn)dlsym(h, "mkl_serv_cpu_detect") : NULL;
    return f ? f : (detect_fn)dlsym(RTLD_NEXT, "mkl_serv_cpu_detect");
}

/* MKL passes arguments we don't need; take them as integers and hand them on. */
long mkl_serv_cpu_detect(long a, long b, long c, long d) {
    if (forced >= 0)
        return forced;
    static detect_fn real;
    if (!real)
        real = real_detect();
    return real ? real(a, b, c, d) : 0; /* 0: MKL's generic path */
}
