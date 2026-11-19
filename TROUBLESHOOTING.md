# Troubleshooting

Problems seen on real installs, and the fixes for them. For first-time setup, see
[INSTALL.md](INSTALL.md).

## `ffs_viewer`: "Could not load the Qt platform plugin "xcb""

On a minimal Linux machine (a server or container image, a VNC desktop), the viewer aborts
right after `importing the viewer window...`:

```
qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed to load the Qt xcb platform plugin.
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
...
Aborted (core dumped)
```

The ffs and PySide6 install is fine. Qt's X11 display plugin (`xcb`) needs some system
libraries that desktop Ubuntu ships but minimal images leave out. Reinstalling the Python
packages won't help. Install the libraries with apt (Ubuntu/Debian; drop `sudo` if you are
root):

```bash
sudo apt-get update
sudo apt-get install -y libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 \
  libxkbcommon-x11-0 libxkbcommon0 libegl1 libgl1 libglib2.0-0 libfontconfig1 \
  libdbus-1-3 libxrender1 libxi6
```

The error message names only `libxcb-cursor0`, but a minimal image is usually missing
several of these, so installing the whole set saves repeating the cycle. If the viewer still
fails, ask Qt which library is still missing:

```bash
QT_DEBUG_PLUGINS=1 ffs_viewer 2>&1 | grep -i "cannot\|not found"
```

Also check that `echo $DISPLAY` prints a display such as `:1`. If it prints nothing, the
viewer has no screen to draw on. Run it inside the VNC/X session, or use `ssh -X`.

## CPU runs are slow, or seem to use one core

On CPU, the tools pick their thread count in this order (first match wins):

1. `-device cpu,N` on the command line
2. `$FFS_NUM_THREADS`
3. `$OMP_NUM_THREADS`
4. `$SLURM_CPUS_PER_TASK`
5. otherwise every core the process may run on (physical cores only, if nothing has
   restricted it), capped at the container's CPU quota when there is one

To see what you will get, and why:

```bash
echo "OMP=$OMP_NUM_THREADS FFS=$FFS_NUM_THREADS SLURM=$SLURM_CPUS_PER_TASK"
nproc
cat /sys/fs/cgroup/cpu.max 2>/dev/null
python -c "from fastfuncstuff.utils import resolve_cpu_threads as r; print(r())"
```

Two common causes:

- **`OMP_NUM_THREADS=1` from the environment.** Many managed images and clusters set this.
  ffs follows it on purpose, so it doesn't take over a shared machine.
- **A container CPU quota (Kubernetes and other managed servers).** If `cpu.max` prints
  something like `800000 100000`, the container gets 800000 / 100000 = **8 CPUs** of time,
  even when `nproc` shows 64. With 64 threads on an 8-CPU quota, the kernel pauses the
  process each time the quota runs out (throttling). That is far slower than 8 threads
  (`ffs_moco`, 2-CPU quota, 12 threads: 371 s for a run that takes 16 s), and in `top` it
  looks as if almost nothing is running. ffs reads the quota and caps its default thread
  count to it, but a thread count you set yourself (rules 1-4) is taken as given, so an
  `OMP_NUM_THREADS=64` baked into an image still overrides it.

Fix: set the thread count to the quota, or to what you want to give ffs:

```bash
export FFS_NUM_THREADS=8   # caps ffs only
export OMP_NUM_THREADS=8   # numpy and other libraries read this
```

Add the same lines to `~/.bashrc` to keep them. To override for a single run, use
`-device cpu,8`.

## AMD servers: the MKL speed shim (`FFS_NO_SHIM`)

torch's built-in math library (Intel MKL) runs its slower AVX2 code on any non-Intel CPU,
even one with AVX-512. On AMD Zen 4 and Zen 5 (EPYC 9004/9005, Ryzen 7000 and later) that
costs half or more of torch's matrix speed: on an EPYC 9555, float32 matrix multiply ran
119 GFLOP/s per core without the fix and 271 with it, and a 32-thread GLM fit ran 6.6x
faster.

The `ffs_*` tools fix this themselves. On Linux with an AVX-512 AMD CPU, a tool compiles a
small library once into `~/.cache/fastfuncstuff/` (this needs a C compiler; without one,
nothing changes) and restarts itself with it in `LD_PRELOAD`, before torch loads. Intel
CPUs, older AMD CPUs, Macs and ARM machines are left alone, because they gain nothing.
Python code that imports `fastfuncstuff` (notebooks, your own scripts) cannot restart
itself; start that Python with
`LD_PRELOAD=$(python -c 'from fastfuncstuff._mkl_shim import built_library as b; print(b())')`
to get the same speed.

To check it is active, run `python scripts/bench_cpu_host.py --quick --skip-jobs` with the
same environment and look for `mkl_avx512_shim=True`.

Turn it off with:

```bash
export FFS_NO_SHIM=1
```

Do this if a run hangs at 100% CPU, if you see `Intel oneMKL ERROR: Parameter 6 was
incorrect on entry to DLASWP`, or if results differ from another machine in a way that
matters to you. (They differ in the last digits, as they would between two CPUs.) The
hang is a torch bug in batched solves of matrices larger than 128x128 on MKL's AVX-512
path. ffs routes its own solves around it, but code that calls `torch.linalg.solve` on
such a batch directly, in the same process, can hit it.



A plain `pip install` on Linux pulls the CUDA build of torch, which brings the `nvidia-*`
and `cuda-*` packages with it. They do no harm, but a machine without a GPU can't use them.
To switch to the much smaller CPU build:

```bash
pip uninstall -y torch triton $(pip list --format=freeze | grep -E '^(nvidia-|cuda-)' | cut -d= -f1)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

To avoid it on a fresh install, install CPU torch before the package. See
[INSTALL.md](INSTALL.md#cpu-only-linux-skip-the-cuda-wheels).
