import csv

import numpy as np

np.set_printoptions(linewidth=230, precision=2, suppress=True)


def load(path, roi, K=7):
    rows = [
        r
        for r in csv.DictReader(open(path), delimiter="\t")
        if r["name"] == roi and int(r.get("K", K)) == K
    ]
    t = np.array(sorted({float(r["time"]) for r in rows}))
    v = np.zeros((len(t), K))
    ix = {x: i for i, x in enumerate(t)}
    col = list(rows[0])[-1]
    for r in rows:
        v[ix[float(r["time"])], int(r["depth"]) - 1] = float(r[col])
    return t, v


R = "/mnt/belegost/Projects/HiHiFaces/fine_outputs/ffs_autoproc_sub-3010_ses-fine_nonlmoco.results/lamroi/fit"
for roi in ("roi_lh_v1.lh", "roi_rh_ffa.rh"):
    for cond in ("fixate_hi", "gender_hi"):
        print(f"--- {roi} {cond}   (depth 7 = pial, depth 1 = deep)")
        for name, pfx in (
            ("orig TENT9 smooth-global", R),
            ("TENT9 OLS", "ols9"),
            ("CSPLINzero9", "cz9"),
            ("TENT17 OLS", "ols17"),
            ("TENTzero17 smooth-global", "tz17g"),
        ):
            try:
                t, v = load(f"{pfx}_iresp_{cond}.tsv", roi)
            except Exception as e:
                print(name, "missing", e)
                continue
            n = min(len(t), 12)
            print(
                "%-26s pial" % name,
                v[:n, 6],
                " min %.2f@%.1fs" % (v[:, 6].min(), t[v[:, 6].argmin()]),
                "peak %.2f@%.1fs" % (v[:, 6].max(), t[v[:, 6].argmax()]),
            )
            print(
                "%-26s deep" % "",
                v[:n, 0],
                " min %.2f@%.1fs" % (v[:, 0].min(), t[v[:, 0].argmin()]),
                "peak %.2f@%.1fs" % (v[:, 0].max(), t[v[:, 0].argmax()]),
                "| pial/deep peak %.2f" % (v[:, 6].max() / v[:, 0].max()),
            )
