"""Unit test for common/runtime_env.resolve() with fake JDK / Hadoop folders (runs on any OS)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.runtime_env import resolve, java_major  # noqa: E402


def jdk(root, name, version):
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "bin"), exist_ok=True)
    with open(os.path.join(d, "release"), "w") as f:
        f.write(f'IMPLEMENTOR="Eclipse Adoptium"\nJAVA_VERSION="{version}"\n')
    return d


def hadoop(root, name, dll=True):
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "bin"), exist_ok=True)
    open(os.path.join(d, "bin", "winutils.exe"), "w").close()
    if dll:
        open(os.path.join(d, "bin", "hadoop.dll"), "w").close()
    return d


with tempfile.TemporaryDirectory() as r:
    j21 = jdk(r, "jdk-21.0.10", "21.0.10")
    j17 = jdk(r, "jdk-17.0.20.101-hotspot", "17.0.20.1")
    j8 = jdk(r, "jdk1.8.0_401", "1.8.0_401")
    h_nodll = hadoop(r, "spark-3.5.8-bin-hadoop3/hadoop", dll=False)   # the user's original HADOOP_HOME
    h_ok = hadoop(r, "hadoop")
    g = [os.path.join(r, "jdk-17*")]
    cases = [
        # (env, globs, hadoop candidates) -> expected (java_home, hadoop_home, n_problems)
        ("user laptop: JAVA_HOME=21, HADOOP_HOME without dll",
         {"JAVA_HOME": j21, "HADOOP_HOME": h_nodll}, g, [h_ok], (j17, h_ok, 0)),
        ("explicit overrides win", {"CLICKSTREAM_JAVA_HOME": j17, "JAVA_HOME": j21, "CLICKSTREAM_HADOOP_HOME": h_ok},
         [], [], (j17, h_ok, 0)),
        ("JAVA_HOME already 17", {"JAVA_HOME": j17, "HADOOP_HOME": h_ok}, [], [], (j17, h_ok, 0)),
        ("nothing usable", {"JAVA_HOME": j21, "HADOOP_HOME": h_nodll}, [], [], (None, None, 2)),
    ]
    ok = java_major(j8) == 8 and java_major(j17) == 17 and java_major(j21) == 21
    print(("OK " if ok else "BAD") + " java_major parses 1.8 / 17 / 21")
    for name, env, globs, hc, exp in cases:
        jh, hh, probs = resolve(env, globs, hc)
        got = (jh, hh, len(probs))
        ok &= got == exp
        print(("OK " if got == exp else "BAD") + f" {name}: got={got}")
    print("TEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
