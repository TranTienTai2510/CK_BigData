"""Configure the process environment for local PySpark BEFORE the JVM starts.

On Windows, Spark needs a Java version it supports (17), winutils.exe + hadoop.dll on PATH, and no stray
SPARK_HOME pointing at another Spark install. A shell script (env.ps1) only fixes that for one terminal:
the VS Code debugger, the Run button and Jupyter do not see it. get_spark() calls configure() so every
entry point gets the same environment.

Resolution order (first match wins):
  JAVA_HOME   : $CLICKSTREAM_JAVA_HOME, $JAVA_HOME if it is a JDK 17, then common JDK 17 install folders.
  HADOOP_HOME : $CLICKSTREAM_HADOOP_HOME, $HADOOP_HOME if bin\\hadoop.dll exists, then D:\\hadoop, C:\\hadoop.
No-op on Linux/macOS (Docker images) except PYSPARK_PYTHON.
"""
import glob
import os
import re
import sys

JDK_GLOBS = [
    r"C:\Program Files\Eclipse Adoptium\jdk-17*",
    r"C:\Program Files\Java\jdk-17*",
    r"C:\Program Files\Microsoft\jdk-17*",
    r"C:\Program Files\Zulu\zulu-17*",
    r"C:\Program Files\Amazon Corretto\jdk17*",
]
HADOOP_CANDIDATES = [r"D:\hadoop", r"C:\hadoop"]
REQUIRED_JAVA_MAJOR = 17
REQUIRED_PYSPARK = "3.5."   # must match the Spark image used by the streaming stack


def check_pyspark():
    """Problem string if the interpreter's pyspark is not 3.5.x (e.g. a global 4.x install), else None."""
    try:
        import pyspark
    except ImportError:
        return f"pyspark is not installed for {sys.executable}"
    if not pyspark.__version__.startswith(REQUIRED_PYSPARK):
        return (f"pyspark {pyspark.__version__} found in {sys.executable}; this project needs {REQUIRED_PYSPARK}x. "
                "Activate the project venv (.\\.venv\\Scripts\\Activate.ps1) or: pip install pyspark==3.5.5")
    return None


def java_major(java_home):
    """Major version from <JAVA_HOME>/release (JAVA_VERSION="17.0.12"), or None."""
    try:
        with open(os.path.join(java_home, "release"), encoding="utf-8", errors="ignore") as f:
            m = re.search(r'JAVA_VERSION="(\d+)(?:\.(\d+))?', f.read())
    except OSError:
        return None
    if not m:
        return None
    major = int(m.group(1))
    return int(m.group(2)) if major == 1 and m.group(2) else major  # 1.8 -> 8


def has_hadoop_native(hadoop_home):
    b = os.path.join(hadoop_home, "bin")
    return os.path.isfile(os.path.join(b, "winutils.exe")) and os.path.isfile(os.path.join(b, "hadoop.dll"))


def resolve(env, jdk_globs=JDK_GLOBS, hadoop_candidates=HADOOP_CANDIDATES):
    """Pure function: return (java_home, hadoop_home, problems). Unit-testable on any OS."""
    problems = []
    java_home = None
    for cand in [env.get("CLICKSTREAM_JAVA_HOME"), env.get("JAVA_HOME")] + \
            sorted((p for g in jdk_globs for p in glob.glob(g)), reverse=True):
        if cand and java_major(cand) == REQUIRED_JAVA_MAJOR:
            java_home = cand
            break
    if not java_home:
        problems.append(f"JDK {REQUIRED_JAVA_MAJOR} not found (JAVA_HOME={env.get('JAVA_HOME')}). Install it: "
                        "winget install --id EclipseAdoptium.Temurin.17.JDK -e   "
                        "or set CLICKSTREAM_JAVA_HOME to a JDK 17 folder.")
    hadoop_home = None
    for cand in [env.get("CLICKSTREAM_HADOOP_HOME"), env.get("HADOOP_HOME")] + list(hadoop_candidates):
        if cand and has_hadoop_native(cand):
            hadoop_home = cand
            break
    if not hadoop_home:
        problems.append("winutils.exe + hadoop.dll not found in <HADOOP_HOME>\\bin (checked $CLICKSTREAM_HADOOP_HOME, "
                        f"$HADOOP_HOME={env.get('HADOOP_HOME')}, {', '.join(hadoop_candidates)}). Download both from "
                        "https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.6/bin into D:\\hadoop\\bin")
    return java_home, hadoop_home, problems


_done = False


def configure(verbose=True):
    global _done
    if _done:
        return
    bad = check_pyspark()
    if bad and os.environ.get("CLICKSTREAM_ALLOW_ANY_PYSPARK") != "1":
        raise RuntimeError(bad + "   (override: set CLICKSTREAM_ALLOW_ANY_PYSPARK=1)")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    if os.name != "nt":
        _done = True
        return
    from pyspark import SparkContext
    if SparkContext._active_spark_context is not None:
        print("[runtime_env] WARNING: a SparkContext is already running; environment changes have no effect",
              file=sys.stderr)
    java_home, hadoop_home, problems = resolve(os.environ)
    if problems:
        raise RuntimeError("Local Spark environment is not ready:\n  - " + "\n  - ".join(problems))
    removed = os.environ.pop("SPARK_HOME", None)   # use the jars bundled with the pip pyspark
    os.environ["JAVA_HOME"] = java_home
    os.environ["HADOOP_HOME"] = hadoop_home
    os.environ["PATH"] = os.pathsep.join([os.path.join(java_home, "bin"), os.path.join(hadoop_home, "bin"),
                                          os.environ.get("PATH", "")])
    os.environ["PYSPARK_PYTHON"] = os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    if verbose:
        import pyspark
        print(f"[runtime_env] pyspark={pyspark.__version__} JAVA_HOME={java_home} HADOOP_HOME={hadoop_home}"
              + (f" (ignored SPARK_HOME={removed})" if removed else ""), file=sys.stderr)
    _done = True


if __name__ == "__main__":   # python -m common.runtime_env  -> diagnose only
    if os.name != "nt":
        bad = check_pyspark()
        print(f"python      -> {sys.executable}")
        try:
            import pyspark
            print(f"pyspark     -> {pyspark.__version__}")
        except ImportError:
            pass
        print("JDK/Hadoop  -> not required for Linux/macOS")
        print("OK" if not bad else f"PROBLEMS:\n  - {bad}")
        sys.exit(1 if bad else 0)

    j, h, p = resolve(os.environ)
    bad = check_pyspark()
    if bad:
        p.append(bad)
    print(f"python      -> {sys.executable}")
    try:
        import pyspark
        print(f"pyspark     -> {pyspark.__version__}")
    except ImportError:
        pass
    print(f"JAVA_HOME   -> {j}  (major={java_major(j) if j else None})")
    print(f"HADOOP_HOME -> {h}")
    print(f"SPARK_HOME  -> {os.environ.get('SPARK_HOME')} (will be ignored)")
    print("OK" if not p else "PROBLEMS:\n  - " + "\n  - ".join(p))
    sys.exit(1 if p else 0)
