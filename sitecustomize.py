import os, sys, time, threading, signal
# Always enable faulthandler; register SIGUSR1 for thread dumps
try:
    import faulthandler
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
except Exception:
    pass

# Optional: lightweight call tracing (off by default)
TRACE_ON = os.getenv("TRACE_CALLS","0") == "1"
if TRACE_ON:
    TRACE_FILTER = os.getenv("TRACE_FILTER","")
    TRACE_MAX_DEPTH = int(os.getenv("TRACE_MAX_DEPTH","6"))
    TRACE_LOG = os.getenv("TRACE_LOG","trace_calls.log")
    filt = [p for p in TRACE_FILTER.split(",") if p]
    _fp = open(TRACE_LOG, "a", buffering=1, encoding="utf-8")
    t0 = time.time()
    def _match(path: str) -> bool:
        return (not filt) or any(s in path for s in filt)
    def _prof(frame, event, arg):
        if event != "call": return
        code = frame.f_code; fpath = code.co_filename or ""
        if not _match(fpath): return
        ts = time.time() - t0
        func = code.co_name
        mod  = frame.f_globals.get("__name__", "")
        ln   = frame.f_lineno
        tid  = threading.get_ident()
        d, f = 0, frame
        while f and d < TRACE_MAX_DEPTH:
            f = f.f_back; d += 1
        _fp.write(f"{ts:9.3f}s T{tid} {mod}:{func} ({fpath}:{ln})\n")
    sys.setprofile(_prof)
