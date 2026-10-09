#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(os.environ.get("CQC13_ROOT", "/home/ubuntu/cqc13-review"))
CACHE=ROOT/"v3/full_material_cache"
BATCHES=ROOT/"v3/full_batches"
FORMAL=BATCHES/"formal_review_manifest.json"
STATE=ROOT/"v3/pipeline_state.json"

def resolve_python():
    candidates=[
        os.environ.get("CQC13_PYTHON",""),
        "/home/ubuntu/.venv/notebooklm/bin/python",
        "/home/ubuntu/.hermes/hermes-agent/venv/bin/python",
        sys.executable,
    ]
    seen=set()
    for raw in candidates:
        if not raw or raw in seen:
            continue
        seen.add(raw)
        p=Path(raw)
        if not p.is_file() or not os.access(p,os.X_OK):
            continue
        probe=subprocess.run(
            [str(p),"-c","import pydantic,yaml; import engine.schema_v2"],
            cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
        )
        if probe.returncode==0:
            return str(p)
    raise RuntimeError("cqc13_python_runtime_unavailable")

PY=None

def get_python():
    global PY
    if PY is None:
        PY=resolve_python()
    return PY

def load(p, default=None):
    try: return json.load(open(p, encoding="utf8"))
    except Exception: return default

def atomic(p, obj):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t=p.with_suffix(p.suffix+".tmp")
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    os.replace(t,p)

def run(args, log):
    with open(log,"a",encoding="utf8") as f:
        f.write("\n$ "+" ".join(map(str,args))+"\n"); f.flush()
        r=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    return r.returncode

def jcount(d):
    return len([p for p in Path(d).glob("P*.json") if not p.name.endswith(".fail.json")])

def valid_results(d, ids):
    ok=set()
    for pid in ids:
        x=load(Path(d)/(pid+".json"))
        if isinstance(x,dict) and x.get("schema_version")=="2.0" and "raw_score" in x and not (x.get("_meta") or {}).get("error"):
            ok.add(pid)
    return ok

def write_state(**kw):
    base=load(STATE,{}) or {}
    # Once the pipeline leaves a technical blocker, stale error metadata must
    # disappear or status/reporting layers can misdiagnose a recovered run.
    phase=kw.get("phase")
    if phase and phase!="technical_blocker":
        for key in ("error","detail","retry_in_seconds","repeated_error_count"):
            base.pop(key,None)
    base.update({"schema_version":"3.0","updated_at":time.time(),**kw})
    atomic(STATE,base)

def batch_ids(all_ids, n, size):
    return all_ids[(n-1)*size:n*size]

def ensure_batch(n, ids):
    b=BATCHES/f"batch_{n:03d}"; b.mkdir(parents=True,exist_ok=True)
    for sub in ("results","gpt_review","gpt_review/blind_decisions","gpt_review/adjudication_decisions","gpt_review/expert_evidence","gpt_review/expert_decisions_inbox"):
        (b/sub).mkdir(parents=True,exist_ok=True)
    m=b/"batch_manifest.json"
    want={
        "schema":"hermes.task_scoped_batch.v1",
        "kind":"batch_manifest",
        "batch_number":n,
        "batch_size":40,
        "formal_review_count":200,
        "project_ids":ids,
        "cache_dir":str(CACHE),
    }
    cur=load(m)
    if cur:
        # Legacy batch_001 manifest predates hermes.task_scoped_batch.v1.
        # Its frozen project boundary is authoritative; metadata-shape drift
        # alone must not block later batches. Fail closed only on project IDs.
        if cur.get("project_ids") != ids:
            raise RuntimeError(f"batch_{n:03d}_manifest_project_drift")
    else:
        atomic(m,want)
    return b

def score(b, ids):
    results=b/"results"; good=valid_results(results,ids)
    if len(good)==len(ids): return True
    args=[get_python(),"-m","scripts.run_batch_v3","--cache-dir",str(CACHE),"--out",str(results),"--workers","4"]
    for pid in ids: args += ["--project-id",pid]
    rc=run(args,b/"runner.log")
    return rc==0 and len(valid_results(results,ids))==len(ids)

def route(b):
    g=b/"gpt_review"; m=load(g/"manifest.json")
    if m and int(m.get("n_results",0))==40 and len(m.get("packets_written",[]))==40: return m
    rc=run([get_python(),"-m","scripts.route_gpt_review_v3","--results-dir",str(b/"results"),"--cache-dir",str(CACHE),"--out-dir",str(g)],b/"route_watcher.log")
    if rc: raise RuntimeError("route_failed")
    return load(g/"manifest.json",{})

def reconcile_batch(n,b,ids):
    closure=load(b/"closure_summary_v3.json",{}) or {}
    if closure.get("business_final_closed") is True: return "closed"
    if not score(b,ids): return "score_retry"
    m=route(b); need=[e["project_id"] for e in m.get("entries",[]) if e.get("review_type")!="auto_pass"]
    g=b/"gpt_review"; blinds=g/"blind_decisions"
    have={Path(p).stem for p in glob.glob(str(blinds/"P*.json"))}
    if not set(need)<=have:
        write_state(phase="waiting_blind_review",batch=n,done=len(have & set(need)),expected=len(need)); return "wait"
    bi=g/"blind_import"
    run([get_python(),"-m","scripts.import_gpt_blind_review_v3","--results-dir",str(b/"results"),"--blind-dir",str(blinds),"--out-dir",str(bi)],b/"pipeline_driver.log")
    bs=load(bi/"blind_import_summary.json",{}) or {}
    if bs.get("n_errors",0): raise RuntimeError("blind_import_errors")
    packets={Path(p).stem for p in glob.glob(str(bi/"adjudication_packets/P*.json"))}
    adj=g/"adjudication_decisions"; havea={Path(p).stem for p in glob.glob(str(adj/"P*.json"))}
    if not packets<=havea:
        write_state(phase="waiting_adjudication",batch=n,done=len(havea & packets),expected=len(packets)); return "wait"
    ai=g/"adjudication_import"; ai.mkdir(parents=True,exist_ok=True)
    run([get_python(),"-m","scripts.import_gpt_adjudication_v3","--results-dir",str(b/"results"),"--adjudication-dir",str(adj),"--out",str(ai/"expert_queue.json")],b/"pipeline_driver.log")
    acc=load(ai/"accepted.json",{}) or {}
    if int(acc.get("n",0))!=len(packets): raise RuntimeError("adjudication_import_incomplete")
    run([get_python(),"-m","scripts.import_expert_decisions_v3","--gpt-dir",str(g),"--results-dir",str(b/"results")],b/"pipeline_driver.log")
    closure=load(b/"closure_summary_v3.json",{}) or {}
    if closure.get("business_final_closed") is True: return "closed"
    pending=((closure.get("expert_resolver") or {}).get("remaining_pending_ids") or [])
    write_state(phase="waiting_expert_resolution",batch=n,done=0,expected=len(pending),pending_ids=pending)
    return "wait"

def final_audit(formal):
    seen=[]; batches=[]
    for n in range(1,6):
        b=BATCHES/f"batch_{n:03d}"; c=load(b/"closure_summary_v3.json",{}) or {}; bm=load(b/"batch_manifest.json",{}) or {}
        ids=bm.get("project_ids",[]); seen+=ids
        batches.append({"batch":n,"closed":c.get("business_final_closed") is True,"count":len(ids),"pending":((c.get("expert_resolver") or {}).get("remaining_pending_ids") or [])})
    ok=len(seen)==200 and len(set(seen))==200 and seen==formal["project_ids"] and all(x["closed"] for x in batches)
    out={"schema_version":"3.0","formal_count":200,"unique_count":len(set(seen)),"matches_frozen_manifest":seen==formal["project_ids"],"batches":batches,"final_closed_ready":ok}
    atomic(ROOT/"v3/full_review_final_audit.json",out)
    write_state(phase="final_closed_ready" if ok else "final_audit_failed",batch=5,completed=200 if ok else len(seen))
    return ok

def reconcile_once():
    """Reconcile every batch independently.

    Waiting or batch-local failures MUST NOT stop unrelated batches. Only
    formal-manifest corruption is a global blocker. This preserves completed
    work and lets scoring/review preparation continue around quarantined work.
    """
    formal=load(FORMAL)
    if not formal or len(formal.get("project_ids",[]))!=200:
        raise RuntimeError("formal_200_manifest_invalid")
    ids=formal["project_ids"]; size=int(formal.get("batch_size",40))
    if size<=0 or len(set(ids))!=200:
        raise RuntimeError("formal_200_manifest_integrity_invalid")

    batch_status={}
    quarantined={}
    closed=[]
    total_scored=0

    for n in range(1,6):
        wanted=batch_ids(ids,n,size)
        try:
            b=ensure_batch(n,wanted)
            status=reconcile_batch(n,b,wanted)
            batch_status[str(n)]={"status":status}
            if status=="closed":
                closed.append(n)
        except Exception as e:
            # A corrupt/mismatched batch is quarantined. Other independent
            # batches continue, so one local fault cannot park the whole run.
            batch_status[str(n)]={
                "status":"quarantined",
                "error":type(e).__name__,
                "detail":str(e)[:400],
            }
            quarantined[str(n)]=batch_status[str(n)]
        total_scored += len(valid_results(BATCHES/f"batch_{n:03d}"/"results", wanted))

    if len(closed)==5:
        return final_audit(formal)

    first_open=next((n for n in range(1,6) if n not in closed), 1)
    write_state(
        phase="progressing_with_quarantine" if quarantined else "progressing",
        batch=first_open,
        current_batch=first_open,
        completed=len(closed)*size,
        total_scored=total_scored,
        closed_batches=closed,
        batch_status=batch_status,
        quarantined_batches=quarantined,
        status="wait",
        python_runtime=get_python(),
    )
    return False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--watch",action="store_true")
    ap.add_argument("--interval",type=int,default=30)
    args=ap.parse_args()
    interval=max(10,min(args.interval,300))
    if not args.watch:
        reconcile_once()
        return 0

    write_state(driver_mode="watch",driver_pid=os.getpid(),python_runtime=get_python())
    error_key=None
    error_count=0
    while True:
        try:
            done=reconcile_once()
            error_key=None
            error_count=0
            if done:
                write_state(driver_mode="complete",driver_pid=os.getpid(),python_runtime=get_python())
                return 0
            sleep_for=interval
        except Exception as e:
            key=f"{type(e).__name__}:{str(e)[:200]}"
            error_count=error_count+1 if key==error_key else 1
            error_key=key
            sleep_for=min(300,max(interval,interval*(2**min(error_count-1,4))))
            write_state(
                phase="technical_blocker",
                error=type(e).__name__,
                detail=str(e)[:400],
                retry_in_seconds=sleep_for,
                repeated_error_count=error_count,
                driver_mode="watch",
                driver_pid=os.getpid(),
                python_runtime=get_python(),
            )
        time.sleep(sleep_for)

if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception as e:
        write_state(phase="technical_blocker",error=type(e).__name__,detail=str(e)[:400],python_runtime=get_python())
        raise
