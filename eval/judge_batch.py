import json, sys, os
sys.path.insert(0, "/root/autodl-tmp/visual_rag_agent")
os.chdir("/root/autodl-tmp/visual_rag_agent")
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.judge import judge_prediction_row, load_dotenv

load_dotenv()
if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "${DEEPSEEK_API_KEY}"

pf = sys.argv[1]
rows = [json.loads(l) for l in open(pf)]
todo = [r for r in rows if not isinstance(r.get("judge"), dict)]
print(f"file={pf} rows={len(rows)} to_judge={len(todo)}", flush=True)


def jf(r):
    try:
        r["judge"] = judge_prediction_row(r, model="deepseek-v4-flash", max_tokens=1024)
    except Exception as e:
        r["judge_error"] = str(e)[:120]
    return r


done = 0
with ThreadPoolExecutor(max_workers=32) as ex:
    futs = [ex.submit(jf, r) for r in todo]
    for _ in as_completed(futs):
        done += 1
        if done % 100 == 0:
            print(f"judged {done}/{len(todo)}", flush=True)

with open(pf, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
ok = sum(1 for r in rows if isinstance(r.get("judge"), dict) and r["judge"].get("correct") is True)
jd = sum(1 for r in rows if isinstance(r.get("judge"), dict))
err = sum(1 for r in rows if r.get("judge_error"))
print(f"JUDGE_DONE file_correct={ok}/{jd} errors={err}", flush=True)
