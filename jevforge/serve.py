"""JevForge inference server implementing the public Jev HTTP contract.

POST /v1/systemone   {state, model, questions} -> {model, answers, usage}
GET  /health        liveness
GET  /v1/models     served checkpoint description

Every request/answer pair is appended to a JSONL log so live traffic can be
turned into training records — the data flywheel the project is built around.
Run: python -m jevforge.serve --checkpoint-dir runs/web_run --port 8123
"""
import argparse
import json
import time
from pathlib import Path

from .schema import QUESTION_TYPES, validate_question


def build_app(predictor, log_path, allow_cors=False):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="JevForge", version="1.0")
    if allow_cors:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                           allow_headers=["*"], allow_private_network=True)
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    model_name = f"jevforge/{predictor.base_name}"

    @app.get("/health")
    def health():
        return {"ready": True, "model": model_name, "calls": predictor.calls}

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": model_name, "owned_by": "jevforge",
                          "temperature": predictor.temperature}]}

    @app.get("/demo", response_class=HTMLResponse)
    def demo():
        page = Path(__file__).with_name("local_demo.html")
        return page.read_text(encoding="utf-8")

    @app.post("/v1/systemone")
    def systemone(payload: dict):
        started = time.perf_counter()
        if not isinstance(payload, dict) or set(payload) - {"state", "model", "questions"}:
            return JSONResponse(status_code=422, content={"error": "body must hold state, model, questions"})
        state, questions = payload.get("state"), payload.get("questions")
        if not isinstance(state, str) or not state.strip():
            if isinstance(state, (dict, list)):
                state = json.dumps(state, ensure_ascii=False, sort_keys=True)
            else:
                return JSONResponse(status_code=422, content={"error": "state must be text or JSON"})
        if not isinstance(questions, dict) or not questions:
            return JSONResponse(status_code=422, content={"error": "questions must be a non-empty object"})
        if len(questions) > 96:
            return JSONResponse(status_code=422, content={"error": "at most 96 questions per request"})
        paths = 0
        for qid, question in questions.items():
            try:
                validate_question(qid, question)
            except ValueError as exc:
                return JSONResponse(status_code=422, content={"error": str(exc)})
            paths += 2 if question["type"] == "noul" else len(question["criteria"])
        if paths > 512:
            return JSONResponse(status_code=422, content={"error": f"{paths} candidate paths exceed the 512 limit"})
        try:
            answers = predictor.decide(state, questions)
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"error": str(exc)})
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        response = {"model": model_name, "answers": answers,
                    "usage": {"input_tokens": paths, "output_tokens": 0},
                    "latency_ms": latency_ms}
        entry = {"ts": time.time(), "latency_ms": latency_ms,
                 "request": {"state": state, "questions": questions},
                 "response": response}
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return response

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--log-file", default="logs/jev_requests.jsonl")
    parser.add_argument("--device", default="auto",
                        help="auto selects CUDA, then Apple MPS, then CPU")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"),
                        default="auto")
    parser.add_argument("--allow-cors", action="store_true",
                        help="allow cross-origin requests (local demo pages)")
    args = parser.parse_args()

    from .predict import Predictor

    predictor = Predictor(args.checkpoint_dir, device_name=args.device,
                          precision=args.precision)
    app = build_app(predictor, args.log_file, allow_cors=args.allow_cors)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
