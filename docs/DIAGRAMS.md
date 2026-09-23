# JevForge diagrams (Mermaid sources)

Rendered assets live in the separate
[JevForge web project](https://jev-forge.vercel.app). These Mermaid sources
are kept here so GitHub renders the architecture natively.

## Architecture - one forward, no decoding

```mermaid
flowchart LR
    S["state (canonical JSON)"] --> P["shared prefix: oj:state + oj:question"]
    P --> B["Qwen3.5-0.8B pretrained backbone (grad ckpt + SDPA, no LM head)"]
    C1["[e1] desc -> oj:answer"] --> B
    C2["[e2] desc -> oj:answer"] --> B
    C3["level i / yes-no"] --> B
    B --> H["shared scalar head -> z_k"]
    H --> D["softmax over z_1..z_K = complete distribution"]
    D --> O["choice / noul / score + confidence"]
```

## One-stop pipeline with the data flywheel

```mermaid
flowchart LR
    A[Mind2Web axtree] --> BW["build_web: Jev-contract records, website-disjoint splits"]
    BW --> SY["synthesize: generated supervision, streaming + cache"]
    SY --> TR["train: CE+Brier, cosine LR, best-step + temperature"]
    TR --> SV["serve: POST /v1/systemone"]
    TR --> EV["evaluate: top-1, Brier, ECE, per-site"]
    SV -. "logged requests = training data" .-> SY
    EV --> R["fixed test/ood: JevForge 0.579/0.637, Jev-1.13 0.543/0.610"]
```
