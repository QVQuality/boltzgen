import logging
from pathlib import Path
import subprocess as sp
import time

from fastapi import FastAPI, Form, HTTPException


BASE_DIR = Path("/workspace")
CACHE_DIR = Path("/cache")

# Published weights, resolved out of the image's Hugging Face cache rather than downloaded.
CHECKPOINTS = {
    "diverse": "huggingface:boltzgen/boltzgen-1:boltzgen1_diverse.ckpt",
    "adherence": "huggingface:boltzgen/boltzgen-1:boltzgen1_adherence.ckpt",
}

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="BoltzGen API")


@app.get("/v1/health/ready")
async def health_check():
    return {"status": "ready"}


@app.post("/api/generate")
async def generate(
    run_output_dirname: str = Form(...),
    spec_path: str = Form(...),
    num_designs: int = Form(50),
    protocol: str = Form("nanobody-anything"),
    checkpoint: str = Form("both"),
    sampling_steps: int = Form(500),
    recycling_steps: int = Form(3),
    step_scale: float = Form(None),
):
    """Run BoltzGen's backbone design step only.

    Inputs are read from the shared outputs volume instead of an upload, because a design
    spec refers to its target and its scaffolds by relative path.
    """
    run_dir = (BASE_DIR / run_output_dirname).resolve()
    if not str(run_dir).startswith(str(BASE_DIR)):
        raise HTTPException(status_code=400, detail="run_output_dirname escapes the workspace.")

    spec = (run_dir / spec_path).resolve()
    if not str(spec).startswith(str(run_dir)):
        raise HTTPException(status_code=400, detail="spec_path escapes the run directory.")
    if not spec.is_file():
        raise HTTPException(status_code=400, detail=f"Design spec not found: {spec}")

    if checkpoint not in {"both", *CHECKPOINTS}:
        raise HTTPException(status_code=400, detail=f"Unknown checkpoint `{checkpoint}`.")
    selected = list(CHECKPOINTS.values()) if checkpoint == "both" else [CHECKPOINTS[checkpoint]]

    output_dir = run_dir / "designs"
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "boltzgen", "run", str(spec),
        "--output", str(output_dir),
        "--steps", "design",
        "--num_designs", str(num_designs),
        "--protocol", protocol,
        "--cache", str(CACHE_DIR),
        "--design_checkpoints", *selected,
        "--config", "design",
        f"sampling_steps={sampling_steps}",
        f"recycling_steps={recycling_steps}",
    ]
    if step_scale is not None:
        command += ["--step_scale", str(step_scale)]

    start_time = time.time()
    try:
        logging.info(f"Running BoltzGen with command: {' '.join(command)}")
        _ = sp.run(command, check=True, cwd=str(run_dir))
    except sp.CalledProcessError as e:
        logging.error(f"BoltzGen failed with exit code {e.returncode}")
        raise HTTPException(status_code=500, detail=f"BoltzGen failed with exit code {e.returncode}")
    elapsed_time = time.time() - start_time

    designs = output_dir / "intermediate_designs"
    written = len(list(designs.glob("*.cif"))) if designs.is_dir() else 0
    if not written:
        raise HTTPException(
            status_code=500,
            detail=f"BoltzGen exited cleanly but wrote no designs into {designs}",
        )

    return {
        "status": "success",
        "output_dir": str(output_dir),
        "num_designs": written,
        "elapsed_time": elapsed_time,
    }
