from truss_train import (
    TrainingProject,
    TrainingJob,
    Image,
    Compute,
    Runtime,
    CacheConfig,
    CheckpointingConfig,
    LoadCheckpointConfig,
    BasetenCheckpoint,
)
from truss_train import definitions
from truss.base.truss_config import AcceleratorSpec

BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

training_runtime = Runtime(
    start_commands=["chmod +x ./run_grpo.sh && ./run_grpo.sh"],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
    },
    cache_config=CacheConfig(enabled=True),
    checkpointing_config=CheckpointingConfig(enabled=True),
    load_checkpoint_config=LoadCheckpointConfig(
        enabled=True,
        checkpoints=[
            BasetenCheckpoint.from_latest_checkpoint(
                project_name="cholec50-qwen3vl-4b-sft",
            ),
        ],
    ),
)

training_compute = Compute(
    accelerator=AcceleratorSpec(accelerator="H100", count=1),
)

training_job = TrainingJob(
    image=Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = TrainingProject(
    name="cholec50-qwen3vl-4b-grpo",
    job=training_job,
)
