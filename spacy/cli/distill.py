from typing import Optional, Dict, Any, Union
from pathlib import Path
from wasabi import msg
import typer
import logging
import sys

from ._util import app, Arg, Opt, parse_config_overrides, show_validation_error
from ._util import import_code, setup_gpu
from ..pipeline.trainable_pipe import TrainablePipe
from ..schemas import ConfigSchemaDistill
from ..training.loop import distill as distill_nlp
from ..training.initialize import init_nlp_distill
from .. import util


@app.command(
    "distill",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def distill_cli(
    # fmt: off
    ctx: typer.Context,  # This is only used to read additional arguments
    teacher_model: str = Arg(..., help="Teacher model name or path"),
    student_config_path: Path = Arg(..., help="Path to config file", exists=True, allow_dash=True),
    output_path: Optional[Path] = Opt(None, "--output", "--output-path", "-o", help="Output directory to store trained pipeline in"),
    code_path: Optional[Path] = Opt(None, "--code", "-c", help="Path to Python file with additional code (registered functions) to be imported"),
    verbose: bool = Opt(False, "--verbose", "-V", "-VV", help="Display more information for debugging purposes"),
    use_gpu: int = Opt(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU")
    # fmt: on
):
    """
    Distill a spaCy pipeline from a teacher model.

    DOCS: https://spacy.io/api/cli#distill
    """
    util.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    overrides = parse_config_overrides(ctx.args)
    import_code(code_path)
    distill(
        teacher_model,
        student_config_path,
        output_path,
        use_gpu=use_gpu,
        overrides=overrides,
    )


def distill(
    teacher_model: str,
    student_config_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    use_gpu: int = -1,
    overrides: Dict[str, Any] = util.SimpleFrozenDict(),
):
    student_config_path = util.ensure_path(student_config_path)
    output_path = util.ensure_path(output_path)
    # Make sure all files and paths exists if they are needed
    if not student_config_path or (
        str(student_config_path) != "-" and not student_config_path.exists()
    ):
        msg.fail("Student config file not found", student_config_path, exits=1)
    if not output_path:
        msg.info("No output directory provided")
    else:
        if not output_path.exists():
            output_path.mkdir(parents=True)
            msg.good(f"Created output directory: {output_path}")
        msg.info(f"Saving to output directory: {output_path}")
    setup_gpu(use_gpu)
    teacher = util.load_model(teacher_model)
    with show_validation_error(student_config_path):
        config = util.load_config(
            student_config_path, overrides=overrides, interpolate=False
        )
    msg.divider("Initializing pipeline")
    with show_validation_error(student_config_path, hint_fill=False):
        student = init_nlp_distill(config, teacher, use_gpu=use_gpu)

    # link_rehearsal_models(teacher, student)

    msg.good("Initialized pipeline")
    msg.divider("Distilling pipeline")
    distill_nlp(
        teacher,
        student,
        output_path,
        use_gpu=use_gpu,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def link_rehearsal_models(teacher: "Language", student: "Language"):
    student_config = student.config.interpolate()
    D = util.registry.resolve(student_config["distill"], schema=ConfigSchemaDistill)
    pipe_map = D["pipe_map"]
    teacher_pipes = dict(teacher.pipeline)
    for name, pipe in student.pipeline:
        teacher_pipe_name = pipe_map[name] if name in pipe_map else name
        teacher_pipe = teacher_pipes[teacher_pipe_name]

        if not (isinstance(pipe, TrainablePipe) and hasattr(pipe, "_rehearsal_model")):
            raise ValueError(f"Cannot distill into {name}, pipe cannot rehearse")
        if not isinstance(teacher_pipe, TrainablePipe):
            raise ValueError(f"Teacher pipe is not a trainable pipe")

        pipe._rehearsal_model = teacher_pipe.model
