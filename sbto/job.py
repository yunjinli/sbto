#!/usr/bin/env python3
import sys
import os
import shlex
import subprocess
import hydra


SWEEP_FLAGS = {"-m", "--multirun"}

def split_hydra_args(argv):
    """
    - sweep_triggers: flags that cause Hydra sweep in job.py (-m, --multirun, x=1,2)
    - hydra_args: all original Hydra overrides preserved
    - clean_argv: what Hydra in job.py should see (no sweep flags)
    """
    sweep_triggers = []
    clean_argv = []
    hydra_args = []  # original, complete hydra args passed by user

    for a in argv:
        hydra_args.append(a)

        # direct sweep flags
        if a in SWEEP_FLAGS:
            sweep_triggers.append(a)
            continue

        # sweep syntax (e.g. lr=1e-3,1e-4)
        if "=" in a and "," in a:
            sweep_triggers.append(a)
            continue

        clean_argv.append(a)

    return clean_argv, hydra_args, sweep_triggers


# --- Strip sweep triggers BEFORE Hydra sees them ---
clean_argv, hydra_args_full, sweep_triggers = split_hydra_args(sys.argv[1:])
sys.argv = [sys.argv[0]] + clean_argv
# ---------------------------------------------------


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # Hydra parsed config normally (no sweep)

    # Build Slurm sbatch arguments
    slurm = cfg.slurm
    slurm_args = [
        f"--partition={slurm.partition}",
        f"--clusters={slurm.clusters}",
        f"--nodes={slurm.nodes}",
        f"--cpus-per-task={slurm.cpus}",
        f"--time={slurm.time}",
        f"--mem={slurm.mem}",
        f"--job-name={slurm.jobname}",
        f"--output=slurm-%j.out",
        f"--chdir={os.getcwd()}",
    ]

    # Construct command for sbto/main.py
    python_script = "sbto/main.py"
    quoted_script = shlex.quote(python_script)

    # If sweep triggers were detected: add `-m` to job script
    if sweep_triggers:
        hydra_forward = ["-m"] + hydra_args_full
    else:
        hydra_forward = hydra_args_full

    quoted_args = " ".join(shlex.quote(a) for a in hydra_forward)

    # Environment setup
    conda_activate = (
        "source /dss/lrzsys/sys/spack/release/24.4.0/opt/x86_64/miniconda3/"
        "24.7.1-gcc-t6x7erm/bin/activate sbto"
    )

    wrapped_cmd = f"{conda_activate} && python3 {quoted_script} {quoted_args}"

    cmd = ["sbatch", f"--wrap={wrapped_cmd}"] + slurm_args

    print("Submitting job:")
    print(" ".join(shlex.quote(c) for c in cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
