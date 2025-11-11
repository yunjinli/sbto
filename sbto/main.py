import hydra
from hydra.utils import instantiate
from typing import Optional

from sbto.sim.sim_base import SimRolloutBase
from sbto.tasks.task_base import OCPBase
from sbto.solvers.solver_base import SamplingBasedSolver, SolverState
from sbto.run.optimize import optimize_single_shooting
from sbto.run.save import save_results

def optimize_and_save_data(
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    description: str = "",
    hydra_rundir: str = "",
    save_fig: bool = True,
    init_state_solver: Optional[SolverState] = None 
    ) -> None:

    solver_state, all_samples, all_costs = optimize_single_shooting(
        sim,
        task,
        solver,
        init_state_solver
    )
    save_results(
        sim,
        task,
        solver_state,
        all_samples,
        all_costs,
        description,
        hydra_rundir,
        save_fig,
    )

def instantiate_from_cfg(cfg):
    sim = instantiate(cfg.task.sim)
    task = instantiate(cfg.task, sim=sim)
    solver = instantiate(cfg.solver, D=sim.Nvars_u)
    return sim, task, solver

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):

    sim, task, solver = instantiate_from_cfg(cfg)
    hydra_rundir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    optimize_and_save_data(
        sim,
        task,
        solver,
        cfg.description,
        hydra_rundir,
        cfg.save_fig,
    )
    
if __name__ == "__main__":
    main()