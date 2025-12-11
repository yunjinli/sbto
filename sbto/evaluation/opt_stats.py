def total_sim_timesteps(opt_stats_dict) -> int:
    return sum((
        stats["n_sim_steps_rollout"] for
        stats in
        opt_stats_dict["iterations"].values()
    ))