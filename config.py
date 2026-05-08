CONFIG = {
    "steps": 216,
    "ev_battery_capacity_kwh": 50.0,
    "ev_consumption_rate": 0.2,
    "ev_init_soc_mean": 0.80,
    "ev_init_soc_std": 0.28,
    "ev_request_threshold": 0.16,

    "mcs_speed_km_per_step": 4.0,
    "mcs_num": 20,
    "mcs_price_per_kwh": 1,
    "mcs_cost_per_km": 0.3,

}

CONFIG.update({
    "sim_start_hour": 6,
    "sim_end_hour": 24,
    "sim_step_minutes": 5,
})

CONFIG.update({
    "ev_init_soc_sigma_clip": 2.5,
    "max_charge_minutes": 20,
    "ev_request_timeout_minutes": 30,
})

CONFIG.update({
    "fcs_locations": [
        (104.057630, 30.618628),
        (104.068131, 30.651760),
        (104.059567, 30.599149),
        # (104.040650, 30.616954),
        (104.003988, 30.674417),
        # (104.083433, 30.652157),
        # (104.068204, 30.604719),
        (104.136933, 30.648734)
],

    "fcs_capacity": 10,
    "fcs_absorb_radius_km": 1.0,
    "ev_charge_soc_per_step": 0.216,
    "ev_target_soc": 0.8
})

CONFIG.update({
    "fcs_arrival_total_per_day": 2500,
    "fcs_arrival_base": 0.2,
    "fcs_arrival_morning_amp": 1.0,
    "fcs_arrival_evening_amp": 1.2,
    "fcs_arrival_morning_center_step": 24,   # 08:00
    "fcs_arrival_evening_center_step": 150,  # 18:30
    "fcs_arrival_morning_sigma": 8,
    "fcs_arrival_evening_sigma": 10,
    "fcs_arrival_station_weights": [1, 1, 1, 1, 1],
    "fcs_charge_minutes": 20,
})

CONFIG.update({
    "ev_count": 1000,
})

CONFIG.update({
    "dataset_path": "dataset/20140818.csv",
    "table_path": "dataset/table_20140818.csv",
    "result_path": "result/",
})


CONFIG.update({
    "mcs_service_radius_km": 3,
    "mcs_service_parallel_capacity": 1,
    "mcs_relocate_horizon_steps": 12,
    "mcs_reinforce_ev_per_step": 1.0,
    "mcs_relocate_hotspot_k": 15,
    "mcs_relocate_hotspot_max_iter": 30,
    "mcs_relocate_hotspot_sample_size": 30000,
    "mcs_relocate_hotspot_recompute_each_reset": False,
})

CONFIG.update({
    "use_lstm_summary": True,
    "lstm_predictor_ckpt": "result/predictor/lstm_predictor.pt",
    "lstm_predictor_device": "cuda",
    "observation_schema": "ppo16",
    "ppo_obs_radius_km": 3.0,
    "ppo_future_horizon_steps": 12,
    "ppo_fcs_high_risk_threshold": 1.0,
    "ppo_pred_demand_scale": 50.0,
    "ppo_obs_req_norm": 20.0,
})

CONFIG.update({
    "WEST": 103.9808,
    "SOUTH": 30.5963,
    "EAST": 104.1614,
    "NORTH": 30.7291,
})

CONFIG.update({
    # Align optimization target toward higher service success and lower wait.
    "reward_service_reward": 6.0,
    "reward_fast_service_bonus": 3.0,
    "reward_waiting_penalty": 0.18,
    "reward_serve_wait_penalty": 0.10,
    "reward_timeout_penalty": 6.0,
    "reward_timeout_wait_penalty": 0.12,
    "reward_pending_count_penalty": 0.025,
    "reward_empty_drive_penalty": 0.012,
    "reward_fcs_overload_penalty": 0.10,
    "reward_crowd_penalty": 0.03,
    "reward_invalid_action_penalty": 3.0,
    "reward_success_rate_bonus": 2.0,
    "reward_mcs_success_rate_bonus": 1.0,
    "reward_wait_improvement_bonus": 0.8,
    "reward_income_scale": 0.02,

    "reward_shape_relocate_scale": 0.08,
    "reward_shape_reinforce_scale": 0.10,
    "reward_shape_stay_scale": 0.05,
    "reward_shape_clip": 0.15,
    "reward_shape_value_congestion_weight": 0.5,
    "reward_shape_stay_demand_thr": 0.6,
    "reward_shape_stay_cong_thr": 0.8,

    "reward_clip_abs": 12.0
})
