from __future__ import annotations

"""
Pipeline:
1) (optional) train LSTM predictor
2) (optional) train online policy (PPO or MAPPO)
3) (optional) build offline dataset from online rollouts
4) (optional) train offline DT
5) (optional) fine-tune DT with online PPO+KL
6) (optional) train MAPPO with frozen uncertainty-gated DT prior
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end LSTM -> MAPPO/PPO -> Offline DT -> DT prior -> DT-gated MAPPO pipeline.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-lstm-summary", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lstm-predictor-ckpt", type=str, default="")

    p.add_argument("--skip-lstm-train", action="store_true")
    p.add_argument("--lstm-outdir", type=str, default="result/predictor")
    p.add_argument("--lstm-episodes", type=int, default=400)
    p.add_argument("--lstm-epochs", type=int, default=300)
    p.add_argument("--lstm-early-stop-patience", type=int, default=30)
    p.add_argument("--lstm-max-steps", type=int, default=None)
    p.add_argument("--lstm-policy", type=str, default="random", choices=["random", "stay"])
    p.add_argument("--lstm-seq-len", type=int, default=12)
    p.add_argument("--lstm-future-horizon", type=int, default=6)
    p.add_argument("--lstm-batch-size", type=int, default=256)

    p.add_argument("--skip-ppo-train", action="store_true")
    p.add_argument("--rl-algo", type=str, default="mappo", choices=["ppo", "mappo"])
    p.add_argument("--ppo-ckpt", type=str, default="")
    p.add_argument("--ppo-outdir", type=str, default="result/ppo_for_offline")
    p.add_argument("--ppo-epochs", type=int, default=500)
    p.add_argument("--ppo-episodes-per-epoch", type=int, default=2)
    p.add_argument("--ppo-save-epoch-interval", type=int, default=0)
    p.add_argument("--ppo-max-steps", type=int, default=None)
    p.add_argument("--ppo-eval-every", type=int, default=1)
    p.add_argument("--ppo-eval-episodes", type=int, default=20)

    p.add_argument("--skip-offline-build", action="store_true")
    p.add_argument("--offline-output", type=str, default="dataset/offline_ppo_hq_traj.npz")
    p.add_argument("--offline-episodes", type=int, default=600)
    p.add_argument("--offline-per-source-episodes", type=int, default=0)
    p.add_argument("--offline-ppo-ckpts", type=str, default="", help="Comma-separated checkpoint paths/patterns for offline rollout.")
    p.add_argument("--offline-env-seeds", type=str, default="", help='Comma-separated rollout seeds, e.g. "42,43,44".')
    p.add_argument("--offline-stratified-mix", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--offline-return-bins", type=int, default=3)
    p.add_argument("--offline-stage-bins", type=int, default=3)
    p.add_argument("--offline-max-trajs", type=int, default=12000)
    p.add_argument("--offline-max-steps", type=int, default=None)
    p.add_argument("--offline-trajs-per-episode", type=int, default=1)
    p.add_argument("--offline-traj-selection", type=str, default="top_return", choices=["top_return", "random"])
    p.add_argument("--offline-event-level", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--offline-min-return-quantile", type=float, default=0.25)
    p.add_argument("--offline-log-interval", type=int, default=20)

    p.add_argument("--skip-dt-train", action="store_true")
    p.add_argument("--dt-outdir", type=str, default="result/dt_offline")
    p.add_argument("--dt-context-len", type=int, default=20)
    p.add_argument("--dt-epochs", type=int, default=80)
    p.add_argument("--dt-steps-per-epoch", type=int, default=400)
    p.add_argument("--dt-max-trajs", type=int, default=8000)
    p.add_argument("--dt-early-stop-patience", type=int, default=15)

    p.add_argument("--skip-ft-train", action="store_true")
    p.add_argument("--ft-outdir", type=str, default="result/dt_ppo_ft")
    p.add_argument("--ft-epochs", type=int, default=300)
    p.add_argument("--ft-episodes-per-epoch", type=int, default=4)
    p.add_argument("--ft-max-steps", type=int, default=None)
    p.add_argument("--ft-eval-every", type=int, default=1)
    p.add_argument("--ft-eval-episodes", type=int, default=20)
    p.add_argument("--ft-reward-profile", type=str, default="config", choices=["business", "config"])
    p.add_argument("--ft-policy-warmup-epochs", type=int, default=10)
    p.add_argument("--ft-stability-guard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--target-return", type=float, default=0.0, help="<=0 means auto from offline dataset max RTG * scale.")
    p.add_argument("--target-return-scale", type=float, default=1.2)

    p.add_argument("--skip-gate-train", action="store_true")
    p.add_argument("--gate-outdir", type=str, default="result/mappo_dt_gate")
    p.add_argument("--gate-init-mappo-ckpt", type=str, default="", help="Defaults to PPO/MAPPO best_by_business checkpoint when available.")
    p.add_argument("--gate-dt-ckpt", type=str, default="", help="Defaults to DT finetune best_by_business checkpoint.")
    p.add_argument("--gate-epochs", type=int, default=50)
    p.add_argument("--gate-episodes-per-epoch", type=int, default=4)
    p.add_argument("--gate-max-steps", type=int, default=None)
    p.add_argument("--gate-eval-every", type=int, default=1)
    p.add_argument("--gate-eval-episodes", type=int, default=20)
    p.add_argument("--gate-eval-max-steps", type=int, default=None)
    p.add_argument("--gate-threshold", type=float, default=0.70)
    p.add_argument("--gate-alpha", type=float, default=0.10)
    p.add_argument("--gate-soft-gate", action="store_true")
    p.add_argument("--gate-temperature", type=float, default=0.05)
    p.add_argument("--gate-target-return", type=float, default=0.0, help="Override DT prior target return. <=0 uses checkpoint/default scale.")
    p.add_argument("--gate-target-return-scale", type=float, default=1.2, help="Used when gate DT checkpoint is offline dt_best.pt.")
    p.add_argument("--gate-lr-actor", type=float, default=1e-4)
    p.add_argument("--gate-lr-critic", type=float, default=3e-4)
    p.add_argument("--gate-ppo-clip", type=float, default=0.15)
    p.add_argument("--gate-update-epochs", type=int, default=4)
    p.add_argument("--gate-mini-batch-size", type=int, default=256)
    return p.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("[pipeline] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    py = sys.executable

    lstm_ckpt = Path(args.lstm_predictor_ckpt) if args.lstm_predictor_ckpt else Path(args.lstm_outdir) / "lstm_predictor.pt"
    if bool(args.use_lstm_summary):
        if not args.skip_lstm_train:
            cmd = [
                py,
                "train_lstm_predictor.py",
                "--outdir",
                args.lstm_outdir,
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                "--episodes",
                str(args.lstm_episodes),
                "--epochs",
                str(args.lstm_epochs),
                "--policy",
                str(args.lstm_policy),
                "--seq-len",
                str(args.lstm_seq_len),
                "--future-horizon",
                str(args.lstm_future_horizon),
                "--batch-size",
                str(args.lstm_batch_size),
                "--early-stop-patience",
                str(args.lstm_early_stop_patience),
            ]
            if args.lstm_max_steps is not None:
                cmd.extend(["--max-steps", str(args.lstm_max_steps)])
            run_cmd(cmd)
        if not lstm_ckpt.exists():
            raise RuntimeError(f"LSTM checkpoint not found: {lstm_ckpt}")

    rl_script = "train_mappo.py" if str(args.rl_algo).lower() == "mappo" else "train_ppo.py"
    ppo_ckpt = Path(args.ppo_ckpt) if args.ppo_ckpt else Path(args.ppo_outdir) / "best.pt"
    if not args.skip_ppo_train:
        cmd = [
            py,
            rl_script,
            "--outdir",
            args.ppo_outdir,
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--epochs",
            str(args.ppo_epochs),
            "--episodes-per-epoch",
            str(args.ppo_episodes_per_epoch),
            "--eval-every",
            str(args.ppo_eval_every),
            "--eval-episodes",
            str(args.ppo_eval_episodes),
        ]
        if int(args.ppo_save_epoch_interval) > 0:
            cmd.extend(["--save-epoch-interval", str(args.ppo_save_epoch_interval)])
        if args.ppo_max_steps is not None:
            cmd.extend(["--max-steps", str(args.ppo_max_steps)])
        if bool(args.use_lstm_summary):
            cmd.append("--use-lstm-summary")
            cmd.extend(["--lstm-predictor-ckpt", str(lstm_ckpt)])
        run_cmd(cmd)
    if not ppo_ckpt.exists():
        raise RuntimeError(f"Online policy checkpoint not found: {ppo_ckpt}")

    offline_path = Path(args.offline_output)
    if not args.skip_offline_build:
        if args.offline_ppo_ckpts:
            ppo_ckpts_arg = str(args.offline_ppo_ckpts)
        else:
            ckpt_candidates: list[Path] = []
            seen_ckpts: set[str] = set()

            # Use the best business checkpoint by default. DT is trained by
            # behavior cloning, so mixing early exploration into the default
            # dataset usually dilutes the policy it can imitate.
            ckpt_roles = [
                [Path(args.ppo_outdir) / "best_by_business.pt", ppo_ckpt],
            ]
            for role_paths in ckpt_roles:
                for p in role_paths:
                    key = str(p)
                    if p.exists() and key not in seen_ckpts:
                        ckpt_candidates.append(p)
                        seen_ckpts.add(key)
                        break

            if not ckpt_candidates:
                key = str(ppo_ckpt)
                if key not in seen_ckpts:
                    ckpt_candidates.append(ppo_ckpt)
                    seen_ckpts.add(key)
                ppo_last = Path(args.ppo_outdir) / "last.pt"
                key_last = str(ppo_last)
                if ppo_last.exists() and key_last not in seen_ckpts:
                    ckpt_candidates.append(ppo_last)
                    seen_ckpts.add(key_last)

            ppo_ckpts_arg = ",".join(str(x) for x in ckpt_candidates)
        cmd = [
            py,
            "build_offline_dataset_ppo.py",
            "--ppo-ckpts",
            ppo_ckpts_arg,
            "--output",
            args.offline_output,
            "--episodes",
            str(args.offline_episodes),
            "--per-source-episodes",
            str(args.offline_per_source_episodes),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--return-bins",
            str(args.offline_return_bins),
            "--stage-bins",
            str(args.offline_stage_bins),
            "--window-len",
            str(args.dt_context_len),
            "--trajs-per-episode",
            str(args.offline_trajs_per_episode),
            "--traj-selection",
            str(args.offline_traj_selection),
            "--min-return-quantile",
            str(args.offline_min_return_quantile),
            "--log-interval",
            str(args.offline_log_interval),
        ]
        cmd.append("--event-level" if bool(args.offline_event_level) else "--no-event-level")
        if bool(args.offline_stratified_mix):
            cmd.append("--stratified-mix")
        if args.offline_env_seeds:
            cmd.extend(["--env-seeds", str(args.offline_env_seeds)])
        if int(args.offline_max_trajs) > 0:
            cmd.extend(["--max-trajs", str(args.offline_max_trajs)])
        if args.offline_max_steps is not None:
            cmd.extend(["--max-steps", str(args.offline_max_steps)])
        if bool(args.use_lstm_summary):
            cmd.append("--use-lstm-summary")
            cmd.extend(["--lstm-predictor-ckpt", str(lstm_ckpt)])
        run_cmd(cmd)
    if not offline_path.exists():
        raise RuntimeError(f"Offline dataset not found: {offline_path}")

    if not args.skip_dt_train:
        cmd = [
            py,
            "train_dt_offline.py",
            "--dataset",
            args.offline_output,
            "--outdir",
            args.dt_outdir,
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--epochs",
            str(args.dt_epochs),
            "--steps-per-epoch",
            str(args.dt_steps_per_epoch),
            "--max-trajs",
            str(args.dt_max_trajs),
            "--context-len",
            str(args.dt_context_len),
            "--early-stop-patience",
            str(args.dt_early_stop_patience),
        ]
        run_cmd(cmd)

    dt_ckpt = Path(args.dt_outdir) / "dt_best.pt"
    if not args.skip_ft_train:
        if not dt_ckpt.exists():
            raise RuntimeError(f"DT checkpoint not found: {dt_ckpt}")
        cmd = [
            py,
            "train_dt_ppo_finetune.py",
            "--dt-ckpt",
            str(dt_ckpt),
            "--outdir",
            args.ft_outdir,
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--target-return",
            str(args.target_return),
            "--target-return-scale",
            str(args.target_return_scale),
            "--offline-dataset",
            args.offline_output,
            "--epochs",
            str(args.ft_epochs),
            "--episodes-per-epoch",
            str(args.ft_episodes_per_epoch),
            "--eval-every",
            str(args.ft_eval_every),
            "--eval-episodes",
            str(args.ft_eval_episodes),
            "--context-len",
            str(args.dt_context_len),
            "--reward-profile",
            str(args.ft_reward_profile),
            "--policy-warmup-epochs",
            str(args.ft_policy_warmup_epochs),
        ]
        cmd.append("--stability-guard" if bool(args.ft_stability_guard) else "--no-stability-guard")
        if args.ft_max_steps is not None:
            cmd.extend(["--max-steps", str(args.ft_max_steps)])
        if bool(args.use_lstm_summary):
            cmd.append("--use-lstm-summary")
            cmd.extend(["--lstm-predictor-ckpt", str(lstm_ckpt)])
        run_cmd(cmd)

    if not args.skip_gate_train:
        gate_init_mappo = Path(args.gate_init_mappo_ckpt) if args.gate_init_mappo_ckpt else Path(args.ppo_outdir) / "best_by_business.pt"
        if not gate_init_mappo.exists():
            gate_init_mappo = ppo_ckpt
        if not gate_init_mappo.exists():
            raise RuntimeError(f"Gate init MAPPO checkpoint not found: {gate_init_mappo}")

        if args.gate_dt_ckpt:
            gate_dt_ckpt = Path(args.gate_dt_ckpt)
        elif args.skip_ft_train:
            gate_dt_ckpt = dt_ckpt
        else:
            finetuned_gate_dt = Path(args.ft_outdir) / "best_by_business.pt"
            gate_dt_ckpt = finetuned_gate_dt if finetuned_gate_dt.exists() else dt_ckpt
        if not gate_dt_ckpt.exists():
            raise RuntimeError(
                f"Gate DT checkpoint not found: {gate_dt_ckpt}. "
                "Run DT training/finetune first or pass --gate-dt-ckpt."
            )

        cmd = [
            py,
            "train_mappo_dt_gate.py",
            "--outdir",
            args.gate_outdir,
            "--init-mappo-ckpt",
            str(gate_init_mappo),
            "--dt-ckpt",
            str(gate_dt_ckpt),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--epochs",
            str(args.gate_epochs),
            "--episodes-per-epoch",
            str(args.gate_episodes_per_epoch),
            "--eval-every",
            str(args.gate_eval_every),
            "--eval-episodes",
            str(args.gate_eval_episodes),
            "--threshold",
            str(args.gate_threshold),
            "--alpha",
            str(args.gate_alpha),
            "--temperature",
            str(args.gate_temperature),
            "--target-return",
            str(args.gate_target_return),
            "--target-return-scale",
            str(args.gate_target_return_scale),
            "--lr-actor",
            str(args.gate_lr_actor),
            "--lr-critic",
            str(args.gate_lr_critic),
            "--ppo-clip",
            str(args.gate_ppo_clip),
            "--update-epochs",
            str(args.gate_update_epochs),
            "--mini-batch-size",
            str(args.gate_mini_batch_size),
        ]
        if bool(args.gate_soft_gate):
            cmd.append("--soft-gate")
        if args.gate_max_steps is not None:
            cmd.extend(["--max-steps", str(args.gate_max_steps)])
        if args.gate_eval_max_steps is not None:
            cmd.extend(["--eval-max-steps", str(args.gate_eval_max_steps)])
        cmd.append("--use-lstm-summary" if bool(args.use_lstm_summary) else "--no-use-lstm-summary")
        if bool(args.use_lstm_summary):
            cmd.extend(["--lstm-predictor-ckpt", str(lstm_ckpt)])
        run_cmd(cmd)

    print("[pipeline] done")


if __name__ == "__main__":
    main()
