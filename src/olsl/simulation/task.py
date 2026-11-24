"""Data-driven trial schedules for latent-state learning simulations.

Each TSV row is one trial. The JSON sidecar documents the event columns and
contains a project-specific Simulation mapping from columns to model arrays.
The mapping convention is reward_columns[reward][arm] and
cue_columns[reward][cue][arm].

Task produces the LatentState input contract:
    reward: (REWARDS, ARMS)
    c_vec:  (REWARDS, CUES, ARMS)

Task owns schedule metadata; agents own their output histories.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


class Task:
    """A deterministic, one-row-per-trial event schedule."""

    def __init__(self, events_file, sidecar_file=None):
        self.events_file = Path(events_file)
        self.sidecar_file = (
            Path(sidecar_file) if sidecar_file is not None
            else self.events_file.with_suffix(".json")
        )
        if not self.events_file.is_file():
            raise FileNotFoundError(self.events_file)
        if not self.sidecar_file.is_file():
            raise FileNotFoundError(self.sidecar_file)

        with self.events_file.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            self.events = list(reader)
            self.event_columns = set(reader.fieldnames or [])
        with self.sidecar_file.open(encoding="utf-8") as stream:
            self.sidecar = json.load(stream)
        self.mapping = self.sidecar.get("Simulation")
        if not isinstance(self.mapping, dict):
            raise ValueError("Sidecar must contain a 'Simulation' mapping.")
        self._validate()

    def __len__(self):
        return len(self.events)

    @property
    def dimensions(self):
        rewards = self.mapping["reward_columns"]
        cues = self.mapping["cue_columns"]
        return len(rewards), len(cues[0]), len(rewards[0])

    def _validate(self):
        if not self.events:
            raise ValueError("The event schedule contains no trials.")
        for col in ("onset", "duration"):
            if col not in self.event_columns:
                raise ValueError(f"Missing required event column {col!r}.")
            if any(row.get(col, "") == "" for row in self.events):
                raise ValueError(f"{col!r} cannot contain missing values.")
        if any(float(row["duration"]) < 0 for row in self.events):
            raise ValueError("'duration' must be non-negative.")
        onsets = [float(row["onset"]) for row in self.events]
        if any(b < a for a, b in zip(onsets, onsets[1:])):
            raise ValueError("'onset' must be non-decreasing.")

        rewards = self.mapping.get("reward_columns")
        cues = self.mapping.get("cue_columns")
        if not isinstance(rewards, list) or not rewards:
            raise ValueError("Simulation.reward_columns must be a non-empty list.")
        if not isinstance(cues, list) or len(cues) != len(rewards):
            raise ValueError(
                "Simulation.cue_columns must have one entry per reward dimension."
            )
        n_arms = len(rewards[0])
        if n_arms == 0 or any(len(row) != n_arms for row in rewards):
            raise ValueError("Every reward dimension must map the same arms.")
        for d, cue_dims in enumerate(cues):
            if not cue_dims:
                raise ValueError(f"Reward dimension {d} has no cue dimensions.")
            if any(len(arm_cols) != n_arms or not arm_cols
                   for arm_cols in cue_dims):
                raise ValueError("Every cue dimension must map every arm.")

        mapped = {c for row in rewards for c in row}
        mapped.update(
            c for reward_cues in cues for arm_cols in reward_cues for c in arm_cols
        )
        missing = mapped.difference(self.event_columns)
        if missing:
            raise ValueError(f"Mapped columns missing from TSV: {sorted(missing)}")

        for key in ("trial_type_column", "context_column",
                    "context_reset_column"):
            col = self.mapping.get(key)
            if col is not None and col not in self.event_columns:
                raise ValueError(f"{key} refers to missing column {col!r}.")

    @staticmethod
    def _numeric(row, column):
        try:
            value = float(row[column])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Column {column!r} must be numeric; got {row[column]!r}."
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"Column {column!r} contains a non-finite value.")
        return value

    def _make_world(self, row, trial, previous_onset):
        reward_cols = self.mapping["reward_columns"]
        cue_cols = self.mapping["cue_columns"]
        n_rewards, n_cues, n_arms = self.dimensions
        reward = np.empty((n_rewards, n_arms), dtype=float)
        c_vec = np.empty((n_rewards, n_cues, n_arms), dtype=float)

        for d in range(n_rewards):
            for a, col in enumerate(reward_cols[d]):
                reward[d, a] = self._numeric(row, col)
            for c in range(n_cues):
                for a, col in enumerate(cue_cols[d][c]):
                    c_vec[d, c, a] = self._numeric(row, col)

        onset = float(row["onset"])
        world = {
            "reward": reward,
            "c_vec": c_vec,
            "trial": trial,                  # 1-based
            "ntrials": len(self.events),
            "onset": onset,
            "duration": float(row["duration"]),
            "elapsed": None if previous_onset is None else onset - previous_onset,
            "context_reset": False,
        }
        for key, world_key in (
            ("trial_type_column", "trial_type"),
            ("context_column", "context"),
        ):
            col = self.mapping.get(key)
            if col:
                world[world_key] = row[col]
        reset_col = self.mapping.get("context_reset_column")
        if reset_col:
            value = row[reset_col]
            if isinstance(value, str):
                value = value.strip().lower() in {"1", "true", "yes", "y"}
            world["context_reset"] = bool(value)
        return world

    def run(self, agents: Iterable):
        """Update each agent on each trial; return the same agents.

        Task deliberately does not log model outputs. Each agent maintains
        its own complete parameter/history records.
        """
        agents = list(agents)
        if not agents:
            raise ValueError("Pass at least one agent to Task.run().")
        expected = self.dimensions
        for i, agent in enumerate(agents):
            p = getattr(agent, "par", None)
            if p is None:
                raise TypeError(f"Agent {i} has no 'par' parameter object.")
            actual = (p.REWARDS, p.CUES, p.ARMS)
            if actual != expected:
                raise ValueError(
                    f"Agent {i} dimensions {actual} do not match task "
                    f"dimensions {expected} (REWARDS, CUES, ARMS)."
                )

        previous_onset = None
        for trial, row in enumerate(self.events, start=1):
            world = self._make_world(row, trial, previous_onset)
            for agent in agents:
                # Avoid cross-agent mutation of a shared observation.
                obs = {
                    k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in world.items()
                }
                agent.update(obs)
            previous_onset = float(row["onset"])
        return agents
