"""
Expert oracle abstraction for behavior-cloning / DAgger training.

This module defines an `ExpertOracle` interface and a concrete `MIPExpert`
implementation that delegates to optim.py. It is the layer that BC and
DAgger code interacts with — by keeping the oracle interface abstract, the
training pipeline doesn't care whether the expert is a single-product MIP,
a multi-product MIP, an LP relaxation, or a hand-coded heuristic.

Multi-product extension path:
  - Implement a new MultiProductMIPExpert that wraps a multi-product MIP
  - Plug it into generate_expert_data.py / dagger_train.py — these scripts
    don't import optim.py directly, only ExpertOracle methods.

The expert produces (observation, action) pairs in the same format that
pretrain_bc.py and MaskablePPO expect:
  obs:    np.float32 array shape (18,) — normalized FabEnv observation
  action: np.int64 array shape (4,)    — MultiDiscrete([2,2,2,2]) integers
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .sim import ProductionLine
from .optim import solve_optimal_schedule, starts_to_actions


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------

def state_from_line(line: ProductionLine) -> Dict[str, Any]:
    """Extract a partial-state dict from a ProductionLine, suitable for re-injection
    or for an MIP query.

    The returned dict matches the shape expected by `solve_optimal_schedule`'s
    `initial_state` argument:
        q0, q1, q2, q3, qfin: int counts
        in_flight: list of (machine_id, completion_tick) for currently busy machines
        backlog: list of arrival_t for unmet demand items (in arrival order)
    """
    machines = [line.machine0, line.machine1, line.machine2, line.machine3]
    return {
        "q0": len(line.queues["queue0"]),
        "q1": len(line.queues["queue1"]),
        "q2": len(line.queues["queue2"]),
        "q3": len(line.queues["queue3"]),
        "qfin": len(line.queues["queue_fin"]),
        "in_flight": [(m, mach.finish_time) for m, mach in enumerate(machines) if mach.busy],
        "backlog": list(line.queues["demand"]),
    }


# ---------------------------------------------------------------------------
# Oracle interface
# ---------------------------------------------------------------------------

@dataclass
class ExpertQuery:
    """One (state, action) pair labeled by the expert.

    obs:    18-dim float32 observation (matches FabEnv).
    action: 4-dim int64 action array {0,1}^4 (matches MultiDiscrete([2,2,2,2])).
    mask:   8-dim bool mask flattened from (4,2) (matches FabEnv.action_masks()).
    t:      tick at which the query was made (for diagnostics).
    """
    obs: np.ndarray
    action: np.ndarray
    mask: np.ndarray
    t: int


class ExpertOracle(ABC):
    """Abstract expert oracle. Anything that can answer 'what's the optimal
    action at this state?' qualifies. Multi-product MIPs, hand-coded heuristics,
    etc. should implement this interface.
    """

    @abstractmethod
    def query_action(self, state: Dict[str, Any], t: int) -> np.ndarray:
        """Return the optimal action (np.int64 shape (4,)) for this state at tick t."""

    @abstractmethod
    def generate_full_trajectory(self) -> List[ExpertQuery]:
        """Return the (obs, action, mask) sequence for the whole horizon under
        the unperturbed optimal schedule.
        """

    @abstractmethod
    def generate_perturbed_trajectory(
        self,
        perturb_prob: float,
        rng: Optional[np.random.Generator] = None,
    ) -> List[ExpertQuery]:
        """Run a single perturbed trajectory: take optimal actions most of the
        time, but with probability `perturb_prob` deliberately take a different
        valid action. After each deviation, the expert is re-queried for the
        new optimal sequence. The returned list contains (state-the-policy-saw,
        expert's-recommendation) pairs — i.e., the DAgger-style training
        signal.
        """


# ---------------------------------------------------------------------------
# Concrete MIP expert
# ---------------------------------------------------------------------------

class MIPExpert(ExpertOracle):
    """ExpertOracle backed by the CP-SAT model in optim.py.

    The most expensive operation is `query_action` — each call solves a MIP
    from the given state to (t_start + lookahead). Default lookahead is the
    full remaining horizon, which is correct but slow. Reduce `lookahead` to
    speed up at the cost of myopia near the boundary.

    Parameters
    ----------
    common_cfg
        SimpleFab config (output of make_common_config).
    time_limit_sec
        Wall-time bound per MIP solve. 30s is usually enough with the
        Commander hint.
    lookahead_ticks
        How far ahead the MIP looks past `t_start`. None = full horizon.
        For DAgger queries, smaller (e.g., 1344 = 2 weeks) is dramatically
        faster but may miss far-future demand spikes.
    num_workers
        CP-SAT parallel workers per solve.
    verbose
        Pass through to the solver.
    """

    def __init__(
        self,
        common_cfg: Dict[str, Any],
        time_limit_sec: float = 30.0,
        lookahead_ticks: Optional[int] = None,
        num_workers: int = 8,
        verbose: bool = False,
    ):
        self.common_cfg = common_cfg
        self.time_limit_sec = float(time_limit_sec)
        self.lookahead_ticks = lookahead_ticks
        self.num_workers = int(num_workers)
        self.verbose = bool(verbose)
        self._H = int(common_cfg["time_horizon"])

        # Cache of (full-horizon) optimal solution for the unperturbed case.
        # Populated lazily on first `generate_full_trajectory` call.
        self._cached_full_solution: Optional[Dict[int, List[int]]] = None

    # ---- core oracle methods ---------------------------------------------

    def _solve_from(
        self,
        t_start: int,
        state: Dict[str, Any],
        time_limit_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run the MIP from the given state. Wraps optim.solve_optimal_schedule."""
        return solve_optimal_schedule(
            self.common_cfg,
            t_start=t_start,
            initial_state=state,
            lookahead_ticks=self.lookahead_ticks,
            time_limit_sec=time_limit_sec if time_limit_sec is not None else self.time_limit_sec,
            num_workers=self.num_workers,
            verbose=self.verbose,
            use_commander_hint=True,
        )

    def query_action(self, state: Dict[str, Any], t: int) -> np.ndarray:
        """Solve from state at tick t, return the action at tick t as np.int64 shape (4,)."""
        result = self._solve_from(t, state)
        if result["profit"] is None:
            raise RuntimeError(f"MIP failed to find a feasible solution at t={t}: {result['status']}")
        starts = result["starts"]
        action = np.array(
            [1 if t in set(starts[m]) else 0 for m in range(4)],
            dtype=np.int64,
        )
        return action

    # ---- trajectory generation -------------------------------------------

    def generate_full_trajectory(self) -> List[ExpertQuery]:
        """Solve full horizon once, then replay through sim recording (obs, action) per tick."""
        if self._cached_full_solution is None:
            result = solve_optimal_schedule(
                self.common_cfg,
                time_limit_sec=self.time_limit_sec,
                num_workers=self.num_workers,
                verbose=self.verbose,
                use_commander_hint=True,
            )
            if result["profit"] is None:
                raise RuntimeError(f"Full-horizon MIP failed: {result['status']}")
            self._cached_full_solution = result["starts"]

        return self._replay_to_queries(self._cached_full_solution, perturb_prob=0.0)

    def generate_perturbed_trajectory(
        self,
        perturb_prob: float,
        rng: Optional[np.random.Generator] = None,
    ) -> List[ExpertQuery]:
        """Roll out the env taking optimal actions with occasional random
        deviations. After each deviation, re-solve the MIP from the new state.
        Each tick contributes one (state-visited, expert-recommended-action) pair.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Seed: full-horizon optimal as the "current plan"
        if self._cached_full_solution is None:
            self.generate_full_trajectory()  # populates cache
        current_starts = {m: list(self._cached_full_solution[m]) for m in range(4)}

        line = ProductionLine(common_cfg=self.common_cfg)
        line.logging_enabled = False

        queries: List[ExpertQuery] = []

        for t in range(self._H):
            obs_list = line.get_observation(current_time=t, norm=True)
            obs = np.asarray(obs_list, dtype=np.float32)
            mask_2d = line.compute_action_mask(current_time=t)  # (4, 2)
            mask_flat = mask_2d.astype(bool).reshape(-1)

            # Optimal action under the current MIP plan: 1 if tick t is in starts[m].
            optimal = np.array(
                [1 if t in set(current_starts[m]) else 0 for m in range(4)],
                dtype=np.int64,
            )

            # Record the training pair BEFORE applying any action — the label is
            # always the expert's recommendation for the state we observed.
            queries.append(ExpertQuery(obs=obs, action=optimal, mask=mask_flat, t=t))

            # Decide what to actually apply
            applied = optimal.copy()
            deviated = False
            if rng.random() < perturb_prob:
                # Pick a random valid action that differs from optimal, if one exists.
                applied = self._sample_random_valid_action(mask_2d, optimal, rng)
                deviated = bool(np.any(applied != optimal))

            # Step the sim
            actions_dict = {f"machine{m}": bool(applied[m]) for m in range(4)}
            line.run_step(current_time=t, actions_override=actions_dict, invalid_penalty=0.0)

            # If we deviated, re-solve from the new state for tick t+1 onward
            if deviated and t + 1 < self._H:
                new_state = state_from_line(line)
                result = self._solve_from(t + 1, new_state)
                if result["profit"] is None:
                    # Solver failed — keep current plan as best-effort and continue
                    if self.verbose:
                        print(f"[expert] MIP failed at t={t+1}, keeping prior plan")
                else:
                    new_starts = result["starts"]
                    # Replace post-t portion of current plan with the new solution
                    for m in range(4):
                        current_starts[m] = [s for s in current_starts[m] if s <= t] + list(new_starts[m])

        return queries

    # ---- internals -------------------------------------------------------

    def _replay_to_queries(
        self,
        starts: Dict[int, List[int]],
        perturb_prob: float,
    ) -> List[ExpertQuery]:
        """Replay a fixed schedule through sim, recording (obs, action, mask) per tick.

        Used by generate_full_trajectory; perturb_prob is a placeholder for
        future strategies but must currently be 0.0 here.
        """
        if perturb_prob != 0.0:
            raise NotImplementedError("Use generate_perturbed_trajectory for nonzero perturb_prob.")

        line = ProductionLine(common_cfg=self.common_cfg)
        line.logging_enabled = False
        actions_per_tick = starts_to_actions(starts, self._H)

        queries: List[ExpertQuery] = []
        for t in range(self._H):
            obs_list = line.get_observation(current_time=t, norm=True)
            obs = np.asarray(obs_list, dtype=np.float32)
            mask_2d = line.compute_action_mask(current_time=t)
            mask_flat = mask_2d.astype(bool).reshape(-1)
            act = np.array(
                [1 if actions_per_tick[t][f"machine{m}"] else 0 for m in range(4)],
                dtype=np.int64,
            )
            queries.append(ExpertQuery(obs=obs, action=act, mask=mask_flat, t=t))
            line.run_step(current_time=t, actions_override=actions_per_tick[t], invalid_penalty=0.0)
        return queries

    @staticmethod
    def _sample_random_valid_action(
        mask_2d: np.ndarray,
        optimal: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample a valid 4-tuple action that DIFFERS from `optimal` on at least
        one axis whenever any axis has a non-optimal valid choice.

        Logic: each flippable machine independently flips with prob 0.5. If no
        machine flipped but at least one was flippable, force one to flip. This
        guarantees that a perturbation trigger produces a real deviation
        whenever the env permits one.
        """
        out = optimal.copy()
        flippable: List[int] = []
        for m in range(4):
            opt_m = int(optimal[m])
            alt = 1 - opt_m
            if mask_2d[m, alt]:
                flippable.append(m)
                if rng.random() < 0.5:
                    out[m] = alt
        # Guarantee at least one flip when any axis is flippable.
        if flippable and np.array_equal(out, optimal):
            m = int(rng.choice(flippable))
            out[m] = 1 - int(optimal[m])
        return out
