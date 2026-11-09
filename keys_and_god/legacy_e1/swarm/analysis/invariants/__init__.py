"""Game-invariant checker for finished (or partial) run dirs. Read-only; no LLM, no network.

    python -m swarm.analysis.invariants runs/<run_id> [--examples 5] [--no-prompts] [--game g00-god]

It re-derives the game from the event stream alone (an independent replay; it does not call the
engine or observe) and asserts, per game:
  order      seq contiguous from 0; round non-decreasing; agent-phase turn_index non-decreasing
             (God's phase may be stamped one slot early, the HookGod convention); every turn sits at
             its slot turn_index(r, pos, N) of the round's (re)shuffled order; each agent at most once
             per round; every active agent (not delayed/skipped, not a sat-out winner) plays
  keys       key conservation: copies move only by give/steal, and appear only by a defect_mint paired
             with a correct burn by the same actor in the same slot (defect_gain=mint); every give
             moves a copy the giver held; ok/failed give_key matches the replayed holdings; the
             replayed holdings equal every snapshot
  defects    burn.correct == (info in target's open needs); lose_need takes the actor's first open
             need; no burn of an agent that had already won
  scores     round_end scores, win events, won_round and the outcome match the replay
  delays     no turn while delayed_until_round >= round (wrong-defect delay, board post skip)
  obs        pinned facts equal the replayed state at observation time; inbox/board are exactly the
             messages stamped prev <= s < t (board posts at s + visible_delay_rounds * N; none at all
             when game.board.enabled=false); view_log lines are exactly the gives/board posts of the
             window and the defects released in (prev, t] (victim excluded under victim_then_delayed)
  left       (game.winner_policy=sit_out) a winner leaves when it wins: no later turn (simultaneous:
             none after its win round), no later successful action, no key transfer to/from it, no
             DM to/from it (God's forged ones included); its score stays at its value when it won; every other agent gets
             exactly one "left" notice for it; pinned peers never list a departed agent
  network    (game.network) the graph on game_start is symmetric and fixed (every snapshot); mode
             contacts: every DM (God's forged ones too) and give is along an edge, and every ok
             send/give/defect targets a contact; known_ids: start known = neighbours, the known set
             replayed from each observation (DM / board senders and names in their text, names in
             gift/burn/defect notices) equals every snapshot, and ok actions target known names;
             not_connected / unknown_contact refusals only on agents the actor could not address.
             The obs check then expects pinned peers = contacts / known names (minus departed), the
             board of game.board.scope=neighbors, and only local log lines (game/network.py)
  god        God events only in the god branch; ok powers == scheduled entries, <= K, budget_left
             bookkeeping; forged messages == ok frauds; memory_overwrites == ok cot_controls
  prompts    (llm_calls) no truth field / intervention id / banned word in the system prompt;
             "Your keys" == pinned holdings; "(key K)" in notices only for keys the agent holds or
             held; every "Your note (round r): ..." is the agent's own CURRENT entry; a replaced
             original note never re-appears as that round's entry
  replay     (--base) runner.fork.control_god cells: each control equals the base run's nogod game
             until the base game's first uncacheable call (empty response / error), then may re-sample
  fork       god vs nogod of one game_idx: identical events and LLM calls (requests AND responses)
             before God's first event (common random numbers); the noise branch has no God events
Exit code 1 when any violation is found.
"""

from __future__ import annotations

from .cli import main  # noqa: E402,F401
from .common import (  # noqa: F401  (the old module's names, re-exported)
    AGENT_TEXT_TAGS,
    FOUR_DIGITS,
    KEYS_RE,
    NOTE_RE,
    NOTICE_KEY_RE,
    Replay,
    Violation,
    _jsonl,
    load_cfg,
)
from .game import GameChecker  # noqa: F401
from .prompts import PromptChecks  # noqa: F401
from .runs import (  # noqa: F401
    _norm_cross,
    _norm_event,
    _uncacheable_round,
    check_run,
    control_replay_checks,
    fork_checks,
)
