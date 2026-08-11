#!/usr/bin/env python3
"""Generate all independent CP/LS/RS rhythm and pitch combinations.

The transition models and CP/LS/RS formulas are imported from
``pitch_rhythm_ohmura_cp_rs_ls_generator.py``.  Rhythm and pitch use
separate model names and separate random-number streams, so changing the
pitch model does not change the generated rhythm for a fixed seed.

Candidate policy:

- CP and LS are restricted to pitch/rhythm states observed in the source.
- RS is unrestricted: MIDI 48-83 for pitch and every note/rest rhythm state
  that fits the source meter on a sixteenth-note grid.
- Only candidates with a positive model score are selected.  There is no
  fallback that bypasses the candidate policy; dead ends are backtracked.

Default output files are named like this::

    <stem>_rhythm_cp_pitch_cp.musicxml
    <stem>_rhythm_cp_pitch_ls.musicxml
    ...
    <stem>_rhythm_rs_pitch_rs.musicxml

Example::

    python outputs/pitch_rhythm_9_combinations_generator.py input.musicxml \
        --output-dir generated_9 --seed 20260717

Only monophonic score-partwise MusicXML aligned to a sixteenth-note grid is
supported, matching the source generator.  The standard library is enough.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, TypeVar

from pitch_rhythm_ohmura_cp_rs_ls_generator import (
    PITCH_MAX,
    PITCH_MIN,
    ParsedScore,
    RhythmToken,
    SourceEvent,
    TransitionModel,
    adjacent_pairs,
    build_metric_rhythm_states,
    build_observed_rhythm_states,
    parse_modes,
    parse_score,
    write_musicxml,
)


DEFAULT_MODES = ("cp", "ls", "rs")
MODE_SEED_OFFSETS = {"cp": 101, "ls": 211, "rs": 307}
PITCH_SEED_OFFSET = 1_000_000
RESTRICTED_MODES = frozenset(("cp", "ls"))
EPSILON = 1e-12

T = TypeVar("T")


class GenerationError(RuntimeError):
    """Raised when no complete path satisfies the selected candidate policy."""


def parse_unique_modes(text: str) -> list[str]:
    """Parse a comma-separated model list while preserving its order."""
    modes = parse_modes(text)
    return list(dict.fromkeys(modes))


def combination_output_path(
    output_dir: Path,
    input_path: Path,
    rhythm_mode: str,
    pitch_mode: str,
) -> Path:
    return output_dir / (
        f"{input_path.stem}_rhythm_{rhythm_mode}_pitch_{pitch_mode}.musicxml"
    )


def stream_seed(base_seed: int, mode: str, *, pitch: bool) -> int:
    offset = MODE_SEED_OFFSETS[mode]
    if pitch:
        offset += PITCH_SEED_OFFSET
    return base_seed + offset


def pitch_states_for_mode(parsed: ParsedScore, mode: str) -> list[int]:
    if mode in RESTRICTED_MODES:
        return sorted(set(parsed.pitch_sequence))
    if mode == "rs":
        return list(range(PITCH_MIN, PITCH_MAX + 1))
    raise ValueError(f"Unknown pitch mode: {mode}")


def rhythm_states_for_mode(parsed: ParsedScore, mode: str) -> list[RhythmToken]:
    if mode in RESTRICTED_MODES:
        return build_observed_rhythm_states(parsed.rhythm_sequence)
    if mode == "rs":
        return build_metric_rhythm_states(parsed.measures)
    raise ValueError(f"Unknown rhythm mode: {mode}")


def weighted_order(items: list[T], weights: list[float], rng: random.Random) -> list[T]:
    """Return all candidates in roulette-sampled order, without replacement."""
    remaining_items = list(items)
    remaining_weights = list(weights)
    ordered: list[T] = []

    while remaining_items:
        total = sum(weight for weight in remaining_weights if weight > 0)
        if total <= 0:
            break

        threshold = rng.random() * total
        running = 0.0
        selected_index = len(remaining_items) - 1
        for index, weight in enumerate(remaining_weights):
            if weight <= 0:
                continue
            running += weight
            if running >= threshold:
                selected_index = index
                break

        ordered.append(remaining_items.pop(selected_index))
        remaining_weights.pop(selected_index)

    return ordered


def scored_candidates(
    model: TransitionModel,
    mode: str,
    previous: Optional[T],
    candidates: list[T],
    priors: Counter[T],
) -> tuple[list[T], list[float]]:
    """Filter candidates by the model without using an unrestricted fallback."""
    selected: list[T] = []
    weights: list[float] = []

    for candidate in candidates:
        if previous is None and mode == "rs":
            weight = 1.0
        elif previous is None:
            weight = float(priors[candidate])
        else:
            weight = model.score(previous, candidate, mode)

        if weight > EPSILON:
            selected.append(candidate)
            weights.append(weight)

    return selected, weights


def generate_rhythm_sequence(
    parsed: ParsedScore,
    model: TransitionModel,
    states: list[RhythmToken],
    mode: str,
    rng: random.Random,
) -> list[RhythmToken]:
    priors = Counter(parsed.rhythm_sequence)
    by_position: dict[tuple[int, int], list[RhythmToken]] = {}
    for state in states:
        by_position.setdefault((state.bar_units, state.phase), []).append(state)

    output: list[RhythmToken] = []
    failed: set[tuple[int, int, Optional[RhythmToken]]] = set()
    total_units = sum(measure.target_units for measure in parsed.measures)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), total_units * 4 + 200))

    def search(
        measure_index: int,
        phase: int,
        previous: Optional[RhythmToken],
    ) -> bool:
        if measure_index >= len(parsed.measures):
            return True

        measure = parsed.measures[measure_index]
        if phase == measure.target_units:
            return search(measure_index + 1, 0, previous)

        state_key = (measure_index, phase, previous)
        if state_key in failed:
            return False

        remaining = measure.target_units - phase
        raw_candidates = by_position.get((measure.bar_units, phase), [])
        fitting = [state for state in raw_candidates if state.duration <= remaining]
        candidates, weights = scored_candidates(
            model=model,
            mode=mode,
            previous=previous,
            candidates=fitting,
            priors=priors,
        )

        for candidate in weighted_order(candidates, weights, rng):
            output.append(candidate)
            if search(measure_index, phase + candidate.duration, candidate):
                return True
            output.pop()

        failed.add(state_key)
        return False

    if not search(0, 0, None):
        policy = "observed-state restricted" if mode in RESTRICTED_MODES else "unrestricted"
        raise GenerationError(
            f"Could not generate a complete {mode.upper()} rhythm path "
            f"under the {policy} candidate policy."
        )

    return output


def generate_pitch_sequence(
    parsed: ParsedScore,
    model: TransitionModel,
    states: list[int],
    mode: str,
    length: int,
    rng: random.Random,
) -> list[int]:
    if length == 0:
        return []

    priors = Counter(parsed.pitch_sequence)
    output: list[int] = []
    failed: set[tuple[int, Optional[int]]] = set()
    sys.setrecursionlimit(max(sys.getrecursionlimit(), length * 4 + 200))

    def search(index: int, previous: Optional[int]) -> bool:
        if index == length:
            return True

        state_key = (index, previous)
        if state_key in failed:
            return False

        candidates, weights = scored_candidates(
            model=model,
            mode=mode,
            previous=previous,
            candidates=states,
            priors=priors,
        )
        for candidate in weighted_order(candidates, weights, rng):
            output.append(candidate)
            if search(index + 1, candidate):
                return True
            output.pop()

        failed.add(state_key)
        return False

    if not search(0, None):
        policy = "observed-state restricted" if mode in RESTRICTED_MODES else "unrestricted"
        raise GenerationError(
            f"Could not generate {length} {mode.upper()} pitch events "
            f"under the {policy} candidate policy."
        )

    return output


def combine_streams(
    rhythm_sequence: list[RhythmToken],
    pitch_sequence: list[int],
) -> list[SourceEvent]:
    events: list[SourceEvent] = []
    pitch_index = 0

    for rhythm in rhythm_sequence:
        midi = None
        if rhythm.kind == "N":
            if pitch_index >= len(pitch_sequence):
                raise ValueError("Pitch sequence ended before the rhythm sequence.")
            midi = pitch_sequence[pitch_index]
            pitch_index += 1
        events.append(SourceEvent(rhythm, midi, False, False))

    return events


def run(args: argparse.Namespace) -> tuple[int, list[Path]]:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_path.parent / f"{input_path.stem}_pitch_rhythm_9"
    )

    parsed = parse_score(input_path, part_id=args.part_id)
    rhythm_modes = parse_unique_modes(args.rhythm_models)
    pitch_modes = parse_unique_modes(args.pitch_models)
    rhythm_universe = getattr(args, "rhythm_universe", "metric")
    if rhythm_universe != "metric":
        raise ValueError(
            "--rhythm-universe=observed conflicts with the candidate policy: "
            "CP/LS are always observed-state restricted and RS must be unrestricted."
        )

    pitch_transitions = adjacent_pairs(parsed.pitch_sequence)
    rhythm_transitions = adjacent_pairs(parsed.rhythm_sequence)
    pitch_states = {
        mode: pitch_states_for_mode(parsed, mode)
        for mode in pitch_modes
    }
    rhythm_states = {
        mode: rhythm_states_for_mode(parsed, mode)
        for mode in rhythm_modes
    }
    pitch_models = {
        mode: TransitionModel(pitch_states[mode], pitch_transitions)
        for mode in pitch_modes
    }
    rhythm_models = {
        mode: TransitionModel(rhythm_states[mode], rhythm_transitions)
        for mode in rhythm_modes
    }

    base_seed = (
        args.seed
        if args.seed is not None
        else random.SystemRandom().randrange(1 << 63)
    )

    generated_rhythms = {
        mode: generate_rhythm_sequence(
            parsed=parsed,
            model=rhythm_models[mode],
            states=rhythm_states[mode],
            mode=mode,
            rng=random.Random(stream_seed(base_seed, mode, pitch=False)),
        )
        for mode in rhythm_modes
    }
    note_counts = {
        mode: sum(state.kind == "N" for state in sequence)
        for mode, sequence in generated_rhythms.items()
    }
    maximum_pitch_count = max(note_counts.values(), default=0)
    generated_pitches = {
        mode: generate_pitch_sequence(
            parsed=parsed,
            model=pitch_models[mode],
            states=pitch_states[mode],
            mode=mode,
            length=maximum_pitch_count,
            rng=random.Random(stream_seed(base_seed, mode, pitch=True)),
        )
        for mode in pitch_modes
    }

    written: list[Path] = []
    for rhythm_mode in rhythm_modes:
        for pitch_mode in pitch_modes:
            generated = combine_streams(
                generated_rhythms[rhythm_mode],
                generated_pitches[pitch_mode][:note_counts[rhythm_mode]],
            )

            path = combination_output_path(
                output_dir,
                input_path,
                rhythm_mode,
                pitch_mode,
            )
            write_musicxml(
                path=path,
                parsed=parsed,
                generated_events=generated,
                movement_title=(
                    f"{input_path.stem} rhythm={rhythm_mode.upper()} "
                    f"pitch={pitch_mode.upper()}"
                ),
            )
            written.append(path)

    return base_seed, written


def build_arg_parser() -> argparse.ArgumentParser:
    default_modes = ",".join(DEFAULT_MODES)
    parser = argparse.ArgumentParser(
        description=(
            "Generate the Cartesian product of CP/LS/RS rhythm and pitch "
            "transformations as MusicXML."
        )
    )
    parser.add_argument(
        "input",
        help="Path to a monophonic .musicxml/.xml/.mxl file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help=(
            "Output directory. Defaults to <input_stem>_pitch_rhythm_9 "
            "beside the input file."
        ),
    )
    parser.add_argument(
        "--part-id",
        help="MusicXML part id to read. Defaults to the first part.",
    )
    parser.add_argument(
        "--rhythm-models",
        default=default_modes,
        help=f"Comma-separated rhythm models. Default: {default_modes}.",
    )
    parser.add_argument(
        "--pitch-models",
        default=default_modes,
        help=f"Comma-separated pitch models. Default: {default_modes}.",
    )
    parser.add_argument(
        "--rhythm-universe",
        choices=("metric",),
        default="metric",
        help=(
            "Compatibility option. RS always uses the unrestricted metric "
            "universe; CP/LS always use observed source states."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Base random seed. The chosen seed is printed for reproducibility.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        seed, written = run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"seed={seed}")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
