import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def iter_step_records(replay_data: dict) -> Iterable[dict]:
    steps = replay_data.get("steps", [])
    if not isinstance(steps, list):
        return

    for step in steps:
        if isinstance(step, list):
            for record in step:
                if isinstance(record, dict):
                    yield record


def find_current_state(record: dict) -> Optional[dict]:
    current_state = record.get("current")
    if isinstance(current_state, dict):
        return current_state

    obs = record.get("obs")
    if isinstance(obs, dict):
        current_state = obs.get("current")
        if isinstance(current_state, dict):
            return current_state

    observation = record.get("observation")
    if isinstance(observation, dict):
        current_state = observation.get("current")
        if isinstance(current_state, dict):
            return current_state

    return None


def normalize_deck(deck: Sequence) -> Optional[List[int]]:
    normalized: List[int] = []
    for card in deck:
        if isinstance(card, dict):
            card_id = card.get("id")
        else:
            card_id = card

        if card_id is None:
            return None

        try:
            normalized.append(int(card_id))
        except (TypeError, ValueError):
            return None

    return normalized if len(normalized) == 60 else None


def extract_initial_decks(replay_data: dict) -> List[Tuple[int, List[int]]]:
    def collect_from_tree(node, found):
        if isinstance(node, dict):
            players = node.get("players")
            if isinstance(players, list):
                for player_index, player_state in enumerate(players[:2]):
                    if player_index in found:
                        continue
                    if not isinstance(player_state, dict):
                        continue
                    deck = player_state.get("deck")
                    if isinstance(deck, list):
                        normalized_deck = normalize_deck(deck)
                        if normalized_deck is not None:
                            found[player_index] = normalized_deck

            for value in node.values():
                collect_from_tree(value, found)
        elif isinstance(node, list):
            for item in node:
                collect_from_tree(item, found)

    for record in iter_step_records(replay_data):
        current_state = find_current_state(record)
        if not current_state:
            continue

        players = current_state.get("players")
        if not isinstance(players, list) or len(players) < 2:
            continue

        extracted: List[Tuple[int, List[int]]] = []
        for player_index in range(2):
            player_state = players[player_index]
            if not isinstance(player_state, dict):
                continue

            deck = player_state.get("deck")
            if isinstance(deck, list):
                normalized_deck = normalize_deck(deck)
                if normalized_deck is not None:
                    extracted.append((player_index, normalized_deck))

        if extracted:
            return extracted

    found_decks = {}
    collect_from_tree(replay_data, found_decks)
    if found_decks:
        return [(player_index, found_decks[player_index]) for player_index in sorted(found_decks.keys())[:2]]

    return []


def deck_signature(deck: Sequence[int]) -> str:
    payload = ",".join(map(str, deck)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def write_deck_csv(path: Path, deck: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        for card_id in deck:
            writer.writerow([card_id])


def log(message: str) -> None:
    print(f"[extract_replay_decks] {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 60-card decks from Kaggle replay JSON files.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/kaggle/input/datasets/organizations/kaggle/pokemon-tcg-ai-battle-episodes-2026-07-24",
        help="Folder berisi file replay JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "replay_decks"),
        help="Folder tujuan file deck CSV.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Simpan semua deck meski isinya identik.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input dir tidak ditemukan: {input_dir}")

    replay_files = sorted(input_dir.glob("*.json"))
    if not replay_files:
        raise SystemExit(f"Tidak ada file JSON di: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Mulai ekstraksi dari {input_dir}")
    log(f"Output deck CSV ke {output_dir}")
    log(f"Ditemukan {len(replay_files)} replay JSON")

    manifest = []
    seen_signatures = set()
    extracted_count = 0
    skipped_count = 0
    processed_count = 0

    for replay_path in replay_files:
        processed_count += 1
        log(f"[{processed_count}/{len(replay_files)}] Memproses {replay_path.name}")
        try:
            replay_data = load_json(replay_path)
        except Exception as exc:
            skipped_count += 1
            log(f"[SKIP] {replay_path.name}: gagal membaca JSON ({exc})")
            continue

        episode_id = None
        if isinstance(replay_data, dict):
            info = replay_data.get("info", {})
            if isinstance(info, dict):
                episode_id = info.get("EpisodeId")

        extracted_decks = extract_initial_decks(replay_data if isinstance(replay_data, dict) else {})
        if not extracted_decks:
            skipped_count += 1
            log(f"[SKIP] {replay_path.name}: tidak menemukan deck 60 kartu")
            continue

        for player_index, deck in extracted_decks:
            signature = deck_signature(deck)
            if args.no_dedupe:
                file_stem = f"episode_{episode_id or replay_path.stem}_p{player_index}_{signature}"
            else:
                if signature in seen_signatures:
                    log(f"[DUPLICATE] {replay_path.name} p{player_index}: deck sama, dilewati")
                    continue
                seen_signatures.add(signature)
                file_stem = f"deck_{signature}"

            output_path = output_dir / f"{file_stem}.csv"
            write_deck_csv(output_path, deck)
            log(f"[OK] {replay_path.name} p{player_index}: deck 60 kartu disimpan ke {output_path.name}")
            manifest.append(
                {
                    "source_file": replay_path.name,
                    "episode_id": episode_id,
                    "player_index": player_index,
                    "deck_file": output_path.name,
                    "signature": signature,
                    "card_count": len(deck),
                }
            )
            extracted_count += 1

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file_handle:
        json.dump(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "extracted_count": extracted_count,
                "skipped_count": skipped_count,
                "entries": manifest,
            },
            file_handle,
            indent=2,
            ensure_ascii=False,
        )

    log(f"Selesai. Extracted {extracted_count} deck file(s), skipped {skipped_count} file(s)")
    log(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()