"""
DeckGenome — Representasi deck yang siap dievolusi.
"""
import random
import copy
from typing import Optional

from .card_db import CardDB, CardRow, CardType
from . import config


def _name_counts(cards: list[CardRow]) -> dict:
    counts = {}
    for c in cards:
        counts[c.name] = counts.get(c.name, 0) + 1
    return counts


class DeckGenome:
    """
    Satu individu dalam populasi GA.

    Attributes:
        card_ids: list[int] of length 60
    """

    def __init__(self, card_ids: list[int] = None, db: CardDB = None):
        self.db = db or _get_default_db()
        self._fitness: Optional[float] = None
        self._extra_stats: dict = {}  # win rate, steps, dll

        if card_ids is not None:
            self.card_ids = card_ids[:]
        else:
            self.card_ids = self._random_deck()

    # ─── Load from file ───
    @classmethod
    def from_csv(cls, filepath: str, db: CardDB = None) -> 'DeckGenome':
        """Load deck dari CSV file (satu card ID per baris)."""
        card_ids = cls._load_card_ids(filepath)
        return cls(card_ids, db)

    @staticmethod
    def _load_card_ids(filepath: str) -> list[int]:
        """Read CSV file, return list of card IDs."""
        card_ids = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    card_ids.append(int(line))
        return card_ids

    @classmethod
    def from_csv_dir(cls, dirpath: str, db: CardDB = None, max_count: int = None, repair: bool = True) -> list['DeckGenome']:
        """Load semua deck dari folder CSV files. Repair jika ada yang invalid."""
        import glob, os
        files = sorted(glob.glob(os.path.join(dirpath, "*.csv")))
        if max_count:
            files = files[:max_count]
        decks = []
        for f in files:
            try:
                d = cls.from_csv(f, db)
                if repair and not d.is_valid():
                    d.repair()
                if d.is_valid():
                    decks.append(d)
            except Exception:
                pass
        return decks

    # ─── Random Generation ───
    def _random_deck(self) -> list[int]:
        """Generate deck random 60 kartu yang valid."""
        deck = []
        name_counts = {}
        has_ace_spec = [False]

        db = self.db

        def _can_add(card: CardRow) -> bool:
            if card.is_ace_spec:
                if has_ace_spec[0]:
                    return False
            max_count = 99 if card.is_energy else config.MAX_SAME_NAME
            return name_counts.get(card.name, 0) < max_count

        def _add(card: CardRow):
            deck.append(card.card_id)
            name_counts[card.name] = name_counts.get(card.name, 0) + 1
            if card.is_ace_spec:
                has_ace_spec[0] = True

        # 1. Pilih 2-4 evolution lines
        basics = db.get_basic_pokemon()
        num_lines = random.randint(2, min(4, len(basics)))
        random.shuffle(basics)

        lines_chosen = 0
        for basic in basics:
            if lines_chosen >= num_lines:
                break
            chain = db.get_evolution_chain(basic.name)
            if not chain["basic"]:
                continue

            lines_chosen += 1
            # Basic: 2-4 copies
            for _ in range(random.randint(2, 4)):
                if _can_add(basic):
                    _add(basic)

            # Stage 1: 2-3 copies jika ada
            if chain["stage1"]:
                s1 = random.choice(chain["stage1"])
                for _ in range(random.randint(2, 3)):
                    if _can_add(s1):
                        _add(s1)

            # Stage 2: 1-2 copies jika ada
            if chain["stage2"]:
                s2 = random.choice(chain["stage2"])
                for _ in range(random.randint(1, 2)):
                    if _can_add(s2):
                        _add(s2)

        # 2. Isi trainers (target ~35 kartu)
        trainers_all = db.get_trainers()
        random.shuffle(trainers_all)
        target_trainers = random.randint(28, 38)

        for trainer in trainers_all:
            if len(deck) >= target_trainers:
                break
            if _can_add(trainer):
                cnt = random.randint(1, min(
                    config.MAX_SAME_NAME - name_counts.get(trainer.name, 0),
                    target_trainers - len(deck)
                ))
                for _ in range(cnt):
                    if _can_add(trainer):
                        _add(trainer)

        # 3. Isi energy berdasarkan tipe Pokemon yang ada
        self._fill_energy(deck, name_counts, has_ace_spec)

        # 4. Pastikan tepat 60 kartu (genapkan dengan energy jika kurang)
        if len(deck) < config.DECK_SIZE:
            basic_energies = db.get_basic_energies()
            if basic_energies:
                energy_to_add = min(
                    config.DECK_SIZE - len(deck),
                    15  # Max tambahan energy
                )
                # Pilih energy type yang cocok
                main_type = self._detect_main_type([db.by_id(c) for c in deck if db.by_id(c)])
                matching = [e for e in basic_energies if e.energy_type == main_type] if main_type else basic_energies
                if not matching:
                    matching = basic_energies
                energy_card = matching[0] if isinstance(matching, list) else matching
                if isinstance(matching, list):
                    energy_card = random.choice(matching)
                for _ in range(energy_to_add):
                    if _can_add(energy_card):
                        _add(energy_card)

        # 5. Fix jika kelebihan (trim trainers paling belakang)
        while len(deck) > config.DECK_SIZE:
            # Cari trainer non-esensial
            removed = False
            for i in range(len(deck) - 1, -1, -1):
                card = db.by_id(deck[i])
                if card and card.is_trainer and not card.is_ace_spec:
                    deck.pop(i)
                    removed = True
                    break
            if not removed:
                deck.pop()  # Force pop

        # 6. Fix jika kurang dari 60 (tambah energy)
        while len(deck) < config.DECK_SIZE:
            basic_energies = db.get_basic_energies()
            if basic_energies:
                main_type = self._detect_main_type([db.by_id(c) for c in deck if db.by_id(c)])
                matching = [e for e in basic_energies if e.energy_type == main_type] if main_type else basic_energies
                if not matching:
                    matching = basic_energies
                e = random.choice(list(matching)) if isinstance(matching, list) else matching
                if isinstance(matching, list):
                    e = random.choice(matching)
                deck.append(e.card_id)

        return deck[:config.DECK_SIZE]

    def _detect_main_type(self, cards: list[CardRow]) -> str:
        """Detect tipe energy dominan dari Pokemon di deck."""
        type_counts = {}
        for c in cards:
            if c and c.is_pokemon:
                symbol = c.energy_type.strip("{}")
                if symbol:
                    type_counts[symbol] = type_counts.get(symbol, 0) + 1
        if type_counts:
            return max(type_counts, key=type_counts.get)
        return "G"

    def _fill_energy(self, deck: list, name_counts: dict, has_ace_spec: list):
        """Tambahkan energy cards ke deck, bertahap."""
        db = self.db
        cards_in_deck = [db.by_id(c) for c in deck if db.by_id(c)]
        main_type = self._detect_main_type(cards_in_deck)

        # Basic Energy
        basic_energies = db.get_basic_energies()
        matching_energies = [e for e in basic_energies if e.energy_type == "{" + main_type + "}"]
        if not matching_energies:
            matching_energies = basic_energies

        if matching_energies:
            energy_card = matching_energies[0] if len(matching_energies) == 1 else random.choice(matching_energies)
            target_energy = random.randint(10, 14)
            while len(deck) < target_energy:
                deck.append(energy_card.card_id)
                name_counts[energy_card.name] = name_counts.get(energy_card.name, 0) + 1

        # Special Energy (1-2)
        special = [c for c in db.get_energies() if c.stage == CardType.SPECIAL_ENERGY]
        if special:
            num_special = random.randint(0, min(2, config.DECK_SIZE - len(deck)))
            for _ in range(num_special):
                s = random.choice(special)
                if s.card_id not in deck:  # Max 1 special energy type
                    deck.append(s.card_id)
                    name_counts[s.name] = name_counts.get(s.name, 0) + 1

    # ─── Validation ───
    def validate(self) -> tuple[bool, list[str]]:
        """Cek apakah deck valid. Return (valid, alasan_kegagalan)."""
        errors = []
        if len(self.card_ids) != config.DECK_SIZE:
            errors.append(f"Panjang deck {len(self.card_ids)}, harus {config.DECK_SIZE}")

        cards = [self.db.by_id(c) for c in self.card_ids]
        cards = [c for c in cards if c]

        if not cards:
            errors.append("Deck kosong atau semua kartu tidak dikenal")
            return False, errors

        # Cek basic Pokemon
        has_basic = any(c.is_basic for c in cards)
        if not has_basic:
            errors.append("Tidak ada Basic Pokémon!")

        # Cek max 4 same name (non-energy)
        name_counts_dict = _name_counts(cards)
        for name, count in name_counts_dict.items():
            card = self.db.by_name(name)
            if card and not card[0].is_energy and count > config.MAX_SAME_NAME:
                errors.append(f"{name}: {count} copies (max {config.MAX_SAME_NAME})")

        # Cek ACE SPEC
        ace_count = sum(1 for c in cards if c.is_ace_spec)
        if ace_count > config.MAX_ACE_SPEC:
            errors.append(f"ACE SPEC: {ace_count} (max {config.MAX_ACE_SPEC})")

        return len(errors) == 0, errors

    def is_valid(self) -> bool:
        valid, _ = self.validate()
        return valid

    def repair(self) -> int:
        """
        Perbaiki deck invalid hasil crossover/mutasi.
        Mengembalikan jumlah fix yang dilakukan.

        Fixes:
        - ACE SPEC > 1 → simpan 1, ganti sisanya dengan trainer/item random
        - Nama kartu > 4 copies (non-energy) → kurangi ke 4
        - Ukuran deck tidak 60 → genapkan dengan energy
        - Tidak ada Basic Pokemon → tambah 1 basic random
        """
        fixes = 0
        db = self.db
        deck = self.card_ids[:]

        # 1. Fix ACE SPEC duplicates (max 1)
        ace_indices = [i for i, cid in enumerate(deck) if db.by_id(cid) and db.by_id(cid).is_ace_spec]
        while len(ace_indices) > 1:
            idx = ace_indices.pop()
            trainers = [t for t in db.get_trainers() if not t.is_ace_spec]
            if trainers:
                deck[idx] = random.choice(trainers).card_id
            else:
                energies = db.get_basic_energies()
                if energies:
                    deck[idx] = random.choice(energies).card_id
            fixes += 1

        # 2. Fix > 4 copies of non-energy cards
        cards_in_deck = [db.by_id(cid) for cid in deck if db.by_id(cid)]
        name_counts = _name_counts(cards_in_deck)
        for name, count in list(name_counts.items()):
            if count <= 4:
                continue
            sample = db.by_name(name)
            if not sample or sample[0].is_energy:
                continue
            excess = count - 4
            removed = 0
            for i in range(len(deck) - 1, -1, -1):
                if removed >= excess:
                    break
                c = db.by_id(deck[i])
                if c and c.name == name:
                    deck.pop(i)
                    removed += 1
                    fixes += 1

        # 3. Fix deck size > 60
        basic_energy_ids = {e.card_id for e in db.get_basic_energies()}
        while len(deck) > config.DECK_SIZE:
            # Remove basic energy first, then trainers
            found = False
            for i in range(len(deck) - 1, -1, -1):
                if deck[i] in basic_energy_ids:
                    deck.pop(i)
                    found = True
                    fixes += 1
                    break
            if not found:
                deck.pop()
                fixes += 1

        # 4. Fix no basic Pokemon
        cards_now = [db.by_id(cid) for cid in deck if db.by_id(cid)]
        if not any(c.is_basic for c in cards_now):
            basics = db.get_basic_pokemon()
            if basics:
                # If deck is full, replace an energy
                basic_energy_ids = {e.card_id for e in db.get_basic_energies()}
                for i in range(len(deck) - 1, -1, -1):
                    if deck[i] in basic_energy_ids:
                        deck.pop(i)
                        fixes += 1
                        break
                deck.append(random.choice(basics).card_id)
                fixes += 1

        # 5. Fix deck size < 60
        main_type = self._detect_main_type([db.by_id(c) for c in deck if db.by_id(c)])
        basic_energies = db.get_basic_energies()
        matching = [e for e in basic_energies if e.energy_type == "{" + main_type + "}"] or basic_energies
        while len(deck) < config.DECK_SIZE:
            deck.append(random.choice(matching).card_id)
            fixes += 1

        self.card_ids = deck[:config.DECK_SIZE]
        return fixes

    # ─── Evolution Line Extraction ───
    def extract_evolution_lines(self) -> list[list[int]]:
        """
        Extract evolution lines dari deck.

        Returns:
            list of list[int]: tiap line = [basic_id, stage1_id?, stage2_id?]
        """
        db = self.db
        cards = [db.by_id(cid) for cid in self.card_ids]
        cards = [c for c in cards if c]

        lines = []
        used = set()

        # Cari basic Pokemon
        for c in cards:
            if c.is_basic and c.card_id not in used:
                line = [c.card_id]
                used.add(c.card_id)

                # Cari Stage 1 yang evolve dari basic ini
                for c2 in cards:
                    if c2.is_stage1 and c2.prev_stage_name == c.name and c2.card_id not in used:
                        line.append(c2.card_id)
                        used.add(c2.card_id)

                        # Cari Stage 2
                        for c3 in cards:
                            if c3.is_stage2 and c3.prev_stage_name == c2.name and c3.card_id not in used:
                                line.append(c3.card_id)
                                used.add(c3.card_id)
                                break
                        break
                lines.append(line)

        return lines

    def get_non_evolution_card_ids(self) -> list[int]:
        """Kartu yang bukan bagian dari evolution line (trainers, energy)."""
        line_ids = set()
        for line in self.extract_evolution_lines():
            for cid in line:
                line_ids.add(cid)
        return [cid for cid in self.card_ids if cid not in line_ids]

    # ─── Crossover ───
    def crossover(self, other: 'DeckGenome') -> tuple['DeckGenome', 'DeckGenome']:
        """
        Crossover evolution line-aware.

        1. Ambil evolution line dari parent A
        2. Ambil evolution line dari parent B (yang belum ada)
        3. Isi sisa dengan trainers + energy dari parent A dan B

        Returns:
            (child1, child2) — 2 offspring
        """
        if random.random() > config.CROSSOVER_RATE:
            return (copy.deepcopy(self), copy.deepcopy(other))

        child = []
        child_ids = set()

        # Strategy: ambil evolution lines dari self, trainers+energy dari other
        my_lines = self.extract_evolution_lines()
        other_lines = other.extract_evolution_lines()

        # Pilih subset random dari lines self
        random.shuffle(my_lines)
        num_my_lines = random.randint(1, min(len(my_lines), 3))
        for line in my_lines[:num_my_lines]:
            for cid in line:
                child.append(cid)
                child_ids.add(cid)

        # Tambah lines dari other yang evolution chainnya beda
        for line in other_lines:
            if len(child) >= 50:
                break
            # Cek apakah basic name-nya sudah ada
            basic_name = self.db.by_id(line[0]).name if self.db.by_id(line[0]) else ""
            already_have = False
            for cid in child_ids:
                card = self.db.by_id(cid)
                if card and card.name == basic_name:
                    already_have = True
                    break
            if not already_have:
                for cid in line:
                    child.append(cid)
                    child_ids.add(cid)

        # Isi sisa dengan non-evolution cards dari self dan other
        my_non_evo = self.get_non_evolution_card_ids()
        other_non_evo = other.get_non_evolution_card_ids()

        filler_candidates = my_non_evo + other_non_evo
        random.shuffle(filler_candidates)

        for cid in filler_candidates:
            if len(child) >= config.DECK_SIZE:
                break
            if cid not in child_ids:
                child.append(cid)
                child_ids.add(cid)

        # Isi sampai 60 dengan energy
        child_instance = DeckGenome(child, self.db)
        if len(child) < config.DECK_SIZE:
            child_instance._add_energy_to_fill()

        child1 = DeckGenome(child_instance.card_ids[:config.DECK_SIZE], self.db)

        # Repair untuk fix ACE SPEC duplicate, >4 copies, atau masalah lainnya
        child1.repair()

        # Child 2: swap parent roles
        child2 = other.crossover(self)[0]
        child2.repair()

        return child1, child2

    def _add_energy_to_fill(self):
        """Add basic energy until 60 cards."""
        db = self.db
        main_type = self._detect_main_type([db.by_id(c) for c in self.card_ids if db.by_id(c)])
        basic_energies = db.get_basic_energies()
        matching = [e for e in basic_energies if e.energy_type == "{" + main_type + "}"]
        if not matching:
            matching = basic_energies
        if matching:
            e = random.choice(matching)
            while len(self.card_ids) < config.DECK_SIZE:
                self.card_ids.append(e.card_id)

    # ─── Mutation ───
    def mutate(self) -> bool:
        """
        Mutate deck in-place dengan probabilitas config.MUTATION_RATE.

        Returns:
            True jika termutasi, False jika tidak.
        """
        if random.random() > config.MUTATION_RATE:
            return False

        strategy = random.choice(config.MUTATION_STRATEGIES)
        db = self.db
        cards = [db.by_id(c) for c in self.card_ids]
        cards = [c for c in cards if c]

        if strategy == 'card_swap':
            # Ganti 1-3 kartu random dengan kartu random dari database
            non_energy = db.get_all_non_energy()
            non_basic_energy = [c for c in cards if not (c.is_energy and c.stage == CardType.BASIC_ENERGY)]
            if non_basic_energy and non_energy:
                num_swap = min(random.randint(1, 3), len(non_basic_energy))
                for _ in range(num_swap):
                    old = random.choice(non_basic_energy)
                    new = random.choice(non_energy)
                    if old.card_id in self.card_ids:
                        idx = self.card_ids.index(old.card_id)
                        self.card_ids[idx] = new.card_id

        elif strategy == 'energy_tune':
            # Adjust jumlah energy (+/- 2)
            basic_energies = db.get_basic_energies()
            basic_energy_ids = {e.card_id for e in basic_energies}
            energy_indices = [i for i, cid in enumerate(self.card_ids) if cid in basic_energy_ids]
            if energy_indices:
                to_remove = random.sample(energy_indices, min(random.randint(1, 2), len(energy_indices)))
                for idx in sorted(to_remove, reverse=True):
                    self.card_ids.pop(idx)

                # Cari energy type
                main_type = self._detect_main_type(cards)
                matching = [e for e in basic_energies if e.energy_type == "{" + main_type + "}"]
                if not matching:
                    matching = basic_energies
                if matching:
                    e = random.choice(matching)
                    while len(self.card_ids) < config.DECK_SIZE:
                        self.card_ids.append(e.card_id)

        elif strategy == 'trainer_tune':
            # Ganti trainer random
            trainers_in_deck = [c for c in cards if c.is_trainer]
            all_trainers = db.get_trainers()
            if trainers_in_deck and all_trainers:
                old = random.choice(trainers_in_deck)
                new = random.choice(all_trainers)
                if old.card_id in self.card_ids:
                    idx = self.card_ids.index(old.card_id)
                    self.card_ids[idx] = new.card_id

        # Repair deck — fix ACE SPEC duplicates, >4 copies, missing basics, etc.
        repaired = self.repair()

        self._fitness = None  # Reset fitness
        return True

    # ─── Fitness ───
    @property
    def fitness(self) -> Optional[float]:
        return self._fitness

    @fitness.setter
    def fitness(self, value: float):
        self._fitness = value

    @property
    def extra_stats(self) -> dict:
        return self._extra_stats

    @extra_stats.setter
    def extra_stats(self, stats: dict):
        self._extra_stats = stats

    # ─── Utilities ───
    def to_list(self) -> list[int]:
        return self.card_ids[:]

    def to_csv(self, filepath: str):
        """Simpan deck ke CSV (satu card ID per baris)."""
        with open(filepath, "w") as f:
            for cid in self.card_ids:
                f.write(f"{cid}\n")

    def summary(self) -> str:
        """Human-readable summary of the deck."""
        db = self.db
        cards = [db.by_id(c) for c in self.card_ids]
        cards = [c for c in cards if c]

        pokemon = [c for c in cards if c.is_pokemon]
        trainers = [c for c in cards if c.is_trainer]
        energies = [c for c in cards if c.is_energy]

        lines = self.extract_evolution_lines()
        line_str = "; ".join(
            " → ".join(db.by_id(cid).name if db.by_id(cid) else "?" for cid in line)
            for line in lines
        )

        if self._fitness is not None:
            fitness_str = f"{self._fitness:.3f}"
        else:
            fitness_str = "N/A"
        return (
            f"Deck ({len(self.card_ids)} cards) "
            f"Fitness={fitness_str} | "
            f"Pokemon={len(pokemon)} | Trainers={len(trainers)} | Energy={len(energies)} | "
            f"Lines: {line_str}"
        )

    def __repr__(self) -> str:
        return f"DeckGenome(fitness={self._fitness})"


# ─── Global DB Singleton ───
_DEFAULT_DB: Optional[CardDB] = None


def _get_default_db() -> CardDB:
    global _DEFAULT_DB
    if _DEFAULT_DB is None:
        import os
        _DEFAULT_DB = CardDB(config.CARD_DB_PATH)
    return _DEFAULT_DB
