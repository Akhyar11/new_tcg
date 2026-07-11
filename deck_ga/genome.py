"""
DeckGenome — Representasi deck yang siap dievolusi.

Improvements v2:
- Template-based random generation (ratio realistis)
- Crossover preserves evolution lines + trainer core structure
- Mutation: evo_line_swap, energy type-aware
- Repair: energy type-aware (bukan asal energy)
- Diversity distance metric untuk fitness sharing
"""
import random
import copy
import math
from typing import Optional

from .card_db import CardDB, CardRow, CardType
from . import config


def _name_counts(cards: list[CardRow]) -> dict:
    counts = {}
    for c in cards:
        counts[c.name] = counts.get(c.name, 0) + 1
    return counts


def _card_id_set(card_ids: list[int]) -> set[int]:
    return set(card_ids)


def deck_jaccard_distance(a: list[int], b: list[int]) -> float:
    """
    Jarak antara dua deck berdasarkan Jaccard distance dari set kartu unik.
    0.0 = identik, 1.0 =完全不 sama.
    """
    set_a = set(a)
    set_b = set(b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 1.0
    return 1.0 - (intersection / union)


def deck_cosine_similarity(a: list[int], b: list[int]) -> float:
    """
    Cosine similarity berdasarkan count vector kartu.
    1.0 = identik, 0.0 =完全不 sama.
    """
    from collections import Counter
    ca = Counter(a)
    cb = Counter(b)
    all_ids = set(ca.keys()) | set(cb.keys())
    dot = sum(ca.get(cid, 0) * cb.get(cid, 0) for cid in all_ids)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


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

    # ─── Template-Based Random Generation ───
    def _random_deck(self) -> list[int]:
        """
        Generate deck random 60 kartu yang valid, menggunakan template ratios.
        Improvement: menggunakan rasio Pokemon:Trainer:Energy yang realistis.
        """
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

        # 1. Pilih evolution lines (2-3 lines, lebih konservatif)
        basics = db.get_basic_pokemon()
        num_lines = random.randint(*config.TARGET_EVO_LINES)
        random.shuffle(basics)

        target_pokemon = random.randint(*config.TARGET_POKEMON_RANGE)
        target_trainer = random.randint(*config.TARGET_TRAINER_RANGE)
        # Energy = sisa

        lines_chosen = 0
        for basic in basics:
            if lines_chosen >= num_lines:
                break
            chain = db.get_evolution_chain(basic.name)
            if not chain["basic"]:
                continue

            lines_chosen += 1

            # Hitung sisa slot untuk Pokemon
            current_pokemon = sum(1 for c in [db.by_id(cid) for cid in deck] if c and c.is_pokemon)
            remaining_pokemon = target_pokemon - current_pokemon
            if remaining_pokemon <= 0:
                break

            # Basic: 2-4 copies
            basic_copies = min(random.randint(2, 4), remaining_pokemon)
            for _ in range(basic_copies):
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

        # 2. Isi trainers (target ~29-38 kartu)
        # Hitung current trainer count untuk tracking
        def _current_trainer_count():
            return sum(1 for c in [db.by_id(cid) for cid in deck] if c and c.is_trainer)

        current_trainer = _current_trainer_count()
        target_trainers_to_add = max(0, target_trainer - current_trainer)

        if target_trainers_to_add > 0:
            supporters = [t for t in db.get_trainers(CardType.SUPPORTER)]
            items = [t for t in db.get_trainers(CardType.ITEM)]
            tools = [t for t in db.get_trainers(CardType.TOOL)]
            stadiums = [t for t in db.get_trainers(CardType.STADIUM)]

            # Supporter: 30-40% dari target trainer
            supporter_target = int(target_trainer * random.uniform(0.30, 0.40))
            random.shuffle(supporters)
            for supporter in supporters:
                if _current_trainer_count() >= supporter_target:
                    break
                if _can_add(supporter):
                    remaining = supporter_target - _current_trainer_count()
                    cnt = min(
                        config.MAX_SAME_NAME - name_counts.get(supporter.name, 0),
                        remaining
                    )
                    if cnt <= 0:
                        continue
                    cnt = min(cnt, 4)
                    for _ in range(cnt):
                        if _can_add(supporter):
                            _add(supporter)

            # Item: 40-50% dari target trainer
            item_target = min(
                target_trainer,
                int(target_trainer * random.uniform(0.40, 0.50))
            )
            random.shuffle(items)
            for item in items:
                if _current_trainer_count() >= item_target:
                    break
                if _can_add(item):
                    remaining = item_target - _current_trainer_count()
                    cnt = min(
                        config.MAX_SAME_NAME - name_counts.get(item.name, 0),
                        remaining
                    )
                    if cnt <= 0:
                        continue
                    cnt = min(cnt, 4)
                    for _ in range(cnt):
                        if _can_add(item):
                            _add(item)

            # Tools + Stadiums: sisanya (max 2-3 masing-masing)
            other_trainers = tools + stadiums
            random.shuffle(other_trainers)
            for t in other_trainers:
                if _current_trainer_count() >= target_trainer:
                    break
                if _can_add(t):
                    remaining = target_trainer - _current_trainer_count()
                    cnt = min(
                        config.MAX_SAME_NAME - name_counts.get(t.name, 0),
                        remaining,
                        2
                    )
                    if cnt <= 0:
                        continue
                    for _ in range(cnt):
                        if _can_add(t):
                            _add(t)

        # 3. Isi energy berdasarkan tipe Pokemon yang ada (target 10-15, tapi jangan overfill)
        self._fill_energy(deck, name_counts, has_ace_spec)

        # 4. Post-generation ACE SPEC cleanup (jika masih ada yang lolos)
        ace_count = sum(1 for c in [db.by_id(cid) for cid in deck] if c and c.is_ace_spec)
        while ace_count > 1:
            # Cari ACE SPEC berlebih dan ganti dengan trainer non-ACE atau energy
            for i in range(len(deck) - 1, -1, -1):
                c = db.by_id(deck[i])
                if c and c.is_ace_spec:
                    trainers_non_ace = [t for t in db.get_trainers() if not t.is_ace_spec]
                    if trainers_non_ace:
                        deck[i] = random.choice(trainers_non_ace).card_id
                    else:
                        basic_energies_flat = db.get_basic_energies()
                        if basic_energies_flat:
                            deck[i] = random.choice(basic_energies_flat).card_id
                    ace_count -= 1
                    break

        # 5. Genapkan ke 60 dengan basic energy (type-aware)
        main_type = self._detect_main_type([db.by_id(c) for c in deck if db.by_id(c)])
        basic_energies = db.get_basic_energies()
        matching = [e for e in basic_energies if e.energy_type == "{" + main_type + "}"]
        if not matching:
            matching = basic_energies
        fill_energy = random.choice(matching) if matching else None
        while len(deck) < config.DECK_SIZE and fill_energy:
            deck.append(fill_energy.card_id)
        while len(deck) > config.DECK_SIZE:
            basic_energy_ids = {e.card_id for e in db.get_basic_energies()}
            removed = False
            for i in range(len(deck) - 1, -1, -1):
                if deck[i] in basic_energy_ids:
                    deck.pop(i)
                    removed = True
                    break
            if not removed:
                deck.pop()

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

    def _detect_energy_types(self, cards: list[CardRow]) -> list[str]:
        """Detect semua tipe energy yang dibutuhkan Pokemon di deck."""
        types = set()
        for c in cards:
            if c and c.is_pokemon:
                for a in c.attacks:
                    for ch in a.cost:
                        if ch in "GRWLPFDM":
                            types.add(ch)
        return sorted(types) if types else ["G"]

    def _fill_energy(self, deck: list, name_counts: dict, has_ace_spec: list):
        """Tambahkan energy cards ke deck, bertahap dengan type-aware."""
        db = self.db
        cards_in_deck = [db.by_id(c) for c in deck if db.by_id(c)]
        main_type = self._detect_main_type(cards_in_deck)

        # Basic Energy (tidak pernah ACE SPEC)
        basic_energies = db.get_basic_energies()
        matching_energies = [e for e in basic_energies if e.energy_type == "{" + main_type + "}"]
        if not matching_energies:
            matching_energies = basic_energies

        target_energy = random.randint(*config.TARGET_ENERGY_RANGE)

        if matching_energies:
            energy_card = matching_energies[0] if len(matching_energies) == 1 else random.choice(matching_energies)
            while len(deck) < target_energy:
                deck.append(energy_card.card_id)
                name_counts[energy_card.name] = name_counts.get(energy_card.name, 0) + 1

        # Special Energy (1-2) — hanya jika cocok dengan kebutuhan Pokemon
        special = [c for c in db.get_energies() if c.stage == CardType.SPECIAL_ENERGY]
        if special:
            num_special = min(random.randint(0, 2), config.DECK_SIZE - len(deck))
            relevant_types = self._detect_energy_types(cards_in_deck)
            for _ in range(num_special):
                # Skip ACE SPEC energies agar tidak duplikat
                non_ace_special = [s for s in special if not s.is_ace_spec]
                if not non_ace_special:
                    break
                potential = []
                for s in non_ace_special:
                    s_type = s.energy_type.strip("{}")
                    if s_type in relevant_types or s_type == "C":
                        potential.append(s)
                if not potential:
                    potential = non_ace_special
                s = random.choice(potential)
                if s.card_id not in deck:
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
        Improvement: energy type-aware, bukan asal tambah energy random.
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
                basic_energy_ids = {e.card_id for e in db.get_basic_energies()}
                for i in range(len(deck) - 1, -1, -1):
                    if deck[i] in basic_energy_ids:
                        deck.pop(i)
                        fixes += 1
                        break
                deck.append(random.choice(basics).card_id)
                fixes += 1

        # 5. Fix deck size < 60 — TYPE-AWARE energy fill
        cards_now = [db.by_id(cid) for cid in deck if db.by_id(cid)]
        main_type = self._detect_main_type(cards_now)
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

        for c in cards:
            if c.is_basic and c.card_id not in used:
                line = [c.card_id]
                used.add(c.card_id)

                for c2 in cards:
                    if c2.is_stage1 and c2.prev_stage_name == c.name and c2.card_id not in used:
                        line.append(c2.card_id)
                        used.add(c2.card_id)

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

    def extract_trainer_core(self) -> dict:
        """
        Ekstrak struktur trainer dari deck.
        Returns dict dengan count per category.
        """
        db = self.db
        cards = [db.by_id(cid) for cid in self.card_ids if db.by_id(cid)]
        return {
            "supporters": [c for c in cards if c.is_trainer and c.stage == CardType.SUPPORTER],
            "items": [c for c in cards if c.is_trainer and c.stage == CardType.ITEM],
            "tools": [c for c in cards if c.is_trainer and c.stage == CardType.TOOL],
            "stadiums": [c for c in cards if c.is_trainer and c.stage == CardType.STADIUM],
        }

    def distance_to(self, other: 'DeckGenome') -> float:
        """
        Jarak (dissimilarity) ke deck lain.
        0.0 = identik, 1.0 = sama sekali berbeda.
        """
        return deck_jaccard_distance(self.card_ids, other.card_ids)

    # ─── Crossover (Improved) ───
    def crossover(self, other: 'DeckGenome') -> tuple['DeckGenome', 'DeckGenome']:
        """
        Crossover evolution line-aware + trainer core preservation.

        Improvement:
        1. Ambil evolution line dari parent A
        2. Ambil evolution line dari parent B (yang belum ada)
        3. Pertahankan RATIO trainer:energy dari parent A
        4. Isi sisa dengan candidate dari kedua parent
        5. Repair untuk fix constraint violations

        Returns:
            (child1, child2) — 2 offspring
        """
        if random.random() > config.CROSSOVER_RATE:
            return (copy.deepcopy(self), copy.deepcopy(other))

        child = []
        child_ids = set()

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

        # Isi sisa dengan trainer:energy dari self dan other,
        # pertahankan ratio trainer:energy dari self
        my_trainer_core = self.extract_trainer_core()
        other_trainer_core = other.extract_trainer_core()

        # Gabung trainer candidates
        trainer_candidates = []
        for category in ["supporters", "items", "tools", "stadiums"]:
            combined = my_trainer_core[category] + other_trainer_core[category]
            random.shuffle(combined)
            trainer_candidates.extend(combined)

        my_non_evo = self.get_non_evolution_card_ids()
        other_non_evo = other.get_non_evolution_card_ids()
        my_energies = [cid for cid in my_non_evo if self.db.by_id(cid) and self.db.by_id(cid).is_energy]
        other_energies = [cid for cid in other_non_evo if self.db.by_id(cid) and self.db.by_id(cid).is_energy]

        # Tambah trainers (prioritas supporter dulu)
        added_ids = set(child_ids)
        for trainer in trainer_candidates:
            if len(child) >= config.DECK_SIZE:
                break
            if trainer.card_id not in added_ids:
                child.append(trainer.card_id)
                added_ids.add(trainer.card_id)

        # Tambah energy
        energy_candidates = my_energies + other_energies
        random.shuffle(energy_candidates)
        for eid in energy_candidates:
            if len(child) >= config.DECK_SIZE:
                break
            if eid not in added_ids:
                child.append(eid)
                added_ids.add(eid)

        child1 = DeckGenome(child[:config.DECK_SIZE], self.db)
        child1.repair()

        # Child 2: swap parent roles
        child2 = other.crossover(self)[0]
        child2.repair()

        return child1, child2

    def _add_energy_to_fill(self):
        """Add basic energy until 60 cards (type-aware)."""
        db = self.db
        cards = [db.by_id(c) for c in self.card_ids if db.by_id(c)]
        main_type = self._detect_main_type(cards)
        basic_energies = db.get_basic_energies()
        matching = [e for e in basic_energies if e.energy_type == "{" + main_type + "}"]
        if not matching:
            matching = basic_energies
        if matching:
            e = random.choice(matching)
            key = e.card_id
            while len(self.card_ids) < config.DECK_SIZE:
                self.card_ids.append(key)

    # ─── Mutation (Improved) ───
    def mutate(self) -> bool:
        """
        Mutate deck in-place dengan probabilitas config.MUTATION_RATE.

        Improvement: tambah strategi evo_line_swap.

        Returns:
            True jika termutasi, False jika tidak.
        """
        if random.random() > config.MUTATION_RATE:
            return False

        strategy = random.choice(config.MUTATION_STRATEGIES)
        db = self.db

        if strategy == 'card_swap':
            self._mutate_card_swap(db)
        elif strategy == 'energy_tune':
            self._mutate_energy_tune(db)
        elif strategy == 'trainer_tune':
            self._mutate_trainer_tune(db)
        elif strategy == 'evo_line_swap':
            self._mutate_evo_line_swap(db)

        # Repair deck
        repaired = self.repair()
        self._fitness = None
        return True

    def _mutate_card_swap(self, db: CardDB):
        """Ganti 1-3 kartu random dengan kartu random dari database."""
        non_energy = db.get_all_non_energy()
        cards = [db.by_id(c) for c in self.card_ids]
        cards = [c for c in cards if c]
        non_basic_energy = [c for c in cards if not (c.is_energy and c.stage == CardType.BASIC_ENERGY)]
        if non_basic_energy and non_energy:
            num_swap = min(random.randint(1, 3), len(non_basic_energy))
            for _ in range(num_swap):
                old = random.choice(non_basic_energy)
                new = random.choice(non_energy)
                if old.card_id in self.card_ids:
                    idx = self.card_ids.index(old.card_id)
                    self.card_ids[idx] = new.card_id

    def _mutate_energy_tune(self, db: CardDB):
        """Adjust jumlah energy (+/- 2), type-aware."""
        basic_energies = db.get_basic_energies()
        basic_energy_ids = {e.card_id for e in basic_energies}
        energy_indices = [i for i, cid in enumerate(self.card_ids) if cid in basic_energy_ids]
        if energy_indices:
            to_remove = random.sample(energy_indices, min(random.randint(1, 2), len(energy_indices)))
            for idx in sorted(to_remove, reverse=True):
                self.card_ids.pop(idx)

            cards = [db.by_id(c) for c in self.card_ids if db.by_id(c)]
            main_type = self._detect_main_type(cards)
            matching = [e for e in basic_energies if e.energy_type == "{" + main_type + "}"]
            if not matching:
                matching = basic_energies
            if matching:
                e = random.choice(matching)
                while len(self.card_ids) < config.DECK_SIZE:
                    self.card_ids.append(e.card_id)

    def _mutate_trainer_tune(self, db: CardDB):
        """Ganti trainer random."""
        cards = [db.by_id(c) for c in self.card_ids if db.by_id(c)]
        trainers_in_deck = [c for c in cards if c.is_trainer]
        all_trainers = db.get_trainers()
        if trainers_in_deck and all_trainers:
            old = random.choice(trainers_in_deck)
            new = random.choice(all_trainers)
            if old.card_id in self.card_ids:
                idx = self.card_ids.index(old.card_id)
                self.card_ids[idx] = new.card_id

    def _mutate_evo_line_swap(self, db: CardDB):
        """
        Ganti satu evolution line dengan line lain.
        Mencari line termahal (slot terbanyak), ganti dengan line random baru.
        """
        lines = self.extract_evolution_lines()
        if len(lines) < 2:
            # Jika cuma 1 line, ganti line itu dengan line baru
            pass
        else:
            # Hapus line terbesar
            largest_line = max(lines, key=len)
            for cid in largest_line:
                if cid in self.card_ids:
                    self.card_ids.remove(cid)

        # Tambah line baru
        basics = db.get_basic_pokemon()
        random.shuffle(basics)
        name_counts = _name_counts([db.by_id(c) for c in self.card_ids if db.by_id(c)])
        has_ace_spec = [any(db.by_id(c) and db.by_id(c).is_ace_spec for c in self.card_ids)]

        for basic in basics:
            chain = db.get_evolution_chain(basic.name)
            if not chain["basic"]:
                continue

            def _can_add(card):
                if card.is_ace_spec and has_ace_spec[0]:
                    return False
                max_count = 99 if card.is_energy else config.MAX_SAME_NAME
                return name_counts.get(card.name, 0) < max_count

            # Cek sudah ada line ini
            already = any(
                db.by_id(cid) and db.by_id(cid).name == basic.name
                for cid in self.card_ids if db.by_id(cid)
            )
            if already:
                continue

            # Tambah basic
            if _can_add(basic):
                self.card_ids.append(basic.card_id)
                name_counts[basic.name] = name_counts.get(basic.name, 0) + 1

            # Stage 1
            if chain["stage1"]:
                s1 = random.choice(chain["stage1"])
                if _can_add(s1):
                    self.card_ids.append(s1.card_id)
                    name_counts[s1.name] = name_counts.get(s1.name, 0) + 1

            # Stage 2
            if chain["stage2"]:
                s2 = random.choice(chain["stage2"])
                if _can_add(s2):
                    self.card_ids.append(s2.card_id)
                    name_counts[s2.name] = name_counts.get(s2.name, 0) + 1

            break  # Tambah 1 line saja

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
