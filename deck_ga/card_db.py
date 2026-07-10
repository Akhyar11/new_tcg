"""
Card Database — Load dan query data kartu dari EN_Card_Data.csv.

Menyediakan:
- all_card_data()       → list[dict] semua kartu unik
- CardType.*            → enum kategori kartu
- DBPokemon              → class Pokemon dengan fitur penting
- query functions        → filter by type, stage, evolution line, dll
"""
import csv
import os
from dataclasses import dataclass, field
from typing import Optional

# ─── Enum Kategori Kartu ───
class CardType:
    BASIC_POKEMON = "Basic Pokémon"
    STAGE1_POKEMON = "Stage 1 Pokémon"
    STAGE2_POKEMON = "Stage 2 Pokémon"
    ITEM = "Item"
    SUPPORTER = "Supporter"
    TOOL = "Pokémon Tool"
    STADIUM = "Stadium"
    BASIC_ENERGY = "Basic Energy"
    SPECIAL_ENERGY = "Special Energy"

    @classmethod
    def all_pokemon(cls):
        return {cls.BASIC_POKEMON, cls.STAGE1_POKEMON, cls.STAGE2_POKEMON}

    @classmethod
    def all_trainers(cls):
        return {cls.ITEM, cls.SUPPORTER, cls.TOOL, cls.STADIUM}

    @classmethod
    def all_energy(cls):
        return {cls.BASIC_ENERGY, cls.SPECIAL_ENERGY}


@dataclass
class CardAttack:
    """Satu serangan dari sebuah Pokemon."""
    name: str
    cost: str          # e.g. "{G}{G}●"
    damage: str        # e.g. "100" or "30×" or "n/a"
    effect: str        # Penjelasan efek


@dataclass
class CardRow:
    """Representasi satu kartu unik (semua data dari CSV)."""
    card_id: int
    name: str
    expansion: str
    stage: str                    # CardType.*
    rule: str                     # e.g. "", "Pokémon ex", "ACE SPEC"
    category: str                 # e.g. "", "Tera(Stellar)", "Ancient"
    prev_stage_name: str          # Nama Pokemon sebelumnya (untuk evolusi)
    hp: int
    energy_type: str              # {G}, {R}, {W}, {L}, {P}, {F}, {D}, {M}, {C}
    weakness: str
    resistance: str
    retreat: int
    attacks: list[CardAttack] = field(default_factory=list)

    @property
    def is_pokemon(self) -> bool:
        return self.stage in CardType.all_pokemon()

    @property
    def is_basic(self) -> bool:
        return self.stage == CardType.BASIC_POKEMON

    @property
    def is_stage1(self) -> bool:
        return self.stage == CardType.STAGE1_POKEMON

    @property
    def is_stage2(self) -> bool:
        return self.stage == CardType.STAGE2_POKEMON

    @property
    def is_trainer(self) -> bool:
        return self.stage in CardType.all_trainers()

    @property
    def is_energy(self) -> bool:
        return self.stage in CardType.all_energy()

    @property
    def is_ace_spec(self) -> bool:
        return "ACE SPEC" in self.rule

    @property
    def is_ex(self) -> bool:
        return "ex" in self.rule  # catches "Pokémon ex" and "Mega Pokémon ex"

    @property
    def max_damage(self) -> int:
        """Damage terbesar dari semua attack."""
        best = 0
        for a in self.attacks:
            try:
                d = int(a.damage.replace("×", ""))
                best = max(best, d)
            except ValueError:
                pass
        return best

    @property
    def total_energy_cost(self) -> int:
        """Jumlah energy yang dibutuhkan untuk attack termahal."""
        best = 0
        for a in self.attacks:
            cost_str = a.cost.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
            # Hitung huruf + ●
            colored = sum(1 for c in cost_str if c in "GRWLPFDM")
            colorless = cost_str.count("●")
            total = colored + colorless
            if a.cost != "n/a" and total > best:
                best = total
        return best

    @property
    def attack_energy_types(self) -> set:
        """Set of energy types required for attacks."""
        types = set()
        for a in self.attacks:
            for c in a.cost:
                if c in "GRWLPFDM":
                    types.add(c)
        return types


class CardDB:
    """Database kartu singleton."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._by_id: dict[int, CardRow] = {}
        self._by_name: dict[str, list[CardRow]] = {}
        self._load()

    def _load(self):
        """Load CSV dan bangun index."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Card DB not found: {self.csv_path}")

        raw_rows = {}
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = int(row["Card ID"])
                if cid not in raw_rows:
                    raw_rows[cid] = {
                        "card_id": cid,
                        "name": row["Card Name"],
                        "expansion": row["Expansion"],
                        "stage": row["Stage (Pokémon)/Type (Energy and Trainer)"],
                        "rule": row["Rule"].strip() if row["Rule"].strip() != "n/a" else "",
                        "category": row["Category"].strip() if row["Category"].strip() != "n/a" else "",
                        "prev_stage_name": row["Previous stage"].strip() if row["Previous stage"].strip() != "n/a" else "",
                        "hp": int(row["HP"]) if row["HP"].strip().isdigit() else 0,
                        "energy_type": row["Type"].strip() if row["Type"].strip() != "n/a" else "",
                        "weakness": row["Weakness"].strip() if row["Weakness"].strip() != "n/a" else "",
                        "resistance": row["Resistance (Type)"].strip() if row["Resistance (Type)"].strip() != "n/a" else "",
                        "retreat": int(row["Retreat"]) if row["Retreat"].strip().isdigit() else 0,
                        "attacks": [],
                    }

                # Append attack if exists
                move_name = row["Move Name"].strip()
                if move_name and move_name != "n/a":
                    raw_rows[cid]["attacks"].append(CardAttack(
                        name=move_name,
                        cost=row["Cost"].strip() if row["Cost"].strip() != "n/a" else "",
                        damage=row["Damage"].strip() if row["Damage"].strip() != "n/a" else "",
                        effect=row["Effect Explanation"].strip() if row["Effect Explanation"].strip() != "n/a" else "",
                    ))

        for cid, data in raw_rows.items():
            card = CardRow(**data)
            self._by_id[cid] = card
            self._by_name.setdefault(card.name, []).append(card)

    # ─── Query by ID ───
    def by_id(self, card_id: int) -> Optional[CardRow]:
        return self._by_id.get(card_id)

    def by_name(self, name: str) -> list[CardRow]:
        return self._by_name.get(name, [])

    # ─── Query by Type ───
    def get_basic_pokemon(self) -> list[CardRow]:
        return [c for c in self._by_id.values() if c.is_basic]

    def get_stage1(self) -> list[CardRow]:
        return [c for c in self._by_id.values() if c.is_stage1]

    def get_stage2(self) -> list[CardRow]:
        return [c for c in self._by_id.values() if c.is_stage2]

    def get_trainers(self, stage: str = None) -> list[CardRow]:
        if stage:
            return [c for c in self._by_id.values() if c.stage == stage]
        return [c for c in self._by_id.values() if c.is_trainer]

    def get_energies(self, energy_type: str = None) -> list[CardRow]:
        if energy_type:
            return [c for c in self._by_id.values() if c.is_energy and c.energy_type == energy_type]
        return [c for c in self._by_id.values() if c.is_energy]

    def get_basic_energies(self) -> list[CardRow]:
        return [c for c in self._by_id.values() if c.stage == CardType.BASIC_ENERGY]

    def get_all_non_energy(self) -> list[CardRow]:
        return [c for c in self._by_id.values() if not c.is_energy]

    # ─── Evolution Chain ───
    def get_evolution_chain(self, basic_name: str) -> dict:
        """
        Cari evolution chain dari Basic Pokemon.

        Returns:
            {"basic": [CardRow], "stage1": [CardRow], "stage2": [CardRow]}
        """
        chain = {"basic": [], "stage1": [], "stage2": []}
        basics = self.by_name(basic_name)
        for b in basics:
            if b.is_basic:
                chain["basic"].append(b)

        # Stage 1: cari yang prev_stage_name = basic_name
        for card in self._by_id.values():
            if card.is_stage1 and card.prev_stage_name == basic_name:
                chain["stage1"].append(card)

        # Stage 2: cari yang prev_stage_name = stage1_name
        for s1 in chain["stage1"]:
            for card in self._by_id.values():
                if card.is_stage2 and card.prev_stage_name == s1.name:
                    chain["stage2"].append(card)

        return chain

    def get_all_evolution_chains(self) -> list[dict]:
        """
        Return semua evolution chain yang lengkap di dataset.
        """
        chains = []
        seen_basic = set()
        for card in self._by_id.values():
            if card.is_basic and card.name not in seen_basic:
                seen_basic.add(card.name)
                chain = self.get_evolution_chain(card.name)
                if chain["basic"]:
                    chains.append(chain)
        return chains

    # ─── Utility ───
    def name_to_energy_symbol(self, type_str: str) -> str:
        """Map 'Grass' → 'G', 'Fire' → 'R', etc."""
        mapping = {
            "Grass": "G", "Fire": "R", "Water": "W", "Lightning": "L",
            "Psychic": "P", "Fighting": "F", "Darkness": "D", "Metal": "M",
            "Dragon": "N", "Colorless": "C",
        }
        for k, v in mapping.items():
            if k in type_str:
                return v
        return "C"

    def energy_symbol_to_basic_id(self, symbol: str) -> Optional[int]:
        """'G' → 1, 'R' → 2, dll."""
        mapping = {"G": 1, "R": 2, "W": 3, "L": 4, "P": 5, "F": 6, "D": 7, "M": 8}
        return mapping.get(symbol)

    @property
    def all_card_ids(self) -> list[int]:
        return sorted(self._by_id.keys())

    @property
    def basic_energy_ids(self) -> list[int]:
        return [c.card_id for c in self.get_basic_energies()]

    def __len__(self) -> int:
        return len(self._by_id)

    def __repr__(self) -> str:
        pokemon = sum(1 for c in self._by_id.values() if c.is_pokemon)
        trainers = sum(1 for c in self._by_id.values() if c.is_trainer)
        energies = sum(1 for c in self._by_id.values() if c.is_energy)
        return f"CardDB({len(self)} unique cards: {pokemon}P, {trainers}T, {energies}E)"
