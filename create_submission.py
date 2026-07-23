import os
import shutil
import tarfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.join(ROOT, "submission")

best_deck_file = os.path.join(ROOT, "best_deck.txt")
if os.path.exists(best_deck_file):
    with open(best_deck_file, "r") as f:
        best_deck_name = f.read().strip()
    DECK_PATH = os.path.join(ROOT, "new_deck", f"{best_deck_name}.csv")
else:
    DECK_PATH = os.path.join(ROOT, "new_deck", "Excadrill Drill Smash.csv")

MODEL_PATH = os.path.join(ROOT, "checkpoints", "model_lstm_pointer_final.msgpack")

# Buat folder submission
if os.path.exists(SUBMISSION_DIR):
    shutil.rmtree(SUBMISSION_DIR)
os.makedirs(SUBMISSION_DIR)

# 1. Salin deck terbaik
shutil.copy(DECK_PATH, os.path.join(SUBMISSION_DIR, "deck.csv"))

# 2. Salin model terbaik
shutil.copy(MODEL_PATH, os.path.join(SUBMISSION_DIR, "model_final.msgpack"))

# 3. Salin modul tcg_core
tcg_core_dest = os.path.join(SUBMISSION_DIR, "tcg_core")
if os.path.exists(tcg_core_dest):
    shutil.rmtree(tcg_core_dest)
shutil.copytree(os.path.join(ROOT, "tcg_core"), tcg_core_dest)

# 3.5 Salin modul cg (engine & api definitions)
cg_dest = os.path.join(SUBMISSION_DIR, "cg")
if os.path.exists(cg_dest):
    shutil.rmtree(cg_dest)
shutil.copytree(os.path.join(ROOT, "cg"), cg_dest)


# 4. Buat main.py untuk submission
MAIN_PY_CONTENT = """import os
import sys
import numpy as np

# Konfigurasi CPU untuk lingkungan Submission (menghindari error GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Tambahkan path root agar modul agent_rl bisa di-import
try:
    _base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _base_dir = "/kaggle_simulations/agent"
sys.path.append(_base_dir)

import jax
import jax.numpy as jnp
from flax import serialization
import dataclasses

from cg.api import Observation, to_observation_class, OptionType
from tcg_core.models.ptr import PokemonAgent
from tcg_core.feature_extractor import extract_features
from tcg_core.action_mapping import get_action_index_for_option, create_action_mask

# Variabel Global untuk menyimpan model (agar tidak perlu dimuat ulang setiap turn)
GLOBAL_MODEL_APPLY = None
GLOBAL_PARAMS = None
GLOBAL_CARRY = None

def init_model():
    global GLOBAL_MODEL_APPLY, GLOBAL_PARAMS, GLOBAL_CARRY
    if GLOBAL_MODEL_APPLY is not None:
        return
    
    import flax.linen as nn
    model = PokemonAgent(num_actions=250)
    rng = jax.random.PRNGKey(42)
    _, init_rng = jax.random.split(rng)
    dummy_seq = jnp.zeros((1, 173, 31))
    dummy_glob = jnp.zeros((1, 266))
    dummy_carry = nn.LSTMCell(features=256).initialize_carry(rng, (1,))
    
    params = model.init(init_rng, dummy_seq, dummy_glob, dummy_carry)
    
    # Lokasi file saat di-extract oleh server evaluasi Kaggle
    model_path = "model_final.msgpack"
    if not os.path.exists(model_path):
        model_path = "/kaggle_simulations/agent/model_final.msgpack"
        
    with open(model_path, 'rb') as f:
        params = serialization.from_bytes(params, f.read())
        
    GLOBAL_MODEL_APPLY = jax.jit(model.apply)
    # Warmup
    _ = GLOBAL_MODEL_APPLY(params, dummy_seq, dummy_glob, dummy_carry)
    GLOBAL_PARAMS = params
    GLOBAL_CARRY = dummy_carry

def softmax(x):
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / (exp_x.sum() + 1e-10)

def read_deck_csv() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().strip().split("\\n")
    deck = []
    for line in csv:
        line = line.strip()
        if line and line.isdigit():
            deck.append(int(line))
    return deck[:60]

def agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)
    
    if obs.select is None:
        return read_deck_csv()
        
    # Inisialisasi model pada turn pertama
    init_model()
    
    global GLOBAL_CARRY
    if getattr(obs.current, "turn", 0) <= 1:
        # Reset carry pada awal game (asumsi turn <= 1 adalah langkah awal)
        import flax.linen as nn
        rng = jax.random.PRNGKey(42)
        GLOBAL_CARRY = nn.LSTMCell(features=256).initialize_carry(rng, (1,))
    
    if not obs.select.option:
        return []

    your_index = obs.current.yourIndex
    features = extract_features(obs.current, obs.select, your_index)
    seq_input = np.expand_dims(features["seq_input"], axis=0)
    glob_input = np.expand_dims(features["glob_input"], axis=0)

    logits_raw, _, new_carry = GLOBAL_MODEL_APPLY(GLOBAL_PARAMS, seq_input, glob_input, GLOBAL_CARRY)
    GLOBAL_CARRY = new_carry
    logits_np = np.array(logits_raw[0])

    options = obs.select.option
    min_c = obs.select.minCount
    max_c = obs.select.maxCount
    
    mock_options = []
    for o in options:
        d = dataclasses.asdict(o)
        d["type"] = OptionType(o.type).name
        mock_options.append(d)
    mock_select = {"options": mock_options}

    mask_array = create_action_mask(mock_select, min_c, max_c)
    masked = logits_np - 1e9 * (1.0 - mask_array)
    probs = softmax(masked)

    sampled_indices = []
    if probs.sum() > 0:
        remaining = probs.copy()
        for _ in range(max_c):
            if remaining.sum() <= 0:
                break
            p = remaining / remaining.sum()
            idx = int(np.random.choice(len(p), p=p))
            if idx == 196:
                has_end_option = any(get_action_index_for_option(opt, i) == 196 for i, opt in enumerate(mock_select["options"]))
                if has_end_option:
                    sampled_indices.append(idx)
                    remaining[idx] = 0.0
                elif len(sampled_indices) >= min_c:
                    break
                else:
                    remaining[idx] = 0.0
                    continue
            else:
                sampled_indices.append(idx)
                remaining[idx] = 0.0
    else:
        sampled_indices = [196] # Fallback (Pass)

    choices = []
    for jax_idx in sampled_indices:
        for cpp_idx, opt in enumerate(mock_select["options"]):
            mapped_idx = get_action_index_for_option(opt, cpp_idx)
            if mapped_idx == jax_idx and cpp_idx not in choices:
                choices.append(cpp_idx)
                break

    if len(choices) < min_c:
        for cpp_idx in range(len(options)):
            if cpp_idx not in choices:
                choices.append(cpp_idx)
            if len(choices) >= min_c:
                break

    return choices
"""

with open(os.path.join(SUBMISSION_DIR, "main.py"), "w") as f:
    f.write(MAIN_PY_CONTENT)

# 5. Pack menjadi submission.tar.gz
TAR_NAME = "submission.tar.gz"
if os.path.exists(TAR_NAME):
    os.remove(TAR_NAME)

with tarfile.open(TAR_NAME, "w:gz") as tar:
    for item in os.listdir(SUBMISSION_DIR):
        tar.add(os.path.join(SUBMISSION_DIR, item), arcname=item)

print(f"✅ Paket submission berhasil dibuat: {TAR_NAME}")
print("Anda siap untuk meng-upload file 'submission.tar.gz' ini ke sistem kompetisi!")
