from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import json
import os

# Limit JAX resource usage on server
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

AI_MODEL_PARAMS = None
AI_MODEL_APPLY = None
try:
    from agent_rl.model import PokemonAgent
    print("Memuat JAX AI Model...")
    model = PokemonAgent(num_actions=250)
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    dummy_seq = jnp.zeros((1, 93, 31))
    dummy_glob = jnp.zeros((1, 266))
    AI_MODEL_PARAMS = model.init(init_rng, dummy_seq, dummy_glob)
    
    cp_path = "checkpoints/model_final.msgpack"
    if os.path.exists(cp_path):
        with open(cp_path, 'rb') as f:
            AI_MODEL_PARAMS = serialization.from_bytes(AI_MODEL_PARAMS, f.read())
        print(f"JAX AI Model Checkpoint Loaded: {cp_path}")
    else:
        print("JAX AI Checkpoint not found, using random weights!")
    
    AI_MODEL_APPLY = jax.jit(model.apply)
except Exception as e:
    print(f"Gagal memuat JAX AI Model: {e}")

app = FastAPI(title="Pokemon TCG AI Server")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount folder assets (kartu, UI, dll) agar bisa diakses lewat web
import os
os.makedirs("assets/cards", exist_ok=True)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Simpan HTML Frontend di sini (sementara inline untuk kerangka)
HTML_CONTENT = """
<!DOCTYPE html>
<html>
    <head>
        <title>Pokemon TCG vs AI</title>
        <style>
            body { background: #1a1a2e; color: white; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; }
            h1 { text-align: center; color: #0f3460; text-shadow: 2px 2px 4px #e94560; }
            #board { display: flex; flex-direction: column; gap: 20px; align-items: center; margin-top: 50px; }
            .zone { border: 2px dashed #16213e; padding: 20px; min-height: 150px; min-width: 600px; border-radius: 10px; background: rgba(0,0,0,0.3); }
            .card { width: 100px; height: 140px; border-radius: 5px; cursor: pointer; transition: transform 0.2s; display: inline-block; margin: 5px; background: #0f3460; color: white; text-align: center; line-height: 140px; font-size: 10px; overflow: hidden; }
            .card img { width: 100%; height: 100%; object-fit: cover; }
            .card:hover { transform: scale(1.1); z-index: 10; position: relative; box-shadow: 0 0 15px #e94560; }
            #logs { position: fixed; bottom: 0; left: 0; width: 100%; height: 150px; background: #16213e; overflow-y: scroll; padding: 10px; box-sizing: border-box; }
        </style>
    </head>
    <body>
        <h1>Pokémon TCG Web Client</h1>
        
        <div id="board">
            <h2>Lawan (AI)</h2>
            <div id="opp_zone" class="zone">Area Lawan</div>
            
            <h2>Pemain (Kamu)</h2>
            <div id="my_zone" class="zone">Area Kamu</div>
        </div>
        
        <div id="logs">
            <p>Sistem: Memulai Koneksi WebSocket...</p>
        </div>

        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            var logs = document.getElementById("logs");
            
            function logMsg(msg) {
                var p = document.createElement("p");
                p.innerHTML = msg;
                logs.appendChild(p);
                logs.scrollTop = logs.scrollHeight;
            }

            ws.onmessage = function(event) {
                var data = JSON.parse(event.data);
                logMsg("Server: " + JSON.stringify(data));
                
                // Di sini nanti update DOM berdasarkan data observation dari C++ Engine
            };
            
            function sendAction(action_idx) {
                ws.send(JSON.stringify({action: action_idx}));
            }
        </script>
    </body>
</html>
"""

from fastapi import Request

@app.post("/validate_deck")
async def validate_deck(request: Request):
    data = await request.json()
    deck = data.get("deck", [])
    
    if len(deck) != 60:
        return {"valid": False, "reason": "Deck must contain exactly 60 cards."}
    
    try:
        import cg.game
        player_deck = [int(x) for x in deck]
        # Test the deck by starting a fake battle against itself
        obs, start_data = cg.game.battle_start(player_deck, player_deck)
        
        if obs is None:
            error_code = start_data.errorType
            reason = f"Ditolak oleh C++ Engine (Error Tidak Diketahui). Raw Code: {error_code}"
            if error_code == 1:
                reason = "Ada Kartu yang tidak dikenali oleh Engine (Invalid Card ID)."
            elif error_code == 2:
                reason = "Melanggar Aturan Deck: Terdapat lebih dari 4 kartu dengan nama yang sama (selain Basic Energy)."
            elif error_code == 3:
                reason = "Tidak ada Basic Pokémon di dalam deck!"
            elif error_code == 4:
                reason = "Melanggar Aturan Deck: Hanya boleh memiliki 1 kartu ACE SPEC / Radiant Pokémon di dalam deck!"
                
            return {"valid": False, "reason": reason}
            
        return {"valid": True, "reason": "Deck Valid"}
    except Exception as e:
        return {"valid": False, "reason": f"System Error: {str(e)}"}

# State untuk menampung koneksi pemain
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    async def process_ai_turns(obs):
        import random
        import asyncio
        import cg.game
        from cg.api import to_dataclass, Observation, OptionType
        from agent_rl.feature_extractor import extract_features
        from agent_rl.action_mapping import decode_action
        
        while obs and obs.get("current", {}).get("yourIndex") == 1:
            await manager.send_personal_message({"type": "update", "obs": obs}, websocket)
            await asyncio.sleep(0.5) # Beri delay sedikit agar UI frontend sempat render animasi
            
            select_data = obs.get("select")
            if not select_data or not select_data.get("option"):
                break
                
            opts = select_data["option"]
            opt_count = len(opts)
            min_c = select_data.get("minCount", 1)
            
            # Jika JAX model tersedia, gunakan model
            if AI_MODEL_APPLY is not None and AI_MODEL_PARAMS is not None:
                try:
                    obs_dataclass = to_dataclass(obs, Observation)
                    features = extract_features(obs_dataclass.current, obs_dataclass.select, 1) # Giliran AI = 1
                    
                    seq_input = np.expand_dims(features["seq_input"], axis=0)
                    glob_input = np.expand_dims(features["glob_input"], axis=0)
                    
                    masked_logits, _ = AI_MODEL_APPLY(AI_MODEL_PARAMS, seq_input, glob_input)
                    logits_np = np.array(masked_logits[0])
                    sorted_action_indices = np.argsort(logits_np)[::-1].tolist()
                    
                    mock_select_dict = {"options": [{"type": OptionType(o.type).name, "index": o.index} for o in obs_dataclass.select.option]}
                    choices = decode_action(sorted_action_indices, mock_select_dict, min_c)
                    
                    print(f"JAX AI (RL Model) auto-playing choices {choices}")
                    obs = cg.game.battle_select(choices)
                except Exception as e:
                    import traceback
                    print(f"Error pada JAX AI Inference: {e}")
                    traceback.print_exc()
                    print("Fallback ke random agent!")
                    idx = random.randint(0, opt_count-1)
                    obs = cg.game.battle_select([idx])
            else:
                idx = random.randint(0, opt_count-1)
                print(f"Random AI auto-playing idx {idx} out of {opt_count}")
                obs = cg.game.battle_select([idx])
                
        return obs

    try:
        # Kirim data awal
        await manager.send_personal_message({"type": "init", "message": "Game Engine Ready. Menunggu deck..."}, websocket)
        
        while True:
            data = await websocket.receive_text()
            action_data = json.loads(data)
            if action_data.get("type") == "start_ai_vs_ai":
                import cg.game
                import glob
                import random
                import asyncio
                from cg.api import to_dataclass, Observation, OptionType
                from agent_rl.feature_extractor import extract_features
                from agent_rl.action_mapping import decode_action

                print("Starting AI vs AI battle...")
                deck_files = glob.glob("agent_rl/deck/*.csv")
                
                # Pick deck for Player 0
                deck0_file = random.choice(deck_files) if deck_files else "agent_rl/deck/gen_deck_000.csv"
                with open(deck0_file, "r") as f:
                    deck0 = [int(line.strip()) for line in f if line.strip().isdigit()]
                
                # Pick deck for Player 1
                deck1_file = random.choice(deck_files) if deck_files else "agent_rl/deck/gen_deck_000.csv"
                with open(deck1_file, "r") as f:
                    deck1 = [int(line.strip()) for line in f if line.strip().isdigit()]

                print(f"Player 0 Deck: {deck0_file}")
                print(f"Player 1 Deck: {deck1_file}")

                obs, start_data = cg.game.battle_start(deck0, deck1)
                
                while obs and not obs.get("current", {}).get("isGameOver", False):
                    await manager.send_personal_message({"type": "update", "obs": obs}, websocket)
                    await asyncio.sleep(0.3)
                    
                    select_data = obs.get("select")
                    if not select_data or not select_data.get("option"):
                        print("No options available. Game over?")
                        break
                        
                    opts = select_data["option"]
                    opt_count = len(opts)
                    min_c = select_data.get("minCount", 1)
                    curr_player = obs.get("current", {}).get("yourIndex", 0)

                    # Jika JAX model tersedia, gunakan model
                    if AI_MODEL_APPLY is not None and AI_MODEL_PARAMS is not None:
                        try:
                            obs_dataclass = to_dataclass(obs, Observation)
                            features = extract_features(obs_dataclass.current, obs_dataclass.select, curr_player)
                            
                            seq_input = np.expand_dims(features["seq_input"], axis=0)
                            glob_input = np.expand_dims(features["glob_input"], axis=0)
                            
                            masked_logits, _ = AI_MODEL_APPLY(AI_MODEL_PARAMS, seq_input, glob_input)
                            logits_np = np.array(masked_logits[0])
                            sorted_action_indices = np.argsort(logits_np)[::-1].tolist()
                            
                            mock_select_dict = {"options": [{"type": OptionType(o.type).name, "index": o.index} for o in obs_dataclass.select.option]}
                            choices = decode_action(sorted_action_indices, mock_select_dict, min_c)
                            
                            print(f"JAX AI (Player {curr_player}) auto-playing choices {choices}")
                            obs = cg.game.battle_select(choices)
                        except Exception as e:
                            import traceback
                            print(f"Error pada JAX AI Inference (Player {curr_player}): {e}")
                            traceback.print_exc()
                            idx = random.randint(0, opt_count-1)
                            obs = cg.game.battle_select([idx])
                    else:
                        idx = random.randint(0, opt_count-1)
                        print(f"Random AI (Player {curr_player}) auto-playing idx {idx}")
                        obs = cg.game.battle_select([idx])

                await manager.send_personal_message({"type": "update", "obs": obs}, websocket)

            elif action_data.get("type") == "start":
                import cg.game
                player_deck = action_data.get("deck")
                print(f"Received start request. Deck length: {len(player_deck) if player_deck else 0}")
                
                if not player_deck or len(player_deck) != 60:
                    print("Deck is not 60 cards! Falling back to gen_deck_000.csv")
                    with open("agent_rl/deck/gen_deck_000.csv", "r") as f:
                        player_deck = [int(line.strip()) for line in f]
                
                try:
                    player_deck = [int(x) for x in player_deck]
                    
                    import glob
                    import random
                    deck_files = glob.glob("agent_rl/deck/*.csv")
                    if deck_files:
                        chosen_deck = random.choice(deck_files)
                        print(f"Loading AI deck from: {chosen_deck}")
                        with open(chosen_deck, "r") as f:
                            ai_deck = [int(line.strip()) for line in f if line.strip().isdigit()]
                        if len(ai_deck) != 60:
                            print("AI deck length is not 60, fallback to player deck.")
                            ai_deck = player_deck.copy()
                    else:
                        print("No decks found in agent_rl/deck/, fallback to player deck.")
                        ai_deck = player_deck.copy()
                    
                    print(f"Deck first 10 cards: {player_deck[:10]}")
                    print("Starting battle in C++ Engine...")
                    obs, start_data = cg.game.battle_start(player_deck, ai_deck)
                    
                    if obs is None:
                        print("ERROR: User deck is invalid (obs is None)! Trying fallback deck...")
                        with open("agent_rl/deck/gen_deck_000.csv", "r") as f:
                            fallback_deck = [int(line.strip()) for line in f]
                        obs, start_data = cg.game.battle_start(fallback_deck, fallback_deck)
                        
                    if obs is None:
                        print("ERROR: Even fallback deck failed!")
                        await manager.send_personal_message({"type": "error", "message": "Engine failed to start even with fallback deck."}, websocket)
                        await manager.send_personal_message({"type": "error", "message": "Engine failed to start. Deck might be invalid."}, websocket)
                    else:
                        print(f"Battle started successfully! obs keys: {list(obs.keys())}")
                        obs = await process_ai_turns(obs)
                        
                        # Auto-skip early game YES/NO prompts for Player
                        while obs and obs.get("current", {}).get("yourIndex") == 0:
                            opts = obs.get("select", {}).get("option", [])
                            # Type 1 = YES, 2 = NO
                            if len(opts) > 0 and all(o["type"] in [1, 2] for o in opts) and any(o["type"] == 1 for o in opts):
                                print("Auto-accepting startup prompt (YES)...")
                                yes_idx = next(i for i, o in enumerate(opts) if o["type"] == 1)
                                obs = cg.game.battle_select([yes_idx])
                                obs = await process_ai_turns(obs)
                            else:
                                break
                                
                        await manager.send_personal_message({"type": "update", "obs": obs}, websocket)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    await manager.send_personal_message({"type": "error", "message": f"Battle Start Error: {str(e)}"}, websocket)
                
            elif action_data.get("type") == "select":
                import cg.game
                options = action_data.get("options", [0])
                try:
                    obs = cg.game.battle_select(options)
                    obs = await process_ai_turns(obs)
                    await manager.send_personal_message({"type": "update", "obs": obs}, websocket)
                except Exception as e:
                    await manager.send_personal_message({"type": "error", "message": str(e)}, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
