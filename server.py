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

agent_p0 = None
agent_p1 = None

try:
    from eval.eval_ptr_gameplay import PointerAgent, action_mapping
    from tcg_core.models.ptr import PokemonAgent as PTRModel

    checkpoints_dir = "checkpoints"
    cp_path_p0 = os.path.join(checkpoints_dir, "model_lstm_pointer_v2_final.msgpack")
    cp_path_p1 = os.path.join(checkpoints_dir, "model_lstm_pointer_final.msgpack")
    if not os.path.exists(cp_path_p1):
        cp_path_p1 = os.path.join(checkpoints_dir, "model_final.msgpack")

    print("Memuat JAX AI Agents (P0: LSTM PTR V2 | P1: PTR V1)...")
    agent_p0 = PointerAgent("PTR_V2_P0", PTRModel, action_mapping, cp_path_p0 if os.path.exists(cp_path_p0) else None)
    print(f"✅ Player 0 AI Agent (LSTM PTR V2) Ready! Checkpoint: {cp_path_p0}")

    agent_p1 = PointerAgent("PTR_V1_P1", PTRModel, action_mapping, cp_path_p1 if os.path.exists(cp_path_p1) else None)
    print(f"✅ Player 1 AI Agent (PTR V1) Ready! Checkpoint: {cp_path_p1}")
except Exception as e:
    print(f"Gagal memuat JAX AI Agents: {e}")

def predict_ai_action(obs, player_index: int):
    agent = agent_p0 if player_index == 0 else agent_p1
    model_name = "LSTM PTR V2 (P0)" if player_index == 0 else "PTR V1 (P1)"

    if agent is not None:
        try:
            from cg.api import to_dataclass, Observation
            obs_dataclass = to_dataclass(obs, Observation) if isinstance(obs, dict) else obs
            choices = agent.select_action(obs_dataclass, deterministic=False)
            if choices:
                print(f"JAX AI [{model_name}] auto-playing choices {choices}")
                return choices
        except Exception as e:
            import traceback
            print(f"Error pada JAX AI Inference [{model_name}]: {e}")
            traceback.print_exc()

    # Fallback ke Random AI jika model bermasalah
    select_data = obs.get("select", {}) if isinstance(obs, dict) else (getattr(obs, 'select', {}) or {})
    opts = select_data.get("option", []) if isinstance(select_data, dict) else (getattr(select_data, 'option', []) or [])
    min_c = select_data.get("minCount", 1) if isinstance(select_data, dict) else (getattr(select_data, 'minCount', 1) or 1)
    opt_count = len(opts)

    import random
    target_c = min(max(min_c, 1), opt_count)
    choices = random.sample(range(opt_count), target_c) if opt_count > 0 else []
    print(f"Random AI [{model_name}] auto-playing choices {choices}")
    return choices

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
        import asyncio
        import cg.game

        while obs and obs.get("current", {}).get("yourIndex") == 1:
            await manager.send_personal_message({"type": "update", "obs": obs}, websocket)
            await asyncio.sleep(0.5)

            select_data = obs.get("select")
            if not select_data or not select_data.get("option"):
                break

            choices = predict_ai_action(obs, 1)
            obs = cg.game.battle_select(choices)
                
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

                print("Starting AI vs AI battle (P0: LSTM PTR V2 vs P1: PTR V1)...")
                deck_files = glob.glob("deck_generated/*.csv")
                
                # Pick deck for Player 0
                deck0_file = random.choice(deck_files) if deck_files else "deck_generated/gen_deck_100.csv"
                with open(deck0_file, "r") as f:
                    deck0 = [int(line.strip()) for line in f if line.strip().isdigit()]
                
                # Pick deck for Player 1
                deck1_file = random.choice(deck_files) if deck_files else "deck_generated/gen_deck_200.csv"
                with open(deck1_file, "r") as f:
                    deck1 = [int(line.strip()) for line in f if line.strip().isdigit()]

                print(f"Player 0 Deck: {deck0_file}")
                print(f"Player 1 Deck: {deck1_file}")

                if agent_p0 and hasattr(agent_p0, 'reset'): agent_p0.reset()
                if agent_p1 and hasattr(agent_p1, 'reset'): agent_p1.reset()

                obs, start_data = cg.game.battle_start(deck0, deck1)
                
                while obs and obs.get("current", {}).get("result", -1) == -1:
                    frontend_obs = json.loads(cg.game.visualize_data())[-1]
                    if "select" in obs and "select" not in frontend_obs:
                        frontend_obs["select"] = obs["select"]
                    
                    await manager.send_personal_message({"type": "update", "obs": frontend_obs}, websocket)
                    await asyncio.sleep(0.3)
                    
                    select_data = obs.get("select")
                    if not select_data or not select_data.get("option"):
                        print("No options available. Game over?")
                        break
                        
                    curr_player = obs.get("current", {}).get("yourIndex", 0)
                    choices = predict_ai_action(obs, curr_player)
                    obs = cg.game.battle_select(choices)

                if obs:
                    frontend_obs = json.loads(cg.game.visualize_data())[-1]
                    if "current" in obs and "result" in obs["current"]:
                        frontend_obs.setdefault("current", {})["result"] = obs["current"]["result"]
                    await manager.send_personal_message({"type": "update", "obs": frontend_obs}, websocket)
                else:
                    await manager.send_personal_message({"type": "update", "obs": obs}, websocket)
            elif action_data.get("type") == "start":
                import cg.game
                player_deck = action_data.get("deck")
                print(f"Received start request. Deck length: {len(player_deck) if player_deck else 0}")
                
                if not player_deck or len(player_deck) != 60:
                    print("Deck is not 60 cards! Falling back to gen_deck_000.csv")
                    with open("deck_generated/gen_deck_000.csv", "r") as f:
                        player_deck = [int(line.strip()) for line in f]
                
                try:
                    player_deck = [int(x) for x in player_deck]
                    
                    import glob
                    import random
                    deck_files = glob.glob("deck_generated/*.csv")
                    if deck_files:
                        chosen_deck = random.choice(deck_files)
                        print(f"Loading AI deck from: {chosen_deck}")
                        with open(chosen_deck, "r") as f:
                            ai_deck = [int(line.strip()) for line in f if line.strip().isdigit()]
                        if len(ai_deck) != 60:
                            print("AI deck length is not 60, fallback to player deck.")
                            ai_deck = player_deck.copy()
                    else:
                        print("No decks found in deck_generated/, fallback to player deck.")
                        ai_deck = player_deck.copy()
                    
                    print(f"Deck first 10 cards: {player_deck[:10]}")
                    print("Starting battle in C++ Engine...")
                    obs, start_data = cg.game.battle_start(player_deck, ai_deck)
                    
                    if obs is None:
                        print("ERROR: User deck is invalid (obs is None)! Trying fallback deck...")
                        with open("deck_generated/gen_deck_000.csv", "r") as f:
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
