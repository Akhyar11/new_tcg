from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import json

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
        while obs and obs.get("current", {}).get("yourIndex") == 1:
            await manager.send_personal_message({"type": "update", "obs": obs}, websocket)
            await asyncio.sleep(0.5) # Beri delay sedikit agar UI frontend sempat render animasi
            
            select_data = obs.get("select")
            if not select_data or not select_data.get("option"):
                break
                
            opts = select_data["option"]
            idx = random.randint(0, len(opts)-1)
            print(f"AI auto-playing idx {idx} out of {len(opts)}")
            obs = cg.game.battle_select([idx])
        return obs

    try:
        # Kirim data awal
        await manager.send_personal_message({"type": "init", "message": "Game Engine Ready. Menunggu deck..."}, websocket)
        
        while True:
            data = await websocket.receive_text()
            action_data = json.loads(data)
            
            if action_data.get("type") == "start":
                import cg.game
                player_deck = action_data.get("deck")
                print(f"Received start request. Deck length: {len(player_deck) if player_deck else 0}")
                
                if not player_deck or len(player_deck) != 60:
                    print("Deck is not 60 cards! Falling back to gen_deck_000.csv")
                    with open("agent_rl/deck_generated/gen_deck_000.csv", "r") as f:
                        player_deck = [int(line.strip()) for line in f]
                
                try:
                    player_deck = [int(x) for x in player_deck]
                    ai_deck = player_deck.copy()
                    print(f"Deck first 10 cards: {player_deck[:10]}")
                    print("Starting battle in C++ Engine...")
                    obs, start_data = cg.game.battle_start(player_deck, ai_deck)
                    
                    if obs is None:
                        print("ERROR: User deck is invalid (obs is None)! Trying fallback deck...")
                        with open("agent_rl/deck_generated/gen_deck_000.csv", "r") as f:
                            fallback_deck = [int(line.strip()) for line in f]
                        obs, start_data = cg.game.battle_start(fallback_deck, fallback_deck)
                        
                    if obs is None:
                        print("ERROR: Even fallback deck failed!")
                        await manager.send_personal_message({"type": "error", "message": "Engine failed to start even with fallback deck."}, websocket)
                        await manager.send_personal_message({"type": "error", "message": "Engine failed to start. Deck might be invalid."}, websocket)
                    else:
                        print(f"Battle started successfully! obs keys: {list(obs.keys())}")
                        obs = await process_ai_turns(obs)
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
