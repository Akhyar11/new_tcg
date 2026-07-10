"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

interface Card {
  'Card ID': number;
  'Card Name': string;
  [key: string]: any;
}

export default function PlayAIPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [gameState, setGameState] = useState<'SELECT_DECK' | 'PLAYING'>('SELECT_DECK');
  const [availableDecks, setAvailableDecks] = useState<any[]>([]);
  const [allCardsData, setAllCardsData] = useState<Card[]>([]);
  
  const [deck, setDeck] = useState<Card[]>([]);
  const [error, setError] = useState('');
  
  // ================= GAME ENGINE STATE =================
  const [obs, setObs] = useState<any>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);
  // =====================================================

  const [aiBenchCount, setAiBenchCount] = useState(0);
  const [aiHandCount, setAiHandCount] = useState(5);

  const [previewCard, setPreviewCard] = useState<{card: Card, energies: Card[]} | null>(null);
  const [showAttackMenu, setShowAttackMenu] = useState(false);

  useEffect(() => {
    // Check login and fetch deck
    const initializeGame = async () => {
      try {
        const res = await fetch('/api/deck');
        if (res.status === 401) {
          router.push('/login');
          return;
        }
        const data = await res.json();
        
        if (!data.decks || data.decks.length === 0) {
          setError('Kamu belum memiliki deck. Silakan rakit deck terlebih dahulu di Deck Builder!');
          setLoading(false);
          return;
        }

        // Save available decks
        setAvailableDecks(data.decks);
        
        // Fetch all cards details once
        const cardRes = await fetch('/cards.json');
        const allCards = await cardRes.json() as Card[];
        setAllCardsData(allCards);
        
        setLoading(false);
      } catch (e) {
        setError('Gagal memuat data game.');
        setLoading(false);
      }
    };
    
    initializeGame();
  }, [router]);

  const startGameWithDeck = (selectedDeck: any) => {
    try {
      const parsedIds = JSON.parse(selectedDeck.cards);
      const loadedCards = parsedIds.map((id: number) => allCardsData.find(c => c['Card ID'] === id)).filter(Boolean);
      
      setDeck(loadedCards);
      
      // ================= C++ ENGINE WEBSOCKET INTEGRATION =================
      const socket = new WebSocket('ws://localhost:8001/ws');
      socket.onopen = () => {
        console.log("Connected to C++ Engine!");
        const deckIds = loadedCards.map((c: any) => c['Card ID']);
        socket.send(JSON.stringify({ type: 'start', deck: deckIds }));
      };
      
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'update') {
          console.log("🔥 State dari C++ Engine:", data.obs);
          setObs(data.obs);
        } else if (data.type === 'error') {
          console.error("Game Engine Error:", data.message);
          alert("GAME ENGINE ERROR: " + data.message);
        } else if (data.type === 'init') {
          console.log(data.message);
        }
      };

      socket.onerror = (e) => {
        console.error("WebSocket Error, pastikan server.py berjalan di port 8001", e);
      };
      
      setWs(socket);
      setGameState('PLAYING');
    } catch (e) {
      setError('Gagal memuat kartu di deck ini.');
    }
  };

  // Helper untuk mengambil data asli kartu berdasarkan ID Engine
  const getCardInfo = (id: number) => allCardsData.find(c => c['Card ID'] === id);

  // Helper untuk mengirim opsi yang dipilih ke server
  const sendSelect = (index: number) => {
    if (ws) {
      ws.send(JSON.stringify({ type: 'select', options: [index] }));
    }
  };

  const handleDrop = (e: React.DragEvent, targetArea: number | 'generic', targetIndex: number) => {
    e.preventDefault();
    e.stopPropagation(); // Biar event gak bocor ke div luar
    if (!obs?.select || !obs.select.option) return;
    
    try {
      const data = JSON.parse(e.dataTransfer.getData('text/plain'));
      const sourceArea = data.area;
      const sourceIndex = data.index;
      
      const options = obs.select.option;
      let matchIdx = -1;

      if (obs.select.type === 1) { // CARD selection (e.g. Setup phase)
        matchIdx = options.findIndex((opt: any) => opt.type === 3 && opt.area === sourceArea && opt.index === sourceIndex);
      } else if (obs.select.type === 0) { // MAIN phase
        if (targetArea === 'generic') {
          // Play Trainer/Basic Pokemon (generic play)
          matchIdx = options.findIndex((opt: any) => opt.type === 7 && opt.index === sourceIndex);
        } else {
          // Targeted play (Attach Energy/Tool, Evolve)
          matchIdx = options.findIndex((opt: any) => 
            (opt.type === 8 || opt.type === 9) && 
            opt.area === sourceArea && opt.index === sourceIndex && 
            opt.inPlayArea === targetArea && opt.inPlayIndex === targetIndex
          );
          
          // Fallback: If no Evolve/Attach, but dropped on bench, check if it's PLAY basic pokemon
          if (matchIdx === -1 && targetArea === 5) {
             matchIdx = options.findIndex((opt: any) => opt.type === 7 && opt.index === sourceIndex);
          }
          // Fallback 2: If we dropped on generic Active/Bench but it just expects PLAY
          if (matchIdx === -1) {
             matchIdx = options.findIndex((opt: any) => opt.type === 7 && opt.index === sourceIndex);
          }
        }
      }

      if (matchIdx !== -1) {
        sendSelect(matchIdx);
      } else {
        console.warn("Aksi Drag & Drop tidak valid untuk target ini.");
      }
    } catch (err) {
      console.error("Drop error", err);
    }
  };

  // Data Derivation dari obs C++
  let playerHand: any[] = [];
  let playerActive: any = null;
  let playerBench: any[] = [null, null, null, null, null];
  
  let aiActive: any = null;
  let aiBench: any[] = [null, null, null, null, null];

  if (obs && obs.current) {
    const p0 = obs.current.players[0];
    const p1 = obs.current.players[1];
    
    // Player 0 (Anda)
    if (p0.hand) {
      playerHand = p0.hand.map((c: any) => ({ ...getCardInfo(c.id), engineSerial: c.serial, engineId: c.id }));
    }
    if (p0.active && p0.active[0]) {
      playerActive = { 
        ...getCardInfo(p0.active[0].id), 
        engineSerial: p0.active[0].serial, 
        engineId: p0.active[0].id, 
        hp: p0.active[0].hp, 
        maxHp: p0.active[0].maxHp,
        energyCards: p0.active[0].energyCards ? p0.active[0].energyCards.map((ec: any) => getCardInfo(ec.id)).filter(Boolean) : []
      };
    }
    if (p0.bench) {
      p0.bench.forEach((b: any, i: number) => {
        if (b) playerBench[i] = { 
          ...getCardInfo(b.id), 
          engineSerial: b.serial, 
          engineId: b.id,
          energyCards: b.energyCards ? b.energyCards.map((ec: any) => getCardInfo(ec.id)).filter(Boolean) : []
        };
      });
    }

    // Player 1 (AI)
    if (p1.active && p1.active[0]) {
      // Periksa apakah face-down
      if (p1.active[0].id) {
         aiActive = { 
           ...getCardInfo(p1.active[0].id), 
           engineSerial: p1.active[0].serial, 
           engineId: p1.active[0].id, 
           hp: p1.active[0].hp, 
           maxHp: p1.active[0].maxHp,
           energyCards: p1.active[0].energyCards ? p1.active[0].energyCards.map((ec: any) => getCardInfo(ec.id)).filter(Boolean) : []
         };
      } else {
         aiActive = { isFacedown: true };
      }
    }
    if (p1.bench) {
      p1.bench.forEach((b: any, i: number) => {
        if (b && b.id) aiBench[i] = { 
          ...getCardInfo(b.id), 
          engineSerial: b.serial, 
          engineId: b.id,
          energyCards: b.energyCards ? b.energyCards.map((ec: any) => getCardInfo(ec.id)).filter(Boolean) : []
        };
        else if (b) aiBench[i] = { isFacedown: true };
      });
    }
  }

  // ================= MOCK DRAG & DROP & ATTACK DIBUANG =================
  // Semua aksi sekarang harus melalui Action Panel dari C++ Engine
  
  if (loading) {
    return <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#050b14', color: 'white', fontFamily: 'sans-serif' }}>Memuat Arena...</div>;
  }

  if (error) {
    return (
      <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#050b14', color: 'white', fontFamily: 'sans-serif', gap: '1rem' }}>
        <h2 style={{ color: '#ef4444' }}>Tidak Bisa Memulai Permainan</h2>
        <p>{error}</p>
        <Link href="/deck" style={{ padding: '0.8rem 1.5rem', background: '#3b82f6', color: 'white', textDecoration: 'none', borderRadius: '8px', fontWeight: 'bold' }}>Pergi ke Deck Builder</Link>
      </div>
    );
  }

  // ================= PRE-GAME: DECK SELECTION =================
  if (gameState === 'SELECT_DECK') {
    return (
      <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: '#0f172a', color: 'white', fontFamily: '"Inter", sans-serif', padding: '4rem 2rem', alignItems: 'center' }}>
        
        <Link href="/" style={{ position: 'absolute', top: '2rem', left: '2rem', color: '#94a3b8', textDecoration: 'none', fontWeight: 'bold' }}>← Kembali</Link>
        
        <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem', background: 'linear-gradient(to right, #38bdf8, #818cf8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Pilih Deck Anda
        </h1>
        <p style={{ color: '#94a3b8', marginBottom: '3rem' }}>Pilih salah satu deck yang telah Anda rakit untuk menghadapi JAX AI.</p>

        <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap', justifyContent: 'center', maxWidth: '1000px' }}>
          {availableDecks.map(d => {
            const parsedIds = JSON.parse(d.cards);
            const cardCount = parsedIds.length;
            // Gunakan kartu pertama di deck sebagai gambar cover (atau 1 jika kosong)
            const coverCardId = cardCount > 0 ? parsedIds[0] : 1; 

            return (
              <div 
                key={d.id} 
                onClick={() => startGameWithDeck(d)}
                style={{ 
                  width: '300px', 
                  height: '180px',
                  borderRadius: '20px', 
                  cursor: 'pointer', 
                  transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)', 
                  display: 'flex', 
                  flexDirection: 'column', 
                  justifyContent: 'flex-end',
                  padding: '1.5rem',
                  position: 'relative',
                  overflow: 'hidden',
                  border: '1px solid rgba(255,255,255,0.1)',
                  boxShadow: '0 10px 30px -10px rgba(0,0,0,0.5)'
                }}
                onMouseEnter={e => { 
                  e.currentTarget.style.transform = 'translateY(-8px) scale(1.02)'; 
                  e.currentTarget.style.borderColor = 'rgba(56, 189, 248, 0.6)'; 
                  e.currentTarget.style.boxShadow = '0 20px 40px -10px rgba(56, 189, 248, 0.3)';
                  const img = e.currentTarget.querySelector('.deck-bg') as HTMLElement;
                  if (img) img.style.transform = 'scale(1.1)';
                }}
                onMouseLeave={e => { 
                  e.currentTarget.style.transform = 'translateY(0) scale(1)'; 
                  e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'; 
                  e.currentTarget.style.boxShadow = '0 10px 30px -10px rgba(0,0,0,0.5)';
                  const img = e.currentTarget.querySelector('.deck-bg') as HTMLElement;
                  if (img) img.style.transform = 'scale(1)';
                }}
              >
                {/* Background Image Layer */}
                <div 
                  className="deck-bg"
                  style={{
                    position: 'absolute',
                    inset: 0,
                    backgroundImage: `url(/assets/cards/${coverCardId}.png)`,
                    backgroundSize: 'cover',
                    backgroundPosition: 'center 25%',
                    transition: 'transform 0.6s ease-out',
                    zIndex: 0
                  }}
                ></div>
                
                {/* Gradient Overlay to make text readable */}
                <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(to top, rgba(15,23,42,1) 0%, rgba(15,23,42,0.6) 50%, rgba(15,23,42,0.2) 100%)', zIndex: 1 }}></div>

                {/* Content Layer */}
                <div style={{ position: 'relative', zIndex: 2, display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <h3 style={{ margin: 0, fontSize: '1.4rem', fontWeight: '900', color: 'white', textShadow: '0 2px 4px rgba(0,0,0,0.8)' }}>
                    {d.name}
                  </h3>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ 
                      color: cardCount === 60 ? '#34d399' : '#fbbf24', 
                      fontWeight: 'bold', 
                      background: 'rgba(0,0,0,0.5)', 
                      padding: '0.3rem 0.8rem', 
                      borderRadius: '20px', 
                      fontSize: '0.85rem',
                      border: `1px solid ${cardCount === 60 ? 'rgba(52, 211, 153, 0.3)' : 'rgba(251, 191, 36, 0.3)'}`,
                      backdropFilter: 'blur(4px)'
                    }}>
                      {cardCount} / 60 Kartu
                    </span>
                    {cardCount < 60 && <span style={{ fontSize: '0.8rem', color: '#f87171', fontWeight: 'bold', background: 'rgba(0,0,0,0.5)', padding: '0.2rem 0.5rem', borderRadius: '4px' }}>Incomplete</span>}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  // ================= MAIN ARENA =================
  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: '#121212', color: 'white', fontFamily: '"Inter", sans-serif', overflow: 'hidden' }}>
      
      {/* TOP ROW: OPPONENT HAND */}
      <div style={{ padding: '1rem', display: 'flex', justifyContent: 'center', gap: '5px', minHeight: '120px' }}>
        {[...Array(aiHandCount)].map((_, i) => (
          <div key={i} style={{ width: '75px', height: '105px' }}>
            <img src="/assets/cards/1.png" style={{ width: '100%', height: '100%', borderRadius: '4px' }} alt="Card Back" />
          </div>
        ))}
      </div>

      {/* MIDDLE SECTION: PLAYMAT */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: '0 2rem', gap: '3rem', position: 'relative' }}>
        
        {/* ================= OPPONENT HALF ================= */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          
          {/* Left: Prize Cards */}
          <div style={{ width: '180px', display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
            {[...Array(6)].map((_, i) => (
              <div key={i} style={{ width: '60px', height: '84px', border: '1px dashed #333', borderRadius: '4px', position: 'relative' }}>
                <img src="/assets/cards/1.png" style={{ width: '100%', height: '100%', borderRadius: '4px', position: 'absolute', top: 0, left: 0 }} alt="Prize Back" />
              </div>
            ))}
          </div>

          {/* Center: Active & Bench */}
          <div style={{ flex: 1, display: 'flex', justifyContent: 'center', gap: '2rem', alignItems: 'center' }}>
            {/* Active */}
            <div style={{ position: 'relative', width: '140px', height: '196px', border: '2px solid rgba(255,255,255,0.1)', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {aiActive && !aiActive.isFacedown ? (
                <>
                  <img src={`/assets/cards/${aiActive['Card ID']}.png`} style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '8px', zIndex: 10, position: 'relative' }} onContextMenu={(e) => { e.preventDefault(); setPreviewCard({card: aiActive, energies: aiActive.energyCards || []}); }} />
                  {/* Energy Underneath */}
                  {aiActive.energyCards && aiActive.energyCards.map((en: any, i: number) => (
                    <img key={i} src={`/assets/cards/${en['Card ID']}.png`} style={{ position: 'absolute', width: '100%', height: '100%', top: `${(i+1)*15}px`, left: 0, zIndex: 1, borderRadius: '8px' }} />
                  ))}
                  {aiActive.hp !== undefined && aiActive.hp < aiActive.maxHp && (
                    <div style={{ position: 'absolute', bottom: '-10px', right: '-10px', background: '#ef4444', color: 'white', borderRadius: '50%', width: '30px', height: '30px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', zIndex: 20 }}>{aiActive.maxHp - aiActive.hp}</div>
                  )}
                </>
              ) : aiActive && aiActive.isFacedown ? (
                <img src="/assets/cards/1.png" style={{ width: '100%', height: '100%', borderRadius: '8px' }} />
              ) : null}
            </div>

            {/* Bench Row */}
            <div style={{ display: 'flex', gap: '10px' }}>
              {[...Array(5)].map((_, i) => (
                <div key={i} style={{ position: 'relative', width: '90px', height: '126px', border: '2px dashed rgba(255,255,255,0.1)', borderRadius: '6px' }}>
                  {aiBench[i] && !aiBench[i].isFacedown ? (
                    <>
                      <img src={`/assets/cards/${aiBench[i]['Card ID']}.png`} style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '6px', zIndex: 10, position: 'relative' }} onContextMenu={(e) => { e.preventDefault(); setPreviewCard({card: aiBench[i], energies: aiBench[i].energyCards || []}); }} />
                      {aiBench[i].energyCards && aiBench[i].energyCards.map((en: any, ei: number) => (
                        <img key={ei} src={`/assets/cards/${en['Card ID']}.png`} style={{ position: 'absolute', width: '100%', height: '100%', top: `${(ei+1)*10}px`, left: 0, zIndex: 1, borderRadius: '6px' }} />
                      ))}
                    </>
                  ) : aiBench[i] && aiBench[i].isFacedown ? (
                    <img src="/assets/cards/1.png" style={{ width: '100%', height: '100%', borderRadius: '6px' }} />
                  ) : null}
                </div>
              ))}
            </div>
          </div>

          {/* Right: Discard & Deck */}
          <div style={{ width: '180px', display: 'flex', flexDirection: 'column', gap: '20px', alignItems: 'center' }}>
             <div style={{ fontSize: '0.8rem', color: '#888' }}>Hand [{aiHandCount}]</div>
             <div style={{ position: 'relative', width: '80px', height: '112px', border: '2px dashed #444', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
               <span style={{color: '#666', fontSize:'0.8rem'}}>Discard</span>
             </div>
             <div style={{ position: 'relative', width: '80px', height: '112px' }}>
               <img src="/assets/cards/1.png" style={{ width: '100%', height: '100%', borderRadius: '4px' }} />
               <div style={{position: 'absolute', top: '-20px', width: '100%', textAlign: 'center', fontSize: '0.8rem', color: '#888'}}>Deck [{60 - aiHandCount}]</div>
             </div>
          </div>
        </div>
        
        {/* ACTION PANEL */}
        {obs?.select && obs.current?.yourIndex === 0 && (
          <div style={{ position: 'absolute', left: '2rem', top: '50%', transform: 'translateY(-50%)', background: 'rgba(30, 30, 30, 0.95)', border: '1px solid #444', borderRadius: '8px', padding: '1rem', zIndex: 100, width: '250px' }}>
            <h3 style={{ margin: '0 0 1rem 0', color: '#38bdf8', fontSize: '1rem' }}>Pilih Aksi <span style={{fontSize:'0.7rem', color:'#888'}}>(Ctx: {obs.select.context})</span></h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '40vh', overflowY: 'auto' }}>
              {obs.select.option.map((opt: any, idx: number) => {
                if (opt.type === 7 || opt.type === 8 || opt.type === 9 || (opt.type === 3 && opt.area === 2)) return null;
                let label = `Option ${idx}`;
                if (opt.type === 1) label = "YES";
                else if (opt.type === 2) label = "NO";
                else if (opt.type === 14) label = "END TURN";
                else if (opt.type === 12) label = "RETREAT";
                else if (opt.type === 10) label = `USE ABILITY (Area ${opt.area} Idx ${opt.index})`;
                else if (opt.type === 13) label = `ATTACK ${opt.attackId}`;
                else if (opt.type === 3) label = `SELECT CARD (Area ${opt.area} Idx ${opt.index})`;
                
                return (
                  <button key={idx} onClick={() => sendSelect(idx)} style={{ background: '#2a2a2a', border: '1px solid #444', borderRadius: '4px', padding: '0.6rem', color: 'white', textAlign: 'left', cursor: 'pointer', fontSize: '0.85rem' }}>{label}</button>
                )
              })}
            </div>
          </div>
        )}

        {/* Divider */}
        <div style={{ height: '1px', background: 'rgba(255,255,255,0.05)', margin: '0 4rem' }} />

        {/* ================= PLAYER HALF ================= */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }} onDragOver={(e) => e.preventDefault()} onDrop={(e) => handleDrop(e, 'generic', 0)}>
          
          {/* Left: Prize Cards */}
          <div style={{ width: '180px', display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
            {[...Array(6)].map((_, i) => (
              <div key={i} style={{ width: '60px', height: '84px', border: '1px dashed #333', borderRadius: '4px', position: 'relative' }}>
                <img src="/assets/cards/1.png" style={{ width: '100%', height: '100%', borderRadius: '4px', position: 'absolute', top: 0, left: 0 }} alt="Prize Back" />
              </div>
            ))}
          </div>

          {/* Center: Active & Bench */}
          <div style={{ flex: 1, display: 'flex', justifyContent: 'center', gap: '2rem', alignItems: 'center' }}>
            {/* Active */}
            <div 
              style={{ position: 'relative', width: '140px', height: '196px', border: '2px solid rgba(255,255,255,0.1)', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => handleDrop(e, 4, 0)}
            >
              {playerActive ? (
                <>
                  <img src={`/assets/cards/${playerActive['Card ID']}.png`} style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '8px', zIndex: 10, position: 'relative' }} onContextMenu={(e) => { e.preventDefault(); setPreviewCard({card: playerActive, energies: playerActive.energyCards || []}); }} />
                  {playerActive.energyCards && playerActive.energyCards.map((en: any, i: number) => (
                    <img key={i} src={`/assets/cards/${en['Card ID']}.png`} style={{ position: 'absolute', width: '100%', height: '100%', top: `${(i+1)*15}px`, left: 0, zIndex: 1, borderRadius: '8px' }} />
                  ))}
                  {playerActive.hp !== undefined && playerActive.hp < playerActive.maxHp && (
                    <div style={{ position: 'absolute', top: '-10px', right: '-10px', background: '#ef4444', color: 'white', borderRadius: '50%', width: '30px', height: '30px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', zIndex: 20 }}>{playerActive.maxHp - playerActive.hp}</div>
                  )}
                </>
              ) : null}
            </div>

            {/* Bench Row */}
            <div style={{ display: 'flex', gap: '10px' }}>
              {playerBench.map((benchCard, i) => (
                <div 
                  key={i} 
                  style={{ position: 'relative', width: '90px', height: '126px', border: '2px dashed rgba(255,255,255,0.1)', borderRadius: '6px' }}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => handleDrop(e, 5, i)}
                >
                  {benchCard ? (
                    <>
                      <img src={`/assets/cards/${benchCard['Card ID']}.png`} style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '6px', zIndex: 10, position: 'relative' }} onContextMenu={(e) => { e.preventDefault(); setPreviewCard({card: benchCard, energies: benchCard.energyCards || []}); }} />
                      {benchCard.energyCards && benchCard.energyCards.map((en: any, ei: number) => (
                        <img key={ei} src={`/assets/cards/${en['Card ID']}.png`} style={{ position: 'absolute', width: '100%', height: '100%', top: `${(ei+1)*10}px`, left: 0, zIndex: 1, borderRadius: '6px' }} />
                      ))}
                    </>
                  ) : null}
                </div>
              ))}
            </div>
          </div>

          {/* Right: Discard & Deck */}
          <div style={{ width: '180px', display: 'flex', flexDirection: 'column', gap: '20px', alignItems: 'center' }}>
             <div style={{ position: 'relative', width: '80px', height: '112px' }}>
               <img src="/assets/cards/1.png" style={{ width: '100%', height: '100%', borderRadius: '4px' }} />
               <div style={{position: 'absolute', top: '-20px', width: '100%', textAlign: 'center', fontSize: '0.8rem', color: '#888'}}>Deck [{deck.length}]</div>
             </div>
             <div style={{ position: 'relative', width: '80px', height: '112px', border: '2px dashed #444', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
               <span style={{color: '#666', fontSize:'0.8rem'}}>Discard</span>
             </div>
             <div style={{ fontSize: '0.8rem', color: '#888' }}>Hand [{playerHand.length}]</div>
          </div>

        </div>
      </div>

      {/* BOTTOM ROW: PLAYER HAND */}
      <div style={{ padding: '1rem', display: 'flex', justifyContent: 'center', gap: '5px', minHeight: '150px', background: 'rgba(0,0,0,0.5)' }}>
        {playerHand.map((card, i) => (
          <div 
            key={i} 
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData('text/plain', JSON.stringify({ area: 2, index: i }));
            }}
            style={{ width: '90px', height: '126px', cursor: 'grab', transition: 'transform 0.2s', position: 'relative' }}
            onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-20px) scale(1.1)'; e.currentTarget.style.zIndex = '100'; }}
            onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0) scale(1)'; e.currentTarget.style.zIndex = '1'; }}
            onContextMenu={(e) => { e.preventDefault(); setPreviewCard({card: card, energies: []}); }}
          >
            <img src={`/assets/cards/${card['Card ID']}.png`} style={{ width: '100%', height: '100%', borderRadius: '6px', pointerEvents: 'none', boxShadow: '0 5px 15px rgba(0,0,0,0.5)' }} />
          </div>
        ))}
      </div>

      {/* CARD PREVIEW MODAL */}
      {previewCard && (
        <div onClick={() => setPreviewCard(null)} style={{ position: 'fixed', inset: 0, zIndex: 1000, background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(5px)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'zoom-out' }}>
          <div style={{ display: 'flex', gap: '2rem', alignItems: 'flex-start', maxWidth: '1000px', width: '100%', padding: '2rem' }} onClick={(e) => e.stopPropagation()}>
            {/* Card Image */}
            <div style={{ width: '350px', flexShrink: 0, boxShadow: '0 0 50px rgba(255,255,255,0.2)', borderRadius: '16px', position: 'relative' }}>
              <img src={`/assets/cards/${previewCard.card['Card ID']}.png`} style={{ width: '100%', height: 'auto', borderRadius: '16px' }} />
            </div>
            
            {/* Card Details */}
            <div style={{ flex: 1, background: 'rgba(30, 41, 59, 0.9)', padding: '2rem', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.1)', position: 'relative' }}>
              <button onClick={() => setPreviewCard(null)} style={{ position: 'absolute', top: '1rem', right: '1rem', background: 'transparent', border: 'none', color: '#94a3b8', fontSize: '1.5rem', cursor: 'pointer' }}>✕</button>
              <h2 style={{ margin: '0 0 0.5rem 0', fontSize: '2rem', color: '#38bdf8', paddingRight: '2rem' }}>{previewCard.card['Card Name']}</h2>
              <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', color: '#94a3b8' }}>
                <span style={{ background: '#334155', padding: '0.3rem 0.8rem', borderRadius: '20px', fontSize: '0.9rem' }}>{previewCard.card['Stage (Pokémon)/Type (Energy and Trainer)']}</span>
                {previewCard.card['HP'] && <span style={{ background: '#ef4444', color: 'white', padding: '0.3rem 0.8rem', borderRadius: '20px', fontSize: '0.9rem', fontWeight: 'bold' }}>HP {previewCard.card['HP']}</span>}
                {previewCard.card['Type'] && <span style={{ background: '#eab308', color: 'black', padding: '0.3rem 0.8rem', borderRadius: '20px', fontSize: '0.9rem', fontWeight: 'bold' }}>{previewCard.card['Type']}</span>}
              </div>
              
              {/* Attached Energies Info */}
              {previewCard.energies && previewCard.energies.length > 0 && (
                <div style={{ marginBottom: '1.5rem', background: 'linear-gradient(to right, rgba(245, 158, 11, 0.1), transparent)', borderLeft: '4px solid #f59e0b', padding: '0.8rem 1rem', borderRadius: '4px' }}>
                  <div style={{ fontSize: '0.9rem', color: '#f59e0b', fontWeight: 'bold', marginBottom: '0.5rem' }}>Attached Energy: {previewCard.energies.length}</div>
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    {previewCard.energies.map((en, idx) => (
                      <div key={idx} style={{ background: 'rgba(0,0,0,0.4)', padding: '0.3rem 0.6rem', borderRadius: '6px', fontSize: '0.85rem', color: '#cbd5e1', border: '1px solid rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <img src={`/assets/cards/${en['Card ID']}.png`} style={{ width: '20px', height: '28px', borderRadius: '2px', objectFit: 'cover' }} />
                        {en['Card Name']}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {previewCard.card['Attack 1 Name'] && (
                  <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 'bold', marginBottom: '0.5rem', fontSize: '1.1rem' }}>
                      <span>{previewCard.card['Attack 1 Name']}</span>
                      <span>{previewCard.card['Attack 1 Damage']}</span>
                    </div>
                    <div style={{ color: '#94a3b8', fontSize: '0.9rem', marginBottom: '0.5rem' }}>Cost: {previewCard.card['Attack 1 Cost']}</div>
                    {previewCard.card['Attack 1 Text'] && <div style={{ fontSize: '0.9rem', fontStyle: 'italic', color: '#cbd5e1' }}>{previewCard.card['Attack 1 Text']}</div>}
                  </div>
                )}
                {previewCard.card['Attack 2 Name'] && (
                  <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 'bold', marginBottom: '0.5rem', fontSize: '1.1rem' }}>
                      <span>{previewCard.card['Attack 2 Name']}</span>
                      <span>{previewCard.card['Attack 2 Damage']}</span>
                    </div>
                    <div style={{ color: '#94a3b8', fontSize: '0.9rem', marginBottom: '0.5rem' }}>Cost: {previewCard.card['Attack 2 Cost']}</div>
                    {previewCard.card['Attack 2 Text'] && <div style={{ fontSize: '0.9rem', fontStyle: 'italic', color: '#cbd5e1' }}>{previewCard.card['Attack 2 Text']}</div>}
                  </div>
                )}
                {previewCard.card['Card Text'] && (
                  <div style={{ background: 'rgba(56, 189, 248, 0.1)', border: '1px solid rgba(56, 189, 248, 0.3)', padding: '1rem', borderRadius: '8px' }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '0.5rem', color: '#38bdf8' }}>Ability / Effect</div>
                    <div style={{ fontSize: '0.95rem', lineHeight: '1.5' }}>{previewCard.card['Card Text']}</div>
                  </div>
                )}
              </div>
              <div style={{ marginTop: '2rem', display: 'flex', gap: '2rem', fontSize: '0.9rem', color: '#94a3b8', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '1rem' }}>
                <div><strong>Weakness:</strong> {previewCard.card['Weakness'] || 'None'}</div>
                <div><strong>Resistance:</strong> {previewCard.card['Resistance'] || 'None'}</div>
                <div><strong>Retreat:</strong> {previewCard.card['Retreat Cost'] || '0'}</div>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
