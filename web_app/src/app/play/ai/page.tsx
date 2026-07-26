"use client";

import React, { useEffect, useState, useMemo, useRef } from 'react';
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

  const [previewCard, setPreviewCard] = useState<{ card: Card, energies: Card[] } | null>(null);
  const [discardViewer, setDiscardViewer] = useState<{cards: any[], title: string} | null>(null);
  const [showAttackMenu, setShowAttackMenu] = useState(false);
  const [multiSelectIndices, setMultiSelectIndices] = useState<number[]>([]);

  useEffect(() => {
    // Reset multi select if context changes
    setMultiSelectIndices([]);
  }, [obs?.select?.context]);

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
  const getCardInfo = (id: number) => {
    const rows = allCardsData.filter(c => c['Card ID'] === id);
    if (rows.length === 0) return {};
    const base = { ...rows[0] };
    base.moves = rows.map(r => ({
      name: r['Move Name'],
      cost: r['Cost'],
      damage: r['Damage'],
      effect: r['Effect Explanation']
    })).filter(m => m.name || m.effect); // Simpan semua move/effect yang valid
    return base;
  };

  // Helper untuk mengirim opsi yang dipilih ke server
  const sendSelect = (index: number) => {
    if (ws) {
      if (index === -1) {
        ws.send(JSON.stringify({ type: 'select', options: [] }));
      } else {
        ws.send(JSON.stringify({ type: 'select', options: [index] }));
      }
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
  let playerDiscard: any[] = [];
  let playerDeckCount: number = deck.length;
  let playerPrizeCards: any[] = [...Array(6)];

  let aiActive: any = null;
  let aiBench: any[] = [null, null, null, null, null];
  let aiDiscard: any[] = [];
  let aiHandCountVal: number = aiHandCount;
  let aiDeckCountVal: number = 60 - aiHandCount;
  let aiPrizeCards: any[] = [...Array(6)];

  let stadiumCard: any = null;

  const deckOrDiscardOptions = useMemo(() => {
    if (!obs?.select?.option) return [];
    return obs.select.option
      .map((opt: any, originalIdx: number) => ({ ...opt, originalIdx }))
      .filter((opt: any) => opt.type === 3 && (opt.area === 1 || opt.area === 3));
  }, [obs?.select]);

  if (obs && obs.current) {
    const p0 = obs.current.players[0];
    const p1 = obs.current.players[1];

    // Player 0 (Anda)
    if (p0.deckCount !== undefined) playerDeckCount = p0.deckCount;
    if (p0.prize) playerPrizeCards = p0.prize;
    if (p0.hand) {
      playerHand = p0.hand.map((c: any) => ({ ...getCardInfo(c.id), engineSerial: c.serial, engineId: c.id }));
    }
    if (p0.active && p0.active.length > 0) {
      if (p0.active[0] === null) {
        playerActive = { isFacedown: true };
      } else {
        playerActive = {
          ...getCardInfo(p0.active[0].id),
          engineSerial: p0.active[0].serial,
          engineId: p0.active[0].id,
          hp: p0.active[0].hp,
          maxHp: p0.active[0].maxHp,
          energyCards: p0.active[0].energyCards ? p0.active[0].energyCards.map((ec: any) => getCardInfo(ec.id)).filter(Boolean) : []
        };
      }
    }
    if (p0.bench) {
      p0.bench.forEach((b: any, i: number) => {
        if (b === null) {
          playerBench[i] = { isFacedown: true };
        } else if (b && b.id) {
          playerBench[i] = {
            ...getCardInfo(b.id),
            engineSerial: b.serial,
            engineId: b.id,
            energyCards: b.energyCards ? b.energyCards.map((ec: any) => getCardInfo(ec.id)).filter(Boolean) : []
          };
        }
      });
    }
    if (p0.discard) {
      playerDiscard = p0.discard.map((c: any) => ({ ...getCardInfo(c.id), engineSerial: c.serial, engineId: c.id }));
    }

    // Player 1 (AI)
    if (p1.deckCount !== undefined) aiDeckCountVal = p1.deckCount;
    if (p1.handCount !== undefined) aiHandCountVal = p1.handCount;
    else if (p1.hand) aiHandCountVal = p1.hand.length;
    if (p1.prize) aiPrizeCards = p1.prize;
    if (p1.active && p1.active.length > 0) {
      if (p1.active[0] === null) {
        aiActive = { isFacedown: true };
      } else {
        aiActive = {
          ...getCardInfo(p1.active[0].id),
          engineSerial: p1.active[0].serial,
          engineId: p1.active[0].id,
          hp: p1.active[0].hp,
          maxHp: p1.active[0].maxHp,
          energyCards: p1.active[0].energyCards ? p1.active[0].energyCards.map((ec: any) => getCardInfo(ec.id)).filter(Boolean) : []
        };
      }
    }
    if (p1.bench) {
      p1.bench.forEach((b: any, i: number) => {
        if (b === null) {
          aiBench[i] = { isFacedown: true };
        } else if (b && b.id) {
          aiBench[i] = {
            ...getCardInfo(b.id),
            engineSerial: b.serial,
            engineId: b.id,
            energyCards: b.energyCards ? b.energyCards.map((ec: any) => getCardInfo(ec.id)).filter(Boolean) : []
          };
        }
      });
    }
    // Stadium Card (Global Field)
    if (p0.stadium && p0.stadium.length > 0 && p0.stadium[0]) {
      stadiumCard = { ...getCardInfo(p0.stadium[0].id), engineSerial: p0.stadium[0].serial, engineId: p0.stadium[0].id };
    } else if (p1.stadium && p1.stadium.length > 0 && p1.stadium[0]) {
      stadiumCard = { ...getCardInfo(p1.stadium[0].id), engineSerial: p1.stadium[0].serial, engineId: p1.stadium[0].id };
    }
  }

  // ================= MOCK DRAG & DROP & ATTACK DIBUANG =================
  // Semua aksi sekarang harus melalui Action Panel dari C++ Engine

  // Helper untuk merender HP Progress Bar Badge
  const renderHPBadge = (card: any) => {
    if (!card || card.isFacedown || card.hp === undefined || !card.maxHp) return null;
    const currentHp = card.hp;
    const maxHp = card.maxHp;
    const pct = Math.max(0, Math.min(100, (currentHp / maxHp) * 100));
    const barColor = pct > 50 ? '#34d399' : pct > 20 ? '#fbbf24' : '#f43f5e';
    return (
      <div style={{ position: 'absolute', bottom: '4px', left: '4px', right: '4px', background: 'rgba(15, 23, 42, 0.9)', borderRadius: '6px', padding: '3px 6px', border: '1px solid rgba(255,255,255,0.15)', zIndex: 30, display: 'flex', flexDirection: 'column', gap: '2px', backdropFilter: 'blur(4px)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', fontWeight: 'bold', color: 'white' }}>
          <span>HP</span>
          <span style={{ color: barColor }}>{currentHp}/{maxHp}</span>
        </div>
        <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.2)', borderRadius: '2px', overflow: 'hidden' }}>
          <div style={{ width: `${pct}%`, height: '100%', background: barColor, transition: 'width 0.3s' }} />
        </div>
      </div>
    );
  };

  // Helper untuk merender 6 slot Prize Cards (tampilan berkurang saat diambil)
  const renderPrizeSlot = (prizeList: any[], index: number, isPlayer: boolean = false) => {
    const hasCard = index < prizeList.length && prizeList[index] !== undefined && prizeList[index] !== 'empty';
    const prizeOptIdx = isPlayer && obs?.select?.option
      ? obs.select.option.findIndex((opt: any) => opt.type === 3 && opt.area === 6 && opt.index === index)
      : -1;
    const isSelectable = prizeOptIdx !== -1;

    return (
      <div 
        key={index} 
        onClick={() => { if (isSelectable) sendSelect(prizeOptIdx); }}
        style={{ 
          width: '60px', 
          height: '84px', 
          border: isSelectable ? '2px solid #ef4444' : '1px dashed rgba(255,255,255,0.1)', 
          borderRadius: '4px', 
          position: 'relative', 
          cursor: isSelectable ? 'pointer' : 'default', 
          boxShadow: isSelectable ? '0 0 12px rgba(239, 68, 68, 0.8)' : 'none',
          background: 'rgba(0,0,0,0.25)'
        }}
      >
        {hasCard && (
          <img 
            src="/assets/cards/back.png" 
            style={{ width: '100%', height: '100%', borderRadius: '4px', position: 'absolute', top: 0, left: 0 }} 
            alt="Prize Card" 
          />
        )}
      </div>
    );
  };

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

  const isPlayerTurn = obs?.current?.yourIndex === 0;

  // ================= MAIN ARENA =================
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#050b14', color: 'white', fontFamily: '"Inter", sans-serif', overflowX: 'hidden', position: 'relative' }}>

      {/* TURN STATUS BANNER HUD */}
      {obs && (
        <div style={{ position: 'absolute', top: '1rem', left: '50%', transform: 'translateX(-50%)', zIndex: 200, display: 'flex', alignItems: 'center', gap: '0.8rem', background: isPlayerTurn ? 'rgba(56, 189, 248, 0.12)' : 'rgba(244, 63, 94, 0.12)', border: `1px solid ${isPlayerTurn ? 'rgba(56, 189, 248, 0.5)' : 'rgba(244, 63, 94, 0.5)'}`, borderRadius: '30px', padding: '0.5rem 1.5rem', backdropFilter: 'blur(8px)', boxShadow: isPlayerTurn ? '0 0 20px rgba(56, 189, 248, 0.25)' : '0 0 20px rgba(244, 63, 94, 0.25)' }}>
          <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: isPlayerTurn ? '#38bdf8' : '#f43f5e', boxShadow: `0 0 10px ${isPlayerTurn ? '#38bdf8' : '#f43f5e'}` }} />
          <span style={{ fontWeight: '900', fontSize: '0.9rem', letterSpacing: '1.5px', color: isPlayerTurn ? '#38bdf8' : '#f43f5e' }}>
            {isPlayerTurn ? 'GILIRAN ANDA (PLAYER)' : 'GILIRAN LAWAN (JAX AI)'}
          </span>
        </div>
      )}

      {/* TOP ROW: OPPONENT HAND */}
      <div style={{ padding: '0.5rem 1rem', display: 'flex', justifyContent: 'center', gap: '5px', minHeight: '120px', flexShrink: 0 }}>
        {[...Array(aiHandCountVal)].map((_, i) => (
          <div key={i} style={{ width: '75px', height: '105px' }}>
            <img src="/assets/cards/back.png" style={{ width: '100%', height: '100%', borderRadius: '4px' }} alt="Card Back" />
          </div>
        ))}
      </div>

      {/* MIDDLE SECTION: PLAYMAT */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: '1rem 2rem', gap: '2rem', position: 'relative' }}>

        {/* ================= OPPONENT HALF ================= */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>

          {/* Left: Prize Cards */}
          <div style={{ width: '180px', display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
            {[...Array(6)].map((_, i) => renderPrizeSlot(aiPrizeCards, i, false))}
          </div>

          {/* Center: Active & Bench */}
          <div style={{ flex: 1, display: 'flex', justifyContent: 'center', gap: '2rem', alignItems: 'center' }}>
            {/* Active */}
            <div style={{ position: 'relative', width: '140px', height: '196px', border: '2px solid rgba(255,255,255,0.1)', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {aiActive && !aiActive.isFacedown ? (
                <>
                  <img src={`/assets/cards/${aiActive['Card ID']}.png`} style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '8px', zIndex: 10, position: 'relative' }} onContextMenu={(e) => { e.preventDefault(); setPreviewCard({ card: aiActive, energies: aiActive.energyCards || [] }); }} />
                  {/* Compact Energy Fan Underneath */}
                  {aiActive.energyCards && aiActive.energyCards.map((en: any, i: number) => (
                    <img 
                      key={i} 
                      src={`/assets/cards/${en['Card ID']}.png`} 
                      style={{ 
                        position: 'absolute', 
                        width: '100%', 
                        height: '100%', 
                        top: `${Math.min((i + 1) * 4, 12)}px`, 
                        left: `-${Math.min((i + 1) * 8, 24)}px`, 
                        zIndex: 1, 
                        borderRadius: '8px',
                        boxShadow: '0 2px 6px rgba(0,0,0,0.5)'
                      }} 
                    />
                  ))}
                  {/* Energy Badge Pill */}
                  {aiActive.energyCards && aiActive.energyCards.length > 0 && (
                    <div style={{ position: 'absolute', bottom: '-8px', left: '30%', transform: 'translateX(-50%)', zIndex: 30, background: 'rgba(15, 23, 42, 0.95)', border: '1px solid rgba(251, 191, 36, 0.6)', borderRadius: '12px', padding: '1px 6px', fontSize: '0.65rem', fontWeight: 'bold', color: '#fbbf24', display: 'flex', alignItems: 'center', gap: '3px', boxShadow: '0 2px 8px rgba(0,0,0,0.8)' }}>
                      <span>⚡</span>
                      <span>{aiActive.energyCards.length}</span>
                    </div>
                  )}
                  {renderHPBadge(aiActive)}
                </>
              ) : aiActive && aiActive.isFacedown ? (
                <img src="/assets/cards/back.png" style={{ width: '100%', height: '100%', borderRadius: '8px' }} />
              ) : null}
            </div>

            {/* Bench Row */}
            <div style={{ display: 'flex', gap: '10px' }}>
              {[...Array(5)].map((_, i) => (
                <div key={i} style={{ position: 'relative', width: '90px', height: '126px', border: '2px dashed rgba(255,255,255,0.1)', borderRadius: '6px' }}>
                  {aiBench[i] && !aiBench[i].isFacedown ? (
                    <>
                      <img src={`/assets/cards/${aiBench[i]['Card ID']}.png`} style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '6px', zIndex: 10, position: 'relative' }} onContextMenu={(e) => { e.preventDefault(); setPreviewCard({ card: aiBench[i], energies: aiBench[i].energyCards || [] }); }} />
                      {aiBench[i].energyCards && aiBench[i].energyCards.map((en: any, ei: number) => (
                        <img 
                          key={ei} 
                          src={`/assets/cards/${en['Card ID']}.png`} 
                          style={{ 
                            position: 'absolute', 
                            width: '100%', 
                            height: '100%', 
                            top: `${Math.min((ei + 1) * 3, 9)}px`, 
                            left: `-${Math.min((ei + 1) * 6, 18)}px`, 
                            zIndex: 1, 
                            borderRadius: '6px',
                            boxShadow: '0 2px 4px rgba(0,0,0,0.5)'
                          }} 
                        />
                      ))}
                      {aiBench[i].energyCards && aiBench[i].energyCards.length > 0 && (
                        <div style={{ position: 'absolute', bottom: '-6px', left: '30%', transform: 'translateX(-50%)', zIndex: 30, background: 'rgba(15, 23, 42, 0.95)', border: '1px solid rgba(251, 191, 36, 0.6)', borderRadius: '10px', padding: '1px 4px', fontSize: '0.6rem', fontWeight: 'bold', color: '#fbbf24', display: 'flex', alignItems: 'center', gap: '2px', boxShadow: '0 2px 6px rgba(0,0,0,0.8)' }}>
                          <span>⚡</span>
                          <span>{aiBench[i].energyCards.length}</span>
                        </div>
                      )}
                      {renderHPBadge(aiBench[i])}
                    </>
                  ) : aiBench[i] && aiBench[i].isFacedown ? (
                    <img src="/assets/cards/back.png" style={{ width: '100%', height: '100%', borderRadius: '6px' }} />
                  ) : null}
                </div>
              ))}
            </div>
          </div>

          {/* Right: Discard & Deck */}
          <div style={{ width: '180px', display: 'flex', flexDirection: 'column', gap: '20px', alignItems: 'center' }}>
            <div style={{ fontSize: '0.8rem', color: '#888' }}>Hand [{aiHandCountVal}]</div>
            <div 
              onClick={() => setDiscardViewer({cards: aiDiscard, title: "Discard Lawan"})}
              style={{ position: 'relative', width: '80px', height: '112px', border: '2px dashed #444', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}>
              {aiDiscard.length > 0 ? (
                <img src={`/assets/cards/${aiDiscard[aiDiscard.length-1]['Card ID']}.png`} style={{width:'100%', height:'100%', borderRadius:'4px', objectFit: 'contain'}} />
              ) : (
                <span style={{ color: '#666', fontSize: '0.8rem' }}>Discard</span>
              )}
            </div>
            <div style={{ position: 'relative', width: '80px', height: '112px' }}>
              <img src="/assets/cards/back.png" style={{ width: '100%', height: '100%', borderRadius: '4px' }} />
              <div style={{ position: 'absolute', top: '-20px', width: '100%', textAlign: 'center', fontSize: '0.8rem', color: '#888' }}>Deck [{aiDeckCountVal}]</div>
            </div>
          </div>
        </div>

        {/* ACTION PANEL HUD */}
        {obs?.select && obs.current?.yourIndex === 0 && (
          <div style={{ position: 'absolute', left: '250px', top: '55%', transform: 'translateY(-50%)', background: 'rgba(15, 23, 42, 0.92)', border: '1px solid rgba(56, 189, 248, 0.3)', borderRadius: '16px', padding: '1.2rem', zIndex: 100, width: '280px', backdropFilter: 'blur(12px)', boxShadow: '0 15px 35px rgba(0,0,0,0.6)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.08)', pb: '0.5rem' }}>
              <h3 style={{ margin: 0, color: '#38bdf8', fontSize: '1rem', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span>⚡</span> Panel Aksi
              </h3>
              <span style={{ fontSize: '0.7rem', background: 'rgba(255,255,255,0.06)', padding: '0.2rem 0.5rem', borderRadius: '10px', color: '#94a3b8' }}>Ctx: {obs.select.context}</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', maxHeight: '40vh', overflowY: 'auto' }}>
              {obs.select.minCount === 0 && (
                <button onClick={() => sendSelect(-1)} style={{ background: 'linear-gradient(135deg, #0ea5e9, #2563eb)', border: 'none', borderRadius: '8px', padding: '0.75rem', color: 'white', textAlign: 'center', cursor: 'pointer', fontSize: '0.9rem', fontWeight: 'bold', boxShadow: '0 4px 12px rgba(14, 165, 233, 0.3)' }}>SELESAI / LANJUT →</button>
              )}
              {obs.select.option.map((opt: any, idx: number) => {
                if (opt.type === 7 || opt.type === 8 || opt.type === 9 || opt.type === 13 || (opt.type === 3 && opt.area === 2)) return null;
                if (opt.type === 3 && (opt.area === 6 || opt.area === 1 || opt.area === 3)) return null;

                let label = `Aksi ${idx}`;
                let bgStyle = 'linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02))';
                let borderStyle = '1px solid rgba(255,255,255,0.1)';
                let textColor = 'white';

                if (opt.type === 1) { label = "✓ YES"; bgStyle = 'linear-gradient(135deg, #10b981, #059669)'; borderStyle = 'none'; }
                else if (opt.type === 2) { label = "✕ NO"; bgStyle = 'linear-gradient(135deg, #ef4444, #dc2626)'; borderStyle = 'none'; }
                else if (opt.type === 14) { label = "⏹ END TURN"; bgStyle = 'linear-gradient(135deg, #f43f5e, #be123c)'; borderStyle = 'none'; }
                else if (opt.type === 12) { label = "🔄 RETREAT"; bgStyle = 'linear-gradient(135deg, #f59e0b, #d97706)'; borderStyle = 'none'; }
                else if (opt.type === 10) { label = `🔮 USE ABILITY (Area ${opt.area} Idx ${opt.index})`; bgStyle = 'linear-gradient(135deg, #8b5cf6, #6d28d9)'; borderStyle = 'none'; }
                else if (opt.type === 13) { label = `⚔️ ATTACK ${opt.attackId}`; bgStyle = 'linear-gradient(135deg, #dc2626, #991b1b)'; borderStyle = 'none'; }
                else if (opt.type === 3) { label = `🎴 SELECT CARD (Area ${opt.area} Idx ${opt.index})`; }

                return (
                  <button key={idx} onClick={() => sendSelect(idx)} style={{ background: bgStyle, border: borderStyle, borderRadius: '8px', padding: '0.7rem 0.9rem', color: textColor, textAlign: 'left', cursor: 'pointer', fontSize: '0.85rem', fontWeight: '600', transition: 'all 0.2s', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span>{label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* CENTRAL STADIUM SLOT */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', margin: '-0.5rem 0' }}>
          <div style={{ height: '1px', background: 'rgba(255,255,255,0.08)', flex: 1 }} />
          
          <div 
            onClick={() => { if (stadiumCard) setPreviewCard({ card: stadiumCard, energies: [] }); }}
            style={{ 
              width: '90px', 
              height: '126px', 
              border: stadiumCard ? '2px solid #34d399' : '2px dashed rgba(52, 211, 153, 0.3)', 
              borderRadius: '8px', 
              background: stadiumCard ? 'rgba(52, 211, 153, 0.1)' : 'rgba(15, 23, 42, 0.7)', 
              display: 'flex', 
              flexDirection: 'column', 
              alignItems: 'center', 
              justifyContent: 'center', 
              position: 'relative', 
              cursor: stadiumCard ? 'pointer' : 'default',
              boxShadow: stadiumCard ? '0 0 20px rgba(52, 211, 153, 0.4)' : 'none',
              transition: 'all 0.3s ease',
              zIndex: 50,
              margin: '0 2rem'
            }}
          >
            {stadiumCard ? (
              <>
                <img 
                  src={`/assets/cards/${stadiumCard['Card ID']}.png`} 
                  style={{ width: '100%', height: '100%', borderRadius: '6px', objectFit: 'contain' }} 
                  alt={stadiumCard['Card Name'] || 'Stadium'} 
                />
                <div style={{ position: 'absolute', bottom: '-8px', background: '#34d399', color: '#0f172a', padding: '0.1rem 0.5rem', borderRadius: '10px', fontSize: '0.65rem', fontWeight: '900', letterSpacing: '0.5px' }}>
                  STADIUM
                </div>
              </>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px', color: 'rgba(52, 211, 153, 0.5)' }}>
                <span style={{ fontSize: '1.2rem' }}>🏟️</span>
                <span style={{ fontSize: '0.65rem', fontWeight: 'bold', letterSpacing: '1px' }}>STADIUM</span>
              </div>
            )}
          </div>

          <div style={{ height: '1px', background: 'rgba(255,255,255,0.08)', flex: 1 }} />
        </div>

        {/* ================= PLAYER HALF ================= */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }} onDragOver={(e) => e.preventDefault()} onDrop={(e) => handleDrop(e, 'generic', 0)}>

          {/* Left: Prize Cards */}
          <div style={{ width: '180px', display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
            {[...Array(6)].map((_, i) => renderPrizeSlot(playerPrizeCards, i, true))}
          </div>

          {/* Center: Active & Bench */}
          <div style={{ flex: 1, display: 'flex', justifyContent: 'center', gap: '2rem', alignItems: 'center' }}>
            {/* Active */}
            <div
              style={{ position: 'relative', width: '140px', height: '196px', border: '2px solid rgba(255,255,255,0.1)', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => handleDrop(e, 4, 0)}
            >
              {playerActive && !playerActive.isFacedown ? (
                <>
                  <img src={`/assets/cards/${playerActive['Card ID']}.png`} style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '8px', zIndex: 10, position: 'relative' }} onContextMenu={(e) => { e.preventDefault(); setPreviewCard({ card: playerActive, energies: playerActive.energyCards || [] }); }} />
                  {/* Compact Energy Fan Underneath */}
                  {playerActive.energyCards && playerActive.energyCards.map((en: any, i: number) => (
                    <img 
                      key={i} 
                      src={`/assets/cards/${en['Card ID']}.png`} 
                      style={{ 
                        position: 'absolute', 
                        width: '100%', 
                        height: '100%', 
                        top: `${Math.min((i + 1) * 4, 12)}px`, 
                        left: `-${Math.min((i + 1) * 8, 24)}px`, 
                        zIndex: 1, 
                        borderRadius: '8px',
                        boxShadow: '0 2px 6px rgba(0,0,0,0.5)'
                      }} 
                    />
                  ))}
                  {/* Energy Badge Pill */}
                  {playerActive.energyCards && playerActive.energyCards.length > 0 && (
                    <div style={{ position: 'absolute', bottom: '-8px', left: '30%', transform: 'translateX(-50%)', zIndex: 30, background: 'rgba(15, 23, 42, 0.95)', border: '1px solid rgba(251, 191, 36, 0.6)', borderRadius: '12px', padding: '1px 6px', fontSize: '0.65rem', fontWeight: 'bold', color: '#fbbf24', display: 'flex', alignItems: 'center', gap: '3px', boxShadow: '0 2px 8px rgba(0,0,0,0.8)' }}>
                      <span>⚡</span>
                      <span>{playerActive.energyCards.length}</span>
                    </div>
                  )}
                  {renderHPBadge(playerActive)}
                </>
              ) : playerActive && playerActive.isFacedown ? (
                <img src="/assets/cards/back.png" style={{ width: '100%', height: '100%', borderRadius: '8px' }} />
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
                  {benchCard && !benchCard.isFacedown ? (
                    <>
                      <img src={`/assets/cards/${benchCard['Card ID']}.png`} style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '6px', zIndex: 10, position: 'relative' }} onContextMenu={(e) => { e.preventDefault(); setPreviewCard({ card: benchCard, energies: benchCard.energyCards || [] }); }} />
                      {benchCard.energyCards && benchCard.energyCards.map((en: any, ei: number) => (
                        <img 
                          key={ei} 
                          src={`/assets/cards/${en['Card ID']}.png`} 
                          style={{ 
                            position: 'absolute', 
                            width: '100%', 
                            height: '100%', 
                            top: `${Math.min((ei + 1) * 3, 9)}px`, 
                            left: `-${Math.min((ei + 1) * 6, 18)}px`, 
                            zIndex: 1, 
                            borderRadius: '6px',
                            boxShadow: '0 2px 4px rgba(0,0,0,0.5)'
                          }} 
                        />
                      ))}
                      {benchCard.energyCards && benchCard.energyCards.length > 0 && (
                        <div style={{ position: 'absolute', bottom: '-6px', left: '30%', transform: 'translateX(-50%)', zIndex: 30, background: 'rgba(15, 23, 42, 0.95)', border: '1px solid rgba(251, 191, 36, 0.6)', borderRadius: '10px', padding: '1px 4px', fontSize: '0.6rem', fontWeight: 'bold', color: '#fbbf24', display: 'flex', alignItems: 'center', gap: '2px', boxShadow: '0 2px 6px rgba(0,0,0,0.8)' }}>
                          <span>⚡</span>
                          <span>{benchCard.energyCards.length}</span>
                        </div>
                      )}
                      {renderHPBadge(benchCard)}
                    </>
                  ) : benchCard && benchCard.isFacedown ? (
                    <img src="/assets/cards/back.png" style={{ width: '100%', height: '100%', borderRadius: '6px' }} />
                  ) : null}
                </div>
              ))}
            </div>
          </div>

          {/* Right: Discard & Deck */}
          <div style={{ width: '180px', display: 'flex', flexDirection: 'column', gap: '20px', alignItems: 'center' }}>
            <div style={{ position: 'relative', width: '80px', height: '112px' }}>
              <img src="/assets/cards/back.png" style={{ width: '100%', height: '100%', borderRadius: '4px' }} />
              <div style={{ position: 'absolute', top: '-20px', width: '100%', textAlign: 'center', fontSize: '0.8rem', color: '#888' }}>Deck [{playerDeckCount}]</div>
            </div>
            <div 
              onClick={() => setDiscardViewer({cards: playerDiscard, title: "Discard Anda"})}
              style={{ position: 'relative', width: '80px', height: '112px', border: '2px dashed #444', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}>
              {playerDiscard.length > 0 ? (
                <img src={`/assets/cards/${playerDiscard[playerDiscard.length-1]['Card ID']}.png`} style={{width:'100%', height:'100%', borderRadius:'4px', objectFit: 'contain'}} />
              ) : (
                <span style={{ color: '#666', fontSize: '0.8rem' }}>Discard</span>
              )}
            </div>
            <div style={{ fontSize: '0.8rem', color: '#888' }}>Hand [{playerHand.length}]</div>
          </div>

        </div>
      </div>

      {/* BOTTOM ROW: PLAYER HAND */}
      <div style={{ padding: '1rem', display: 'flex', justifyContent: 'center', gap: '5px', minHeight: '160px', flexShrink: 0, background: 'rgba(0,0,0,0.5)', overflow: 'visible' }}>
        {playerHand.map((card, i) => (
          <div
            key={i}
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData('text/plain', JSON.stringify({ area: 2, index: i }));
            }}
            onClick={() => {
              if (obs?.select && obs.select.option) {
                const matchIdx = obs.select.option.findIndex((opt: any) => opt.type === 3 && opt.area === 2 && opt.index === i);
                if (matchIdx !== -1) {
                  sendSelect(matchIdx);
                }
              }
            }}
            style={{ width: '90px', height: '126px', cursor: 'grab', transition: 'transform 0.2s', position: 'relative' }}
            onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-20px) scale(1.1)'; e.currentTarget.style.zIndex = '100'; }}
            onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0) scale(1)'; e.currentTarget.style.zIndex = '1'; }}
            onContextMenu={(e) => { e.preventDefault(); setPreviewCard({ card: card, energies: [] }); }}
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

              {/* Action Buttons for PLAY (7) and ABILITY (10) */}
              {(() => {
                let cardArea = -1;
                let cardIndex = -1;
                const hIdx = playerHand.findIndex((c: any) => c.engineSerial === previewCard.card.engineSerial);
                if (hIdx !== -1) { cardArea = 2; cardIndex = hIdx; }
                else if (playerActive && playerActive.engineSerial === previewCard.card.engineSerial) { cardArea = 4; cardIndex = 0; }
                else {
                  const bIdx = playerBench.findIndex((c: any) => c && c.engineSerial === previewCard.card.engineSerial);
                  if (bIdx !== -1) { cardArea = 5; cardIndex = bIdx; }
                }

                if (obs?.select?.option) {
                  const playOptIdx = cardArea === 2 ? obs.select.option.findIndex((opt: any) => opt.type === 7 && opt.index === cardIndex) : -1;
                  const abilityOptIdx = obs.select.option.findIndex((opt: any) => opt.type === 10 && opt.area === cardArea && (cardArea === 4 || opt.index === cardIndex));

                  return (
                    <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem' }}>
                      {playOptIdx !== -1 && (
                        <button onClick={() => { sendSelect(playOptIdx); setPreviewCard(null); }} style={{ flex: 1, padding: '0.8rem', background: '#3b82f6', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', fontSize: '1rem', boxShadow: '0 4px 15px rgba(59, 130, 246, 0.4)' }}>GUNAKAN KARTU</button>
                      )}
                      {abilityOptIdx !== -1 && (
                        <button onClick={() => { sendSelect(abilityOptIdx); setPreviewCard(null); }} style={{ flex: 1, padding: '0.8rem', background: '#8b5cf6', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', fontSize: '1rem', boxShadow: '0 4px 15px rgba(139, 92, 246, 0.4)' }}>GUNAKAN ABILITY</button>
                      )}
                    </div>
                  );
                }
                return null;
              })()}

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
                {(() => {
                   const attackOptions = obs?.select?.option?.map((opt: any, idx: number) => ({ opt, idx })).filter((x: any) => x.opt.type === 13) || [];
                   let attackCounter = 0;
                   return previewCard.card.moves && previewCard.card.moves.map((move: any, moveIdx: number) => {
                     const isAbility = move.name?.startsWith('[Ability]');
                     let btnOptIdx = -1;
                     if (!isAbility && playerActive && previewCard.card.engineSerial === playerActive.engineSerial) {
                       if (attackCounter < attackOptions.length) {
                         btnOptIdx = attackOptions[attackCounter].idx;
                       }
                       attackCounter++;
                     }
                     return (
                       <div key={moveIdx} style={{ background: isAbility ? 'rgba(139, 92, 246, 0.1)' : 'rgba(0,0,0,0.3)', border: isAbility ? '1px solid rgba(139, 92, 246, 0.3)' : 'none', padding: '1rem', borderRadius: '8px' }}>
                         {move.name && (
                           <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 'bold', marginBottom: '0.5rem', fontSize: '1.1rem' }}>
                             <span style={{ color: isAbility ? '#c4b5fd' : 'white' }}>{move.name}</span>
                             {move.damage && <span>{move.damage}</span>}
                           </div>
                         )}
                         {move.cost && <div style={{ color: '#94a3b8', fontSize: '0.9rem', marginBottom: '0.5rem' }}>Cost: {move.cost}</div>}
                         {move.effect && <div style={{ fontSize: '0.9rem', fontStyle: 'italic', color: '#cbd5e1', lineHeight: '1.5' }}>{move.effect}</div>}
                         {btnOptIdx !== -1 && (
                            <button onClick={() => { sendSelect(btnOptIdx); setPreviewCard(null); }} style={{ marginTop: '1rem', width: '100%', padding: '0.8rem', background: '#ef4444', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', fontSize: '1rem', boxShadow: '0 4px 15px rgba(239, 68, 68, 0.4)' }}>GUNAKAN SERANGAN</button>
                         )}
                       </div>
                     );
                   });
                })()}
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

      {/* DISCARD VIEWER MODAL */}
      {discardViewer && (
        <div onClick={() => setDiscardViewer(null)} style={{ position: 'fixed', inset: 0, zIndex: 900, background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(5px)', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '2rem', overflowY: 'auto' }}>
          <h2 style={{ color: '#38bdf8', marginBottom: '2rem' }}>{discardViewer.title} ({discardViewer.cards.length})</h2>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', justifyContent: 'center', maxWidth: '1200px' }} onClick={(e) => e.stopPropagation()}>
            {discardViewer.cards.map((card, i) => (
              <img 
                key={i} 
                src={`/assets/cards/${card['Card ID']}.png`} 
                style={{ width: '120px', height: '168px', borderRadius: '8px', cursor: 'pointer', transition: 'transform 0.2s' }} 
                onMouseEnter={e => e.currentTarget.style.transform = 'scale(1.1)'}
                onMouseLeave={e => e.currentTarget.style.transform = 'scale(1)'}
                onClick={() => setPreviewCard({card: card, energies: []})}
                onContextMenu={(e) => { e.preventDefault(); setPreviewCard({card: card, energies: []}); }}
              />
            ))}
            {discardViewer.cards.length === 0 && <p style={{color: '#94a3b8'}}>Discard pile kosong.</p>}
          </div>
          <button onClick={() => setDiscardViewer(null)} style={{ marginTop: '2rem', padding: '0.8rem 2rem', background: '#334155', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer' }}>Tutup</button>
        </div>
      )}

        {/* CARD SELECTOR MODAL (Deck / Discard) */}
        {deckOrDiscardOptions.length > 0 && obs.current?.yourIndex === 0 && (
          <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', background: 'rgba(0,0,0,0.9)', zIndex: 9999, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
            <h2 style={{ color: 'white', marginBottom: '1rem', textShadow: '0 2px 4px rgba(0,0,0,0.8)' }}>
              Pilih Kartu (Min: {obs.select.minCount}, Max: {obs.select.maxCount})
            </h2>
            
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', maxWidth: '80%', maxHeight: '60%', overflowY: 'auto', justifyContent: 'center', padding: '20px' }}>
              {deckOrDiscardOptions.map((opt: any) => {
                let cardObj: any = null;
                if (opt.area === 1 && obs.select.deck) cardObj = obs.select.deck[opt.index];
                else if (opt.area === 3 && obs.players) cardObj = obs.players[obs.current.yourIndex].discard[opt.index];
                
                if (cardObj && !cardObj.name) {
                   cardObj = getCardInfo(cardObj.id);
                }

                const isSelected = multiSelectIndices.includes(opt.originalIdx);
                
                return (
                  <div 
                    key={opt.originalIdx} 
                    onClick={() => {
                      if (isSelected) {
                        setMultiSelectIndices(prev => prev.filter(i => i !== opt.originalIdx));
                      } else {
                        if (multiSelectIndices.length < obs.select.maxCount) {
                          setMultiSelectIndices(prev => [...prev, opt.originalIdx]);
                        }
                      }
                    }}
                    style={{ 
                      width: '120px', height: '168px', 
                      border: isSelected ? '4px solid #ef4444' : '2px solid transparent', 
                      borderRadius: '8px', cursor: 'pointer',
                      transform: isSelected ? 'translateY(-10px)' : 'none',
                      transition: 'all 0.2s',
                      boxShadow: isSelected ? '0 10px 20px rgba(239, 68, 68, 0.5)' : '0 4px 6px rgba(0,0,0,0.3)',
                      background: '#222'
                    }}
                  >
                    {cardObj ? (
                      <img src={`/assets/cards/${cardObj.id}.png`} alt={cardObj.name || "Card"} style={{ width: '100%', height: '100%', borderRadius: '4px' }} />
                    ) : (
                      <div style={{ color: 'white', padding: '10px', textAlign: 'center', fontSize: '0.8rem', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        Tutup / Tidak Diketahui
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
            
            <div style={{ marginTop: '2rem', display: 'flex', gap: '1rem' }}>
              <button 
                disabled={multiSelectIndices.length < obs.select.minCount}
                onClick={() => {
                  if (ws) {
                    ws.send(JSON.stringify({ type: 'select', options: multiSelectIndices }));
                  }
                  setMultiSelectIndices([]);
                }}
                style={{ 
                  background: multiSelectIndices.length >= obs.select.minCount ? '#ef4444' : '#666', 
                  color: 'white', padding: '1rem 3rem', borderRadius: '8px', 
                  fontWeight: 'bold', border: 'none', 
                  cursor: multiSelectIndices.length >= obs.select.minCount ? 'pointer' : 'not-allowed', 
                  fontSize: '1.2rem',
                  boxShadow: multiSelectIndices.length >= obs.select.minCount ? '0 4px 15px rgba(239, 68, 68, 0.4)' : 'none',
                  transition: 'background 0.3s'
                }}
              >
                KONFIRMASI ({multiSelectIndices.length} / {obs.select.maxCount})
              </button>
            </div>
          </div>
        )}

    </div>
  );
}
