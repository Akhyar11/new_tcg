"use client";

import React, { useState, useMemo } from 'react';
import Link from 'next/link';

export interface Card {
  'Card ID': number;
  'Card Name': string;
  [key: string]: any;
}

export interface BattleArenaProps {
  obs: any;
  allCardsData: Card[];
  isSpectator?: boolean;
  onSelectOption?: (optionIndex: number) => void;
  onRestartMatch?: () => void;
  onExitMatch?: () => void;
}

export default function BattleArena({
  obs,
  allCardsData,
  isSpectator = false,
  onSelectOption,
  onRestartMatch,
  onExitMatch,
}: BattleArenaProps) {
  const [previewCard, setPreviewCard] = useState<{ card: Card; energies: Card[] } | null>(null);
  const [discardViewer, setDiscardViewer] = useState<{ cards: any[]; title: string } | null>(null);

  // Helper untuk mengambil data asli kartu berdasarkan ID Engine
  const getCardInfo = (id: number) => {
    const rows = allCardsData.filter((c) => c['Card ID'] === id);
    if (rows.length === 0) return { 'Card ID': id, 'Card Name': `Card #${id}` };
    const base = { ...rows[0] };
    base.moves = rows
      .map((r) => ({
        name: r['Move Name'],
        cost: r['Cost'],
        damage: r['Damage'],
        effect: r['Effect Explanation'],
      }))
      .filter((m) => m.name || m.effect);
    return base;
  };

  // Helper universal untuk menangani Klik Kanan (inspect) kartu
  const handleCardContextMenu = (e: React.MouseEvent, card: any, energies: any[] = []) => {
    e.preventDefault();
    e.stopPropagation();
    if (!card || card.isFacedown) return;

    const cardId = card['Card ID'] || card.engineId || card.id;
    const fullCardInfo = card['Card Name'] ? card : { ...getCardInfo(cardId), ...card };

    setPreviewCard({
      card: fullCardInfo,
      energies: energies || card.energyCards || [],
    });
  };

  // Helper untuk mendapatkan simbol, warna, dan nama tipe Energy berdasarkan data kartu
  const getEnergySymbolAndColor = (typeStr: string = '', nameStr: string = '') => {
    const combined = (typeStr || '') + ' ' + (nameStr || '');
    if (combined.includes('{G}') || combined.toLowerCase().includes('grass')) {
      return { symbol: '🌿', name: 'Grass', color: '#34d399', border: 'rgba(52, 211, 153, 0.8)' };
    }
    if (combined.includes('{L}') || combined.toLowerCase().includes('lightning') || combined.toLowerCase().includes('electric')) {
      return { symbol: '⚡', name: 'Lightning', color: '#fbbf24', border: 'rgba(251, 191, 36, 0.8)' };
    }
    if (combined.includes('{R}') || combined.toLowerCase().includes('fire')) {
      return { symbol: '🔥', name: 'Fire', color: '#f43f5e', border: 'rgba(244, 63, 94, 0.8)' };
    }
    if (combined.includes('{W}') || combined.toLowerCase().includes('water')) {
      return { symbol: '💧', name: 'Water', color: '#38bdf8', border: 'rgba(56, 189, 248, 0.8)' };
    }
    if (combined.includes('{P}') || combined.toLowerCase().includes('psychic')) {
      return { symbol: '🔮', name: 'Psychic', color: '#c084fc', border: 'rgba(192, 132, 252, 0.8)' };
    }
    if (combined.includes('{F}') || combined.toLowerCase().includes('fighting')) {
      return { symbol: '🥊', name: 'Fighting', color: '#fb923c', border: 'rgba(251, 146, 60, 0.8)' };
    }
    if (combined.includes('{D}') || combined.toLowerCase().includes('darkness') || combined.toLowerCase().includes('dark')) {
      return { symbol: '👁️', name: 'Darkness', color: '#94a3b8', border: 'rgba(148, 163, 184, 0.8)' };
    }
    if (combined.includes('{M}') || combined.toLowerCase().includes('metal')) {
      return { symbol: '⚙️', name: 'Metal', color: '#cbd5e1', border: 'rgba(203, 213, 225, 0.8)' };
    }
    return { symbol: '⭐', name: 'Colorless', color: '#e2e8f0', border: 'rgba(226, 232, 240, 0.8)' };
  };

  // Helper untuk merender Badges Energi yang terpasang secara dinamis per tipe
  const renderEnergyBadges = (energyCards: any[], isBench: boolean = false) => {
    if (!energyCards || energyCards.length === 0) return null;

    const groups: { [key: string]: { symbol: string; count: number; color: string; border: string } } = {};

    energyCards.forEach((en) => {
      const typeStr = en['Type'] || '';
      const nameStr = en['Card Name'] || '';
      const info = getEnergySymbolAndColor(typeStr, nameStr);

      if (!groups[info.symbol]) {
        groups[info.symbol] = { symbol: info.symbol, count: 0, color: info.color, border: info.border };
      }
      groups[info.symbol].count += 1;
    });

    const badgeList = Object.values(groups);

    return (
      <div
        style={{
          position: 'absolute',
          top: isBench ? '-6px' : '-8px',
          left: isBench ? '4px' : '8px',
          zIndex: 30,
          display: 'flex',
          gap: '3px',
          flexWrap: 'wrap',
        }}
      >
        {badgeList.map((b, i) => (
          <div
            key={i}
            style={{
              background: 'rgba(15, 23, 42, 0.95)',
              border: `1px solid ${b.border}`,
              borderRadius: isBench ? '10px' : '12px',
              padding: isBench ? '1px 5px' : '2px 7px',
              fontSize: isBench ? '0.6rem' : '0.7rem',
              fontWeight: '800',
              color: b.color,
              display: 'flex',
              alignItems: 'center',
              gap: '3px',
              boxShadow: '0 4px 10px rgba(0,0,0,0.8)',
              backdropFilter: 'blur(8px)',
            }}
          >
            <span>{b.symbol}</span>
            <span>{b.count}</span>
          </div>
        ))}
      </div>
    );
  };

  // Helper untuk merender HP Progress Bar Badge
  const renderHPBadge = (card: any) => {
    if (!card || card.isFacedown || card.hp === undefined || !card.maxHp) return null;
    const currentHp = card.hp;
    const maxHp = card.maxHp;
    const pct = Math.max(0, Math.min(100, (currentHp / maxHp) * 100));
    const barColor = pct > 50 ? '#34d399' : pct > 20 ? '#fbbf24' : '#f43f5e';
    return (
      <div
        style={{
          position: 'absolute',
          bottom: '4px',
          left: '4px',
          right: '4px',
          background: 'rgba(15, 23, 42, 0.9)',
          borderRadius: '6px',
          padding: '3px 6px',
          border: '1px solid rgba(255,255,255,0.15)',
          zIndex: 30,
          display: 'flex',
          flexDirection: 'column',
          gap: '2px',
          backdropFilter: 'blur(4px)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', fontWeight: 'bold', color: 'white' }}>
          <span>HP</span>
          <span style={{ color: barColor }}>
            {currentHp}/{maxHp}
          </span>
        </div>
        <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.2)', borderRadius: '2px', overflow: 'hidden' }}>
          <div style={{ width: `${pct}%`, height: '100%', background: barColor, transition: 'width 0.3s' }} />
        </div>
      </div>
    );
  };

  // Helper untuk merender slot Prize Cards (tampilan berkurang saat diambil)
  const renderPrizeSlot = (prizeList: any[], index: number, isPlayerArea: boolean = false) => {
    const hasCard = index < prizeList.length && prizeList[index] !== undefined && prizeList[index] !== 'empty';
    const prizeOptIdx =
      !isSpectator && isPlayerArea && obs?.select?.option
        ? obs.select.option.findIndex((opt: any) => opt.type === 3 && opt.area === 6 && opt.index === index)
        : -1;
    const isSelectable = prizeOptIdx !== -1;

    return (
      <div
        key={index}
        onClick={() => {
          if (isSelectable && onSelectOption) onSelectOption(prizeOptIdx);
        }}
        style={{
          width: '60px',
          height: '84px',
          border: isSelectable ? '2px solid #ef4444' : '1px dashed rgba(255,255,255,0.1)',
          borderRadius: '4px',
          position: 'relative',
          cursor: isSelectable ? 'pointer' : 'default',
          boxShadow: isSelectable ? '0 0 12px rgba(239, 68, 68, 0.8)' : 'none',
          background: 'rgba(0,0,0,0.25)',
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

  // State derivations dari C++ Engine obs
  let playerHand: any[] = [];
  let playerActive: any = null;
  let playerBench: any[] = [null, null, null, null, null];
  let playerDiscard: any[] = [];
  let playerDeckCount: number = 60;
  let playerPrizeCards: any[] = [...Array(6)];

  let aiHand: any[] = [];
  let aiHandCountVal: number = 0;
  let aiActive: any = null;
  let aiBench: any[] = [null, null, null, null, null];
  let aiDiscard: any[] = [];
  let aiDeckCountVal: number = 60;
  let aiPrizeCards: any[] = [...Array(6)];

  let stadiumCard: any = null;

  const deckOrDiscardOptions = useMemo(() => {
    if (!obs?.select?.option) return [];
    return obs.select.option
      .map((opt: any, originalIdx: number) => ({ ...opt, originalIdx }))
      .filter((opt: any) => opt.type === 3 && (opt.area === 1 || opt.area === 3));
  }, [obs?.select]);

  if (obs && obs.current && obs.current.players) {
    let p0: any;
    let p1: any;

    if (isSpectator) {
      p0 = obs.current.players[0];
      p1 = obs.current.players[1];
    } else {
      p0 =
        obs.current.players[0]?.hand && Array.isArray(obs.current.players[0].hand)
          ? obs.current.players[0]
          : obs.current.players[1]?.hand && Array.isArray(obs.current.players[1].hand)
          ? obs.current.players[1]
          : obs.current.players[0];

      p1 = p0 === obs.current.players[0] ? obs.current.players[1] : obs.current.players[0];
    }

    // Player 0 (Anda)
    if (p0.deckCount !== undefined && p0.deckCount !== null) playerDeckCount = p0.deckCount;
    else if (p0.deck_count !== undefined && p0.deck_count !== null) playerDeckCount = p0.deck_count;
    else if (Array.isArray(p0.deck)) playerDeckCount = p0.deck.length;

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
          energyCards: p0.active[0].energyCards ? p0.active[0].energyCards.map((ec: any) => getCardInfo(ec.id)).filter(Boolean) : [],
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
            energyCards: b.energyCards ? b.energyCards.map((ec: any) => getCardInfo(ec.id)).filter(Boolean) : [],
          };
        }
      });
    }
    if (p0.discard) {
      playerDiscard = p0.discard.map((c: any) => ({ ...getCardInfo(c.id), engineSerial: c.serial, engineId: c.id }));
    }

    // Player 1 (Opponent / AI)
    if (p1.deckCount !== undefined && p1.deckCount !== null) aiDeckCountVal = p1.deckCount;
    else if (p1.deck_count !== undefined && p1.deck_count !== null) aiDeckCountVal = p1.deck_count;
    else if (Array.isArray(p1.deck)) aiDeckCountVal = p1.deck.length;

    if (p1.handCount !== undefined) aiHandCountVal = p1.handCount;
    else if (p1.hand) {
      if (Array.isArray(p1.hand)) {
        aiHandCountVal = p1.hand.length;
        aiHand = p1.hand.map((c: any) => ({ ...getCardInfo(c.id), engineSerial: c.serial, engineId: c.id }));
      } else if (typeof p1.hand === 'number') {
        aiHandCountVal = p1.hand;
      }
    }

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
          energyCards: p1.active[0].energyCards ? p1.active[0].energyCards.map((ec: any) => getCardInfo(ec.id)).filter(Boolean) : [],
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
            energyCards: b.energyCards ? b.energyCards.map((ec: any) => getCardInfo(ec.id)).filter(Boolean) : [],
          };
        }
      });
    }
    if (p1.discard) {
      aiDiscard = p1.discard.map((c: any) => ({ ...getCardInfo(c.id), engineSerial: c.serial, engineId: c.id }));
    }

    // Stadium Card (Global Field)
    if (p0.stadium && p0.stadium.length > 0 && p0.stadium[0]) {
      stadiumCard = { ...getCardInfo(p0.stadium[0].id), engineSerial: p0.stadium[0].serial, engineId: p0.stadium[0].id };
    } else if (p1.stadium && p1.stadium.length > 0 && p1.stadium[0]) {
      stadiumCard = { ...getCardInfo(p1.stadium[0].id), engineSerial: p1.stadium[0].serial, engineId: p1.stadium[0].id };
    }
  }

  const isPlayerTurn = obs?.current?.yourIndex === 0;

  // Handle Drag & Drop
  const handleDrop = (e: React.DragEvent, targetArea: number | 'generic', targetIndex: number) => {
    e.preventDefault();
    e.stopPropagation();
    if (isSpectator || !obs?.select || !obs.select.option || !onSelectOption) return;

    try {
      const data = JSON.parse(e.dataTransfer.getData('text/plain'));
      const sourceArea = data.area;
      const sourceIndex = data.index;

      const options = obs.select.option;
      let matchIdx = -1;

      if (obs.select.type === 1) {
        matchIdx = options.findIndex((opt: any) => opt.type === 3 && opt.area === sourceArea && opt.index === sourceIndex);
      } else if (obs.select.type === 0) {
        if (targetArea === 'generic') {
          matchIdx = options.findIndex((opt: any) => opt.type === 7 && opt.index === sourceIndex);
        } else {
          matchIdx = options.findIndex(
            (opt: any) =>
              (opt.type === 8 || opt.type === 9) &&
              opt.area === sourceArea &&
              opt.index === sourceIndex &&
              opt.inPlayArea === targetArea &&
              opt.inPlayIndex === targetIndex
          );
          if (matchIdx === -1 && targetArea === 5) {
            matchIdx = options.findIndex((opt: any) => opt.type === 7 && opt.index === sourceIndex);
          }
          if (matchIdx === -1) {
            matchIdx = options.findIndex((opt: any) => opt.type === 7 && opt.index === sourceIndex);
          }
        }
      }

      if (matchIdx !== -1) {
        onSelectOption(matchIdx);
      }
    } catch (err) {
      console.error("Drop error", err);
    }
  };

  return (
    <div style={{ height: '100vh', maxHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#050b14', color: 'white', fontFamily: '"Inter", sans-serif', overflow: 'hidden', position: 'relative' }}>

      {/* GAME OVER MODAL OVERLAY */}
      {obs?.current?.result !== undefined && obs.current.result !== -1 && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(5, 11, 20, 0.92)', backdropFilter: 'blur(16px)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', animation: 'fadeIn 0.4s ease-out' }}>
          <div style={{ background: obs.current.result === 0 ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(6, 78, 59, 0.5))' : obs.current.result === 1 ? 'linear-gradient(135deg, rgba(244, 63, 94, 0.25), rgba(136, 19, 55, 0.5))' : 'linear-gradient(135deg, rgba(245, 158, 11, 0.25), rgba(120, 53, 15, 0.5))', border: `2px solid ${obs.current.result === 0 ? '#10b981' : obs.current.result === 1 ? '#f43f5e' : '#f59e0b'}`, borderRadius: '24px', padding: '3rem 3.5rem', textAlign: 'center', maxWidth: '540px', boxShadow: `0 25px 60px ${obs.current.result === 0 ? 'rgba(16, 185, 129, 0.4)' : obs.current.result === 1 ? 'rgba(244, 63, 94, 0.4)' : 'rgba(245, 158, 11, 0.4)'}`, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.5rem' }}>
            <div style={{ fontSize: '4.5rem', filter: 'drop-shadow(0 0 20px currentColor)' }}>
              {obs.current.result === 0 ? '🏆' : obs.current.result === 1 ? '💀' : '⚖️'}
            </div>

            <div>
              <h2 style={{ fontSize: '2.5rem', fontWeight: '900', margin: 0, letterSpacing: '2px', color: obs.current.result === 0 ? '#34d399' : obs.current.result === 1 ? '#fb7185' : '#fbbf24' }}>
                {isSpectator
                  ? 'PERTANDINGAN SELESAI!'
                  : obs.current.result === 0
                  ? 'KEMENANGAN!'
                  : obs.current.result === 1
                  ? 'KEKALAHAN!'
                  : 'HASIL SERI!'}
              </h2>
              <p style={{ color: '#94a3b8', margin: '0.6rem 0 0 0', fontSize: '1rem', lineHeight: '1.5' }}>
                {isSpectator
                  ? obs.current.result === 0
                    ? 'Player 0 (AI 1) memenangkan pertempuran!'
                    : obs.current.result === 1
                    ? 'Player 1 (AI 2) memenangkan pertempuran!'
                    : 'Pertandingan berakhir seri.'
                  : obs.current.result === 0
                  ? 'Selamat! Anda berhasil mengalahkan JAX AI dalam pertempuran ini.'
                  : obs.current.result === 1
                  ? 'JAX AI memenangkan pertempuran (Prize habis / Deck out).'
                  : 'Pertempuran berakhir seimbang.'}
              </p>
            </div>

            <div style={{ display: 'flex', gap: '1rem', width: '100%', marginTop: '1rem' }}>
              {onRestartMatch && (
                <button onClick={onRestartMatch} style={{ flex: 1, padding: '0.9rem 1.5rem', background: 'linear-gradient(135deg, #0ea5e9, #2563eb)', color: 'white', border: 'none', borderRadius: '12px', fontWeight: 'bold', fontSize: '1rem', cursor: 'pointer', boxShadow: '0 4px 15px rgba(14, 165, 233, 0.4)' }}>
                  🔄 Main Lagi
                </button>
              )}
              {onExitMatch ? (
                <button onClick={onExitMatch} style={{ flex: 1, padding: '0.9rem 1.5rem', background: 'rgba(255, 255, 255, 0.08)', color: 'white', border: '1px solid rgba(255, 255, 255, 0.2)', borderRadius: '12px', fontWeight: 'bold', fontSize: '1rem', cursor: 'pointer' }}>
                  🏠 Menu Utama
                </button>
              ) : (
                <Link href="/" style={{ flex: 1, padding: '0.9rem 1.5rem', background: 'rgba(255, 255, 255, 0.08)', color: 'white', border: '1px solid rgba(255, 255, 255, 0.2)', borderRadius: '12px', fontWeight: 'bold', fontSize: '1rem', textDecoration: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  🏠 Menu Utama
                </Link>
              )}
            </div>
          </div>
        </div>
      )}

      {/* TURN STATUS BANNER HUD */}
      {obs && obs.current?.result === -1 && (
        <div style={{ position: 'absolute', top: '0.3rem', left: '50%', transform: 'translateX(-50%)', zIndex: 200, display: 'flex', alignItems: 'center', gap: '0.6rem', background: isPlayerTurn ? 'rgba(56, 189, 248, 0.15)' : 'rgba(244, 63, 94, 0.15)', border: `1px solid ${isPlayerTurn ? 'rgba(56, 189, 248, 0.6)' : 'rgba(244, 63, 94, 0.6)'}`, borderRadius: '30px', padding: '0.3rem 1.2rem', backdropFilter: 'blur(8px)', boxShadow: isPlayerTurn ? '0 0 15px rgba(56, 189, 248, 0.25)' : '0 0 15px rgba(244, 63, 94, 0.25)' }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: isPlayerTurn ? '#38bdf8' : '#f43f5e', boxShadow: `0 0 8px ${isPlayerTurn ? '#38bdf8' : '#f43f5e'}` }} />
          <span style={{ fontWeight: '900', fontSize: '0.8rem', letterSpacing: '1.2px', color: isPlayerTurn ? '#38bdf8' : '#f43f5e' }}>
            {isSpectator
              ? isPlayerTurn
                ? 'GILIRAN PLAYER 0 (AI 1)'
                : 'GILIRAN PLAYER 1 (AI 2)'
              : isPlayerTurn
              ? 'GILIRAN ANDA (PLAYER)'
              : 'GILIRAN LAWAN (JAX AI)'}
          </span>
        </div>
      )}

      {/* TOP ROW: OPPONENT HAND */}
      <div style={{ paddingTop: '2.2rem', paddingBottom: '0.3rem', paddingLeft: '1rem', paddingRight: '1rem', display: 'flex', justifyContent: 'center', gap: '5px', height: '95px', flexShrink: 0 }}>
        {isSpectator && Array.isArray(aiHand) && aiHand.length > 0 ? (
          aiHand.map((card, i) => (
            <div
              key={i}
              onContextMenu={(e) => handleCardContextMenu(e, card)}
              style={{ width: '55px', height: '77px', transition: 'transform 0.2s', position: 'relative' }}
              onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(10px) scale(1.1)'; e.currentTarget.style.zIndex = '100'; }}
              onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0) scale(1)'; e.currentTarget.style.zIndex = '1'; }}
            >
              <img src={`/assets/cards/${card['Card ID'] || card.engineId}.png`} style={{ width: '100%', height: '100%', borderRadius: '4px', boxShadow: '0 5px 15px rgba(0,0,0,0.5)', objectFit: 'contain' }} alt={card['Card Name']} />
            </div>
          ))
        ) : (
          [...Array(aiHandCountVal)].map((_, i) => (
            <div key={i} style={{ width: '55px', height: '77px' }}>
              <img src="/assets/cards/back.png" style={{ width: '100%', height: '100%', borderRadius: '4px' }} alt="Card Back" />
            </div>
          ))
        )}
      </div>

      {/* MIDDLE SECTION: PLAYMAT */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', padding: '0.4rem 1.5rem', gap: '0.4rem', position: 'relative', overflow: 'hidden' }}>

        {/* ================= OPPONENT HALF ================= */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>

          {/* Left: Prize Cards */}
          <div style={{ width: '160px', display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}>
            {[...Array(6)].map((_, i) => renderPrizeSlot(aiPrizeCards, i, false))}
          </div>

          {/* Center: Active & Bench */}
          <div style={{ flex: 1, display: 'flex', justifyContent: 'center', gap: '1.5rem', alignItems: 'center' }}>
            {/* Active */}
            <div
              onContextMenu={(e) => handleCardContextMenu(e, aiActive, aiActive?.energyCards)}
              style={{ position: 'relative', width: '115px', height: '160px', border: '2px solid rgba(255,255,255,0.1)', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: aiActive && !aiActive.isFacedown ? 'pointer' : 'default' }}
            >
              {aiActive && !aiActive.isFacedown ? (
                <>
                  <img src={`/assets/cards/${aiActive['Card ID'] || aiActive.engineId}.png`} style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '8px', zIndex: 10, position: 'relative' }} onContextMenu={(e) => handleCardContextMenu(e, aiActive, aiActive.energyCards)} />
                  {/* Dynamic Energy Badges HUD */}
                  {renderEnergyBadges(aiActive.energyCards)}
                  {renderHPBadge(aiActive)}
                </>
              ) : aiActive && aiActive.isFacedown ? (
                <img src="/assets/cards/back.png" style={{ width: '100%', height: '100%', borderRadius: '8px' }} />
              ) : null}
            </div>

            {/* Bench Row */}
            <div style={{ display: 'flex', gap: '8px' }}>
              {[...Array(5)].map((_, i) => (
                <div
                  key={i}
                  onContextMenu={(e) => handleCardContextMenu(e, aiBench[i], aiBench[i]?.energyCards)}
                  style={{ position: 'relative', width: '75px', height: '105px', border: '2px dashed rgba(255,255,255,0.1)', borderRadius: '6px', cursor: aiBench[i] && !aiBench[i].isFacedown ? 'pointer' : 'default' }}
                >
                  {aiBench[i] && !aiBench[i].isFacedown ? (
                    <>
                      <img src={`/assets/cards/${aiBench[i]['Card ID'] || aiBench[i].engineId}.png`} style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '6px', zIndex: 10, position: 'relative' }} onContextMenu={(e) => handleCardContextMenu(e, aiBench[i], aiBench[i].energyCards)} />
                      {/* Dynamic Energy Badges HUD */}
                      {renderEnergyBadges(aiBench[i].energyCards, true)}
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
          <div style={{ width: '160px', display: 'flex', flexDirection: 'column', gap: '10px', alignItems: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: '#888' }}>Hand [{aiHandCountVal}]</div>
            <div
              onClick={() => setDiscardViewer({ cards: aiDiscard, title: isSpectator ? 'Discard Player 1' : 'Discard Lawan' })}
              style={{ position: 'relative', width: '65px', height: '91px', border: '2px dashed #444', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}
            >
              {aiDiscard.length > 0 ? (
                <img src={`/assets/cards/${aiDiscard[aiDiscard.length - 1]['Card ID'] || aiDiscard[aiDiscard.length - 1].engineId}.png`} style={{ width: '100%', height: '100%', borderRadius: '4px', objectFit: 'contain' }} />
              ) : (
                <span style={{ color: '#666', fontSize: '0.7rem' }}>Discard</span>
              )}
            </div>
            <div style={{ position: 'relative', width: '65px', height: '91px' }}>
              <img src="/assets/cards/back.png" style={{ width: '100%', height: '100%', borderRadius: '4px' }} />
              <div style={{ position: 'absolute', top: '-18px', width: '100%', textAlign: 'center', fontSize: '0.75rem', color: '#888' }}>Deck [{aiDeckCountVal}]</div>
            </div>
          </div>
        </div>

        {/* ACTION PANEL HUD (For Player vs AI mode) */}
        {!isSpectator && obs?.select && obs.current?.yourIndex === 0 && (
          <div style={{ position: 'absolute', left: '220px', top: '50%', transform: 'translateY(-50%)', background: 'rgba(15, 23, 42, 0.95)', border: '1px solid rgba(56, 189, 248, 0.4)', borderRadius: '16px', padding: '1rem', zIndex: 100, width: '260px', backdropFilter: 'blur(12px)', boxShadow: '0 15px 35px rgba(0,0,0,0.6)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.8rem', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '0.4rem' }}>
              <h3 style={{ margin: 0, color: '#38bdf8', fontSize: '0.95rem', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <span>⚡</span> Panel Aksi
              </h3>
              <span style={{ fontSize: '0.65rem', background: 'rgba(255,255,255,0.06)', padding: '0.2rem 0.5rem', borderRadius: '10px', color: '#94a3b8' }}>Ctx: {obs.select.context}</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '35vh', overflowY: 'auto' }}>
              {obs.select.minCount === 0 && onSelectOption && (
                <button onClick={() => onSelectOption(-1)} style={{ background: 'linear-gradient(135deg, #0ea5e9, #2563eb)', border: 'none', borderRadius: '8px', padding: '0.65rem', color: 'white', textAlign: 'center', cursor: 'pointer', fontSize: '0.85rem', fontWeight: 'bold', boxShadow: '0 4px 12px rgba(14, 165, 233, 0.3)' }}>SELESAI / LANJUT →</button>
              )}
              {obs.select.option.map((opt: any, idx: number) => {
                if (opt.type === 7 || opt.type === 8 || opt.type === 9 || opt.type === 13 || (opt.type === 3 && opt.area === 2)) return null;
                if (opt.type === 3 && (opt.area === 6 || opt.area === 1 || opt.area === 3)) return null;

                let label = `Aksi ${idx}`;
                let bgStyle = 'linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02))';
                let borderStyle = '1px solid rgba(255,255,255,0.1)';
                let textColor = 'white';

                if (opt.type === 1) { label = '✓ YES'; bgStyle = 'linear-gradient(135deg, #10b981, #059669)'; borderStyle = 'none'; }
                else if (opt.type === 2) { label = '✕ NO'; bgStyle = 'linear-gradient(135deg, #ef4444, #dc2626)'; borderStyle = 'none'; }
                else if (opt.type === 14) { label = '⏹ END TURN'; bgStyle = 'linear-gradient(135deg, #f43f5e, #be123c)'; borderStyle = 'none'; }
                else if (opt.type === 12) { label = '🔄 RETREAT'; bgStyle = 'linear-gradient(135deg, #f59e0b, #d97706)'; borderStyle = 'none'; }
                else if (opt.type === 10) { label = `🔮 USE ABILITY (Area ${opt.area} Idx ${opt.index})`; bgStyle = 'linear-gradient(135deg, #8b5cf6, #6d28d9)'; borderStyle = 'none'; }
                else if (opt.type === 13) { label = `⚔️ ATTACK ${opt.attackId}`; bgStyle = 'linear-gradient(135deg, #dc2626, #991b1b)'; borderStyle = 'none'; }
                else if (opt.type === 3) { label = `🎴 SELECT CARD (Area ${opt.area} Idx ${opt.index})`; }

                return (
                  <button key={idx} onClick={() => onSelectOption && onSelectOption(idx)} style={{ background: bgStyle, border: borderStyle, borderRadius: '8px', padding: '0.6rem 0.8rem', color: textColor, textAlign: 'left', cursor: 'pointer', fontSize: '0.8rem', fontWeight: '600', transition: 'all 0.2s', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span>{label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* CENTRAL ARENA DIVIDER WITH PROMINENT STADIUM SLOT */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 30px', position: 'relative', margin: '0.1rem 0' }}>
          {/* Left: Dedicated Stadium Slot */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', width: '200px' }}>
            <div
              onClick={() => { if (stadiumCard) setPreviewCard({ card: stadiumCard, energies: [] }); }}
              onContextMenu={(e) => handleCardContextMenu(e, stadiumCard)}
              style={{
                width: '75px',
                height: '105px',
                border: stadiumCard ? '2px solid #34d399' : '2px dashed rgba(52, 211, 153, 0.6)',
                borderRadius: '8px',
                background: stadiumCard ? 'rgba(52, 211, 153, 0.15)' : 'rgba(15, 23, 42, 0.85)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative',
                cursor: stadiumCard ? 'pointer' : 'default',
                boxShadow: stadiumCard ? '0 0 25px rgba(52, 211, 153, 0.5)' : '0 0 15px rgba(52, 211, 153, 0.15)',
                transition: 'all 0.3s ease',
                zIndex: 50,
              }}
            >
              {stadiumCard ? (
                <>
                  <img
                    src={`/assets/cards/${stadiumCard['Card ID'] || stadiumCard.engineId}.png`}
                    style={{ width: '100%', height: '100%', borderRadius: '6px', objectFit: 'contain' }}
                    alt={stadiumCard['Card Name'] || 'Stadium'}
                  />
                  <div style={{ position: 'absolute', bottom: '-7px', background: '#34d399', color: '#0f172a', padding: '0.1rem 0.4rem', borderRadius: '10px', fontSize: '0.6rem', fontWeight: '900', letterSpacing: '0.5px' }}>
                    STADIUM
                  </div>
                </>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '3px', color: 'rgba(52, 211, 153, 0.8)' }}>
                  <span style={{ fontSize: '1.2rem' }}>🏟️</span>
                  <span style={{ fontSize: '0.65rem', fontWeight: '900', letterSpacing: '1px', color: '#34d399' }}>STADIUM</span>
                </div>
              )}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
              <span style={{ fontSize: '0.7rem', color: '#34d399', fontWeight: '900', letterSpacing: '0.5px' }}>ARENA STADIUM</span>
              <span style={{ fontSize: '0.6rem', color: '#64748b' }}>{stadiumCard ? stadiumCard['Card Name'] : 'Tidak Ada Stadium'}</span>
            </div>
          </div>

          {/* Divider Line */}
          <div style={{ flex: 1, height: '1px', background: 'rgba(255,255,255,0.08)', marginLeft: '1rem' }} />
        </div>

        {/* ================= PLAYER HALF ================= */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }} onDragOver={(e) => e.preventDefault()} onDrop={(e) => handleDrop(e, 'generic', 0)}>

          {/* Left: Prize Cards */}
          <div style={{ width: '160px', display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}>
            {[...Array(6)].map((_, i) => renderPrizeSlot(playerPrizeCards, i, true))}
          </div>

          {/* Center: Active & Bench */}
          <div style={{ flex: 1, display: 'flex', justifyContent: 'center', gap: '1.5rem', alignItems: 'center' }}>
            {/* Active */}
            <div
              style={{ position: 'relative', width: '115px', height: '160px', border: '2px solid rgba(255,255,255,0.1)', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: playerActive && !playerActive.isFacedown ? 'pointer' : 'default' }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => handleDrop(e, 4, 0)}
              onContextMenu={(e) => handleCardContextMenu(e, playerActive, playerActive?.energyCards)}
            >
              {playerActive && !playerActive.isFacedown ? (
                <>
                  <img src={`/assets/cards/${playerActive['Card ID'] || playerActive.engineId}.png`} style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '8px', zIndex: 10, position: 'relative' }} onContextMenu={(e) => handleCardContextMenu(e, playerActive, playerActive?.energyCards)} />
                  {/* Dynamic Energy Badges HUD */}
                  {renderEnergyBadges(playerActive?.energyCards)}
                  {renderHPBadge(playerActive)}
                </>
              ) : playerActive && playerActive.isFacedown ? (
                <img src="/assets/cards/back.png" style={{ width: '100%', height: '100%', borderRadius: '8px' }} />
              ) : null}
            </div>

            {/* Bench Row */}
            <div style={{ display: 'flex', gap: '8px' }}>
              {playerBench.map((benchCard, i) => (
                <div
                  key={i}
                  style={{ position: 'relative', width: '75px', height: '105px', border: '2px dashed rgba(255,255,255,0.1)', borderRadius: '6px', cursor: benchCard && !benchCard.isFacedown ? 'pointer' : 'default' }}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => handleDrop(e, 5, i)}
                  onContextMenu={(e) => handleCardContextMenu(e, benchCard, benchCard?.energyCards)}
                >
                  {benchCard && !benchCard.isFacedown ? (
                    <>
                      <img src={`/assets/cards/${benchCard['Card ID'] || benchCard.engineId}.png`} style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '6px', zIndex: 10, position: 'relative' }} onContextMenu={(e) => handleCardContextMenu(e, benchCard, benchCard?.energyCards)} />
                      {/* Dynamic Energy Badges HUD */}
                      {renderEnergyBadges(benchCard?.energyCards, true)}
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
          <div style={{ width: '160px', display: 'flex', flexDirection: 'column', gap: '10px', alignItems: 'center' }}>
            <div style={{ position: 'relative', width: '65px', height: '91px' }}>
              <img src="/assets/cards/back.png" style={{ width: '100%', height: '100%', borderRadius: '4px' }} />
              <div style={{ position: 'absolute', top: '-18px', width: '100%', textAlign: 'center', fontSize: '0.75rem', color: '#888' }}>Deck [{playerDeckCount}]</div>
            </div>
            <div
              onClick={() => setDiscardViewer({ cards: playerDiscard, title: isSpectator ? 'Discard Player 0' : 'Discard Anda' })}
              style={{ position: 'relative', width: '65px', height: '91px', border: '2px dashed #444', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}
            >
              {playerDiscard.length > 0 ? (
                <img src={`/assets/cards/${playerDiscard[playerDiscard.length - 1]['Card ID'] || playerDiscard[playerDiscard.length - 1].engineId}.png`} style={{ width: '100%', height: '100%', borderRadius: '4px', objectFit: 'contain' }} />
              ) : (
                <span style={{ color: '#666', fontSize: '0.7rem' }}>Discard</span>
              )}
            </div>
            <div style={{ fontSize: '0.75rem', color: '#888' }}>Hand [{playerHand.length}]</div>
          </div>

        </div>
      </div>

      {/* BOTTOM ROW: PLAYER HAND */}
      <div style={{ padding: '0.4rem 0.5rem', display: 'flex', justifyContent: 'center', gap: '5px', height: '125px', flexShrink: 0, background: 'rgba(0,0,0,0.6)', position: 'relative', zIndex: 100, overflow: 'visible' }}>
        {playerHand.map((card, i) => (
          <div
            key={i}
            draggable={!isSpectator}
            onDragStart={(e) => {
              if (!isSpectator) e.dataTransfer.setData('text/plain', JSON.stringify({ area: 2, index: i }));
            }}
            onContextMenu={(e) => handleCardContextMenu(e, card)}
            style={{ width: '75px', height: '105px', cursor: 'grab', transition: 'transform 0.2s', position: 'relative' }}
            onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-15px) scale(1.15)'; e.currentTarget.style.zIndex = '100'; }}
            onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0) scale(1)'; e.currentTarget.style.zIndex = '1'; }}
          >
            <img src={`/assets/cards/${card['Card ID'] || card.engineId}.png`} style={{ width: '100%', height: '100%', borderRadius: '6px', boxShadow: '0 5px 15px rgba(0,0,0,0.5)', objectFit: 'contain' }} alt={card['Card Name']} />
          </div>
        ))}
      </div>

      {/* CARD PREVIEW INSPECTOR MODAL */}
      {previewCard && (
        <div onClick={() => setPreviewCard(null)} style={{ position: 'fixed', inset: 0, zIndex: 1000, background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(5px)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'zoom-out' }}>
          <div style={{ display: 'flex', gap: '2rem', alignItems: 'flex-start', maxWidth: '1000px', width: '100%', padding: '2rem' }} onClick={(e) => e.stopPropagation()}>
            <div style={{ width: '350px', flexShrink: 0, boxShadow: '0 0 50px rgba(255,255,255,0.2)', borderRadius: '16px', position: 'relative' }}>
              <img src={`/assets/cards/${previewCard.card['Card ID'] || previewCard.card.engineId}.png`} style={{ width: '100%', height: 'auto', borderRadius: '16px' }} />
            </div>

            <div style={{ flex: 1, background: 'rgba(30, 41, 59, 0.9)', padding: '2rem', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.1)', position: 'relative' }}>
              <button onClick={() => setPreviewCard(null)} style={{ position: 'absolute', top: '1rem', right: '1rem', background: 'transparent', border: 'none', color: '#94a3b8', fontSize: '1.5rem', cursor: 'pointer' }}>✕</button>
              <h2 style={{ margin: '0 0 0.5rem 0', fontSize: '2rem', color: '#38bdf8', paddingRight: '2rem' }}>{previewCard.card['Card Name']}</h2>
              <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', color: '#94a3b8' }}>
                <span style={{ background: '#334155', padding: '0.3rem 0.8rem', borderRadius: '20px', fontSize: '0.9rem' }}>{previewCard.card['Stage (Pokémon)/Type (Energy and Trainer)']}</span>
                {previewCard.card['HP'] && <span style={{ background: '#ef4444', color: 'white', padding: '0.3rem 0.8rem', borderRadius: '20px', fontSize: '0.9rem', fontWeight: 'bold' }}>HP {previewCard.card['HP']}</span>}
                {previewCard.card['Type'] && <span style={{ background: '#eab308', color: 'black', padding: '0.3rem 0.8rem', borderRadius: '20px', fontSize: '0.9rem', fontWeight: 'bold' }}>{previewCard.card['Type']}</span>}
              </div>

              {!isSpectator && (
                <>
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
                          {playOptIdx !== -1 && onSelectOption && (
                            <button onClick={() => { onSelectOption(playOptIdx); setPreviewCard(null); }} style={{ flex: 1, padding: '0.8rem', background: '#3b82f6', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', fontSize: '1rem', boxShadow: '0 4px 15px rgba(59, 130, 246, 0.4)' }}>GUNAKAN KARTU</button>
                          )}
                          {abilityOptIdx !== -1 && onSelectOption && (
                            <button onClick={() => { onSelectOption(abilityOptIdx); setPreviewCard(null); }} style={{ flex: 1, padding: '0.8rem', background: '#8b5cf6', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', fontSize: '1rem', boxShadow: '0 4px 15px rgba(139, 92, 246, 0.4)' }}>GUNAKAN ABILITY</button>
                          )}
                        </div>
                      );
                    }
                    return null;
                  })()}
                </>
              )}

              {/* Attached Energies Info */}
              {previewCard.energies && previewCard.energies.length > 0 && (
                <div style={{ marginBottom: '1.5rem', background: 'linear-gradient(to right, rgba(245, 158, 11, 0.1), transparent)', borderLeft: '4px solid #f59e0b', padding: '0.8rem 1rem', borderRadius: '4px' }}>
                  <div style={{ fontSize: '0.9rem', color: '#f59e0b', fontWeight: 'bold', marginBottom: '0.5rem' }}>Attached Energy: {previewCard.energies.length}</div>
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    {previewCard.energies.map((en, idx) => (
                      <div key={idx} style={{ background: 'rgba(0,0,0,0.4)', padding: '0.3rem 0.6rem', borderRadius: '6px', fontSize: '0.85rem', color: '#cbd5e1', border: '1px solid rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <img src={`/assets/cards/${en['Card ID'] || en.engineId}.png`} style={{ width: '20px', height: '28px', borderRadius: '2px', objectFit: 'cover' }} />
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
                    if (!isSpectator && !isAbility && playerActive && previewCard.card.engineSerial === playerActive.engineSerial) {
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
                        {btnOptIdx !== -1 && onSelectOption && (
                          <button onClick={() => { onSelectOption(btnOptIdx); setPreviewCard(null); }} style={{ marginTop: '1rem', width: '100%', padding: '0.8rem', background: '#ef4444', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', fontSize: '1rem', boxShadow: '0 4px 15px rgba(239, 68, 68, 0.4)' }}>GUNAKAN SERANGAN</button>
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
                src={`/assets/cards/${card['Card ID'] || card.engineId}.png`}
                style={{ width: '120px', height: '168px', borderRadius: '8px', cursor: 'pointer', transition: 'transform 0.2s' }}
                onMouseEnter={(e) => (e.currentTarget.style.transform = 'scale(1.1)')}
                onMouseLeave={(e) => (e.currentTarget.style.transform = 'scale(1)')}
                onClick={() => setPreviewCard({ card: card, energies: [] })}
                onContextMenu={(e) => handleCardContextMenu(e, card)}
              />
            ))}
            {discardViewer.cards.length === 0 && <p style={{ color: '#94a3b8' }}>Discard pile kosong.</p>}
          </div>
          <button onClick={() => setDiscardViewer(null)} style={{ marginTop: '2rem', padding: '0.8rem 2rem', background: '#334155', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer' }}>Tutup</button>
        </div>
      )}
    </div>
  );
}
