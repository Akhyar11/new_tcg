"use client";

import React, { useEffect, useState, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import BattleArena, { Card } from '@/components/BattleArena';

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
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
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

        setAvailableDecks(data.decks);

        const cardRes = await fetch('/cards.json');
        const allCards = (await cardRes.json()) as Card[];
        setAllCardsData(allCards);

        setLoading(false);
      } catch (e) {
        setError('Gagal memuat data game.');
        setLoading(false);
      }
    };

    initializeGame();

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, [router]);

  const startGameWithDeck = (selectedDeck: any) => {
    try {
      const parsedIds = JSON.parse(selectedDeck.cards);
      const loadedCards = parsedIds.map((id: number) => allCardsData.find((c) => c['Card ID'] === id)).filter(Boolean);

      setDeck(loadedCards);

      if (wsRef.current) wsRef.current.close();

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

      wsRef.current = socket;
      setGameState('PLAYING');
    } catch (e) {
      setError('Gagal memuat kartu di deck ini.');
    }
  };

  const sendSelect = (index: number) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      if (index === -1) {
        wsRef.current.send(JSON.stringify({ type: 'select', options: [] }));
      } else {
        wsRef.current.send(JSON.stringify({ type: 'select', options: [index] }));
      }
    }
  };

  const handleRestart = () => {
    if (wsRef.current) wsRef.current.close();
    setObs(null);
    setGameState('SELECT_DECK');
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

  if (gameState === 'SELECT_DECK') {
    return (
      <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: '#0f172a', color: 'white', fontFamily: '"Inter", sans-serif', padding: '4rem 2rem', alignItems: 'center' }}>
        <Link href="/" style={{ position: 'absolute', top: '2rem', left: '2rem', color: '#94a3b8', textDecoration: 'none', fontWeight: 'bold' }}>← Kembali</Link>

        <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem', background: 'linear-gradient(to right, #38bdf8, #818cf8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Pilih Deck Anda
        </h1>
        <p style={{ color: '#94a3b8', marginBottom: '3rem' }}>Pilih salah satu deck yang telah Anda rakit untuk menghadapi JAX AI.</p>

        <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap', justifyContent: 'center', maxWidth: '1000px' }}>
          {availableDecks.map((d) => {
            const parsedIds = JSON.parse(d.cards);
            const cardCount = parsedIds.length;
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
                  boxShadow: '0 10px 30px -10px rgba(0,0,0,0.5)',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-8px) scale(1.02)';
                  e.currentTarget.style.borderColor = 'rgba(56, 189, 248, 0.6)';
                  e.currentTarget.style.boxShadow = '0 20px 40px -10px rgba(56, 189, 248, 0.3)';
                  const img = e.currentTarget.querySelector('.deck-bg') as HTMLElement;
                  if (img) img.style.transform = 'scale(1.1)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0) scale(1)';
                  e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)';
                  e.currentTarget.style.boxShadow = '0 10px 30px -10px rgba(0,0,0,0.5)';
                  const img = e.currentTarget.querySelector('.deck-bg') as HTMLElement;
                  if (img) img.style.transform = 'scale(1)';
                }}
              >
                <div
                  className="deck-bg"
                  style={{
                    position: 'absolute',
                    inset: 0,
                    backgroundImage: `url(/assets/cards/${coverCardId}.png)`,
                    backgroundSize: 'cover',
                    backgroundPosition: 'center 25%',
                    transition: 'transform 0.6s ease-out',
                    zIndex: 0,
                  }}
                ></div>

                <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(to top, rgba(15,23,42,1) 0%, rgba(15,23,42,0.6) 50%, rgba(15,23,42,0.2) 100%)', zIndex: 1 }}></div>

                <div style={{ position: 'relative', zIndex: 2, display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <h3 style={{ margin: 0, fontSize: '1.4rem', fontWeight: '900', color: 'white', textShadow: '0 2px 4px rgba(0,0,0,0.8)' }}>{d.name}</h3>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ color: cardCount === 60 ? '#34d399' : '#fbbf24', fontWeight: 'bold', fontSize: '0.9rem', background: 'rgba(0,0,0,0.5)', padding: '0.2rem 0.6rem', borderRadius: '12px' }}>{cardCount} / 60 Kartu</span>
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

  return (
    <BattleArena
      obs={obs}
      allCardsData={allCardsData}
      isSpectator={false}
      onSelectOption={sendSelect}
      onRestartMatch={handleRestart}
      onExitMatch={() => router.push('/')}
    />
  );
}
