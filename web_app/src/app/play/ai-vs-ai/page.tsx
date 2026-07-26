"use client";

import React, { useEffect, useState, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import BattleArena, { Card } from '@/components/BattleArena';

export default function PlayAIVsAIPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [gameState, setGameState] = useState<'SELECT_MODE' | 'PLAYING'>('SELECT_MODE');
  const [allCardsData, setAllCardsData] = useState<Card[]>([]);
  const [error, setError] = useState('');

  // ================= GAME ENGINE STATE =================
  const [obs, setObs] = useState<any>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const initializeGame = async () => {
      try {
        const cardRes = await fetch('/cards.json');
        const allCards = (await cardRes.json()) as Card[];
        setAllCardsData(allCards);
        setLoading(false);
      } catch (e) {
        setError('Gagal memuat data kartu game.');
        setLoading(false);
      }
    };

    initializeGame();

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const startAIVsAIMatch = () => {
    try {
      if (wsRef.current) wsRef.current.close();

      const socket = new WebSocket('ws://localhost:8001/ws');
      socket.onopen = () => {
        console.log("Connected to C++ Engine for AI vs AI!");
        socket.send(JSON.stringify({ type: 'start_ai_vs_ai' }));
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'update') {
          console.log("🔥 State dari C++ Engine (AI vs AI):", data.obs);
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
      setError('Gagal memulai simulasi AI vs AI.');
    }
  };

  const handleRestart = () => {
    if (wsRef.current) wsRef.current.close();
    setObs(null);
    startAIVsAIMatch();
  };

  if (loading) {
    return <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#050b14', color: 'white', fontFamily: 'sans-serif' }}>Memuat Arena AI vs AI...</div>;
  }

  if (error) {
    return (
      <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#050b14', color: 'white', fontFamily: 'sans-serif', gap: '1rem' }}>
        <h2 style={{ color: '#ef4444' }}>Tidak Bisa Memulai Simulasi</h2>
        <p>{error}</p>
        <Link href="/" style={{ padding: '0.8rem 1.5rem', background: '#3b82f6', color: 'white', textDecoration: 'none', borderRadius: '8px', fontWeight: 'bold' }}>Kembali ke Menu Utama</Link>
      </div>
    );
  }

  if (gameState === 'SELECT_MODE') {
    return (
      <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: '#0f172a', color: 'white', fontFamily: '"Inter", sans-serif', padding: '4rem 2rem', alignItems: 'center', justifyContent: 'center' }}>
        <Link href="/" style={{ position: 'absolute', top: '2rem', left: '2rem', color: '#94a3b8', textDecoration: 'none', fontWeight: 'bold' }}>← Kembali</Link>

        <div style={{ textAlign: 'center', maxWidth: '600px', background: 'rgba(30, 41, 59, 0.5)', padding: '3rem', borderRadius: '24px', border: '1px solid rgba(255,255,255,0.1)', boxShadow: '0 20px 50px rgba(0,0,0,0.5)', backdropFilter: 'blur(10px)' }}>
          <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>🤖 ⚔️ 🤖</div>
          <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem', background: 'linear-gradient(to right, #f43f5e, #8b5cf6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', fontWeight: '900' }}>
            Simulasi AI vs AI
          </h1>
          <p style={{ color: '#94a3b8', marginBottom: '2.5rem', lineHeight: '1.6' }}>
            Saksikan dua model kecerdasan buatan (JAX PPO Agent) bertarung secara otomatis dalam pertempuran Pokémon TCG secara real-time.
          </p>

          <button
            onClick={startAIVsAIMatch}
            style={{
              width: '100%',
              padding: '1.2rem 2rem',
              background: 'linear-gradient(135deg, #f43f5e, #be123c)',
              color: 'white',
              border: 'none',
              borderRadius: '14px',
              fontSize: '1.2rem',
              fontWeight: '900',
              cursor: 'pointer',
              boxShadow: '0 10px 25px rgba(244, 63, 94, 0.4)',
              transition: 'transform 0.2s',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.transform = 'scale(1.03)')}
            onMouseLeave={(e) => (e.currentTarget.style.transform = 'scale(1)')}
          >
            ▶ Mulai Pertandingan AI vs AI
          </button>
        </div>
      </div>
    );
  }

  return (
    <BattleArena
      obs={obs}
      allCardsData={allCardsData}
      isSpectator={true}
      onRestartMatch={handleRestart}
      onExitMatch={() => router.push('/')}
    />
  );
}
