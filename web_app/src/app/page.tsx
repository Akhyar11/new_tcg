"use client";
import Link from 'next/link';

export default function Home() {
  return (
    <div style={{ minHeight: '100vh', background: '#050b14', color: 'white', fontFamily: '"Inter", sans-serif', position: 'relative', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      
      {/* GLOWING ORBS (CSS Background Effects) */}
      <div style={{ position: 'absolute', top: '-10%', left: '-10%', width: '50vw', height: '50vw', background: 'radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, rgba(0,0,0,0) 70%)', filter: 'blur(100px)', zIndex: 0, pointerEvents: 'none' }}></div>
      <div style={{ position: 'absolute', bottom: '-20%', right: '-10%', width: '60vw', height: '60vw', background: 'radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, rgba(0,0,0,0) 70%)', filter: 'blur(100px)', zIndex: 0, pointerEvents: 'none' }}></div>
      
      {/* HEADER */}
      <header style={{ padding: '2rem 5%', display: 'flex', justifyContent: 'space-between', alignItems: 'center', zIndex: 10 }}>
        <div style={{ fontSize: '1.8rem', fontWeight: '900', letterSpacing: '-1px', background: 'linear-gradient(to right, #ffffff, #94a3b8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          PokéTCG <span style={{ color: '#38bdf8', WebkitTextFillColor: '#38bdf8' }}>Nexus</span>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <Link href="/login" style={{ padding: '0.6rem 1.5rem', color: '#e2e8f0', textDecoration: 'none', fontWeight: 'bold', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)', transition: 'all 0.3s' }} onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'} onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
            Masuk
          </Link>
          <Link href="/register" style={{ padding: '0.6rem 1.5rem', background: '#38bdf8', color: '#0f172a', textDecoration: 'none', fontWeight: 'bold', borderRadius: '8px', transition: 'all 0.3s', boxShadow: '0 4px 14px 0 rgba(56, 189, 248, 0.39)' }} onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'} onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}>
            Daftar Gratis
          </Link>
        </div>
      </header>

      {/* HERO SECTION */}
      <main style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '2rem 5%', zIndex: 10 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', width: '100%', maxWidth: '1400px', gap: '4rem', alignItems: 'center' }}>
          
          {/* LEFT SIDE: Typography */}
          <div style={{ flex: '1 1 500px', paddingRight: '2rem' }}>
            <div style={{ display: 'inline-block', padding: '0.4rem 1rem', background: 'rgba(56, 189, 248, 0.1)', border: '1px solid rgba(56, 189, 248, 0.2)', color: '#38bdf8', borderRadius: '50px', fontSize: '0.9rem', fontWeight: 'bold', marginBottom: '1.5rem', letterSpacing: '1px' }}>
              BETA V1.0 • POWERED BY JAX AI
            </div>
            <h1 style={{ fontSize: 'clamp(3rem, 5vw, 5rem)', fontWeight: '900', lineHeight: '1.1', marginBottom: '1.5rem', letterSpacing: '-2px' }}>
              Era Baru<br />
              <span style={{ background: 'linear-gradient(to right, #38bdf8, #818cf8, #c084fc)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Pertarungan TCG</span>
            </h1>
            <p style={{ fontSize: '1.25rem', color: '#94a3b8', lineHeight: '1.6', marginBottom: '2.5rem', maxWidth: '600px' }}>
              Bangun deck impianmu dari ribuan kartu legendaris. Tantang temanmu dalam pertempuran epik secara real-time, atau uji strategimu melawan kecerdasan buatan super (AI) yang kami latih secara khusus.
            </p>
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
              <Link href="/deck" style={{ padding: '1rem 2rem', background: 'linear-gradient(135deg, #3b82f6, #6366f1)', color: 'white', textDecoration: 'none', borderRadius: '12px', fontWeight: 'bold', fontSize: '1.1rem', transition: 'all 0.3s', boxShadow: '0 10px 25px -5px rgba(99, 102, 241, 0.5)', display: 'flex', alignItems: 'center', gap: '0.5rem' }} onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-3px)'} onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}>
                <span>Mulai Rakit Deck</span>
                <span>→</span>
              </Link>
            </div>
          </div>

          {/* RIGHT SIDE: Interactive Menu Cards */}
          <div style={{ flex: '1 1 400px', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            
            {/* CARD 1: Deck Builder */}
            <Link href="/deck" style={{ textDecoration: 'none', color: 'inherit' }}>
              <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.08)', padding: '2rem', borderRadius: '24px', backdropFilter: 'blur(10px)', transition: 'all 0.3s', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '1.5rem' }} onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)'; e.currentTarget.style.transform = 'translateX(10px)'; e.currentTarget.style.borderColor = 'rgba(56, 189, 248, 0.5)'; }} onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)'; e.currentTarget.style.transform = 'translateX(0)'; e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.08)'; }}>
                <div style={{ width: '60px', height: '60px', borderRadius: '16px', background: 'linear-gradient(135deg, #0ea5e9, #3b82f6)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.8rem', boxShadow: '0 10px 15px -3px rgba(14, 165, 233, 0.3)' }}>
                  🃏
                </div>
                <div>
                  <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1.5rem' }}>Deck Builder</h3>
                  <p style={{ margin: 0, color: '#94a3b8', fontSize: '1rem' }}>Rancang dan simpan strategi terbaikmu dengan koleksi 1.200+ kartu.</p>
                </div>
              </div>
            </Link>

            {/* CARD 2: Lawan AI */}
            <Link href="/play/ai" style={{ textDecoration: 'none', color: 'inherit' }}>
              <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.08)', padding: '2rem', borderRadius: '24px', backdropFilter: 'blur(10px)', transition: 'all 0.3s', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '1.5rem' }} onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)'; e.currentTarget.style.transform = 'translateX(10px)'; e.currentTarget.style.borderColor = 'rgba(236, 72, 153, 0.5)'; }} onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)'; e.currentTarget.style.transform = 'translateX(0)'; e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.08)'; }}>
                <div style={{ width: '60px', height: '60px', borderRadius: '16px', background: 'linear-gradient(135deg, #ec4899, #e11d48)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.8rem', boxShadow: '0 10px 15px -3px rgba(236, 72, 153, 0.3)' }}>
                  🤖
                </div>
                <div>
                  <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1.5rem' }}>Bermain vs AI</h3>
                  <p style={{ margin: 0, color: '#94a3b8', fontSize: '1rem' }}>Uji deck kamu melawan JAX AI yang telah dilatih secara khusus.</p>
                </div>
              </div>
            </Link>

            {/* CARD 3: Multiplayer */}
            <Link href="/play/multiplayer" style={{ textDecoration: 'none', color: 'inherit' }}>
              <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.08)', padding: '2rem', borderRadius: '24px', backdropFilter: 'blur(10px)', transition: 'all 0.3s', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '1.5rem' }} onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)'; e.currentTarget.style.transform = 'translateX(10px)'; e.currentTarget.style.borderColor = 'rgba(139, 92, 246, 0.5)'; }} onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)'; e.currentTarget.style.transform = 'translateX(0)'; e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.08)'; }}>
                <div style={{ width: '60px', height: '60px', borderRadius: '16px', background: 'linear-gradient(135deg, #8b5cf6, #6d28d9)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.8rem', boxShadow: '0 10px 15px -3px rgba(139, 92, 246, 0.3)' }}>
                  ⚔️
                </div>
                <div>
                  <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1.5rem' }}>Multiplayer (PvP)</h3>
                  <p margin={{ margin: 0, color: '#94a3b8', fontSize: '1rem' }}>Undang temanmu atau cari lawan secara acak di arena global.</p>
                </div>
              </div>
            </Link>

          </div>
        </div>
      </main>

    </div>
  );
}
