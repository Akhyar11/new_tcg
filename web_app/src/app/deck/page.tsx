"use client";
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

interface Card {
  "Card ID": number;
  "Card Name": string;
  "Type": string;
  "HP": string;
  "Stage (Pokémon)/Type (Energy and Trainer)": string;
  "Move Name"?: string;
  "Cost"?: string;
  "Damage"?: string;
  "Effect Explanation"?: string;
  "Rule"?: string;
}

export default function DeckBuilder() {
  const [cards, setCards] = useState<Card[]>([]);
  const [deck, setDeck] = useState<Card[]>([]);
  const [search, setSearch] = useState('');
  const [visibleCount, setVisibleCount] = useState(50);
  const [previewCard, setPreviewCard] = useState<Card | null>(null);
  
  // Deck state
  const [deckName, setDeckName] = useState('Deck Baru Saya');
  const [selectedDeckId, setSelectedDeckId] = useState<string | null>(null);
  const [savedDecks, setSavedDecks] = useState<any[]>([]);
  
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const router = useRouter();

  // Load cards and user decks
  useEffect(() => {
    // 1. Fetch Cards
    fetch('/cards.json')
      .then(res => res.json())
      .then(data => {
        const unique = Array.from(new Map(data.map((c: Card) => [c['Card ID'], c])).values()) as Card[];
        setCards(unique);
      });

    // 2. Fetch User Decks (and check login)
    fetch('/api/deck')
      .then(async (res) => {
        if (res.status === 401) {
          router.push('/login');
          return;
        }
        const data = await res.json();
        if (data.decks) {
          setSavedDecks(data.decks);
        }
      });
  }, [router]);

  // Handle deck selection
  const loadDeck = (deckId: string) => {
    if (deckId === 'new') {
      setSelectedDeckId(null);
      setDeckName('Deck Baru Saya');
      setDeck([]);
      return;
    }
    
    const targetDeck = savedDecks.find(d => d.id === deckId);
    if (targetDeck) {
      setSelectedDeckId(targetDeck.id);
      setDeckName(targetDeck.name);
      try {
        const parsedIds = JSON.parse(targetDeck.cards);
        // Find full card objects from the IDs
        const loadedCards = parsedIds.map((id: number) => cards.find(c => c['Card ID'] === id)).filter(Boolean);
        setDeck(loadedCards);
      } catch (e) {
        setDeck([]);
      }
    }
  };

  // Reset visible count when search changes
  useEffect(() => {
    setVisibleCount(50);
  }, [search]);

  const filteredCards = cards.filter(c => 
    c['Card Name'].toLowerCase().includes(search.toLowerCase()) ||
    (c['Type'] && c['Type'].toLowerCase().includes(search.toLowerCase()))
  ).slice(0, visibleCount);

  const addToDeck = (card: Card) => {
    if (deck.length >= 60) {
      setError('Deck maksimal 60 kartu!');
      setTimeout(() => setError(''), 3000);
      return;
    }
    
    // Validasi maksimal 4 kartu bernama sama (kecuali Basic Energy)
    if (!card["Stage (Pokémon)/Type (Energy and Trainer)"]?.includes("Basic Energy")) {
      const count = deck.filter(c => c['Card Name'] === card['Card Name']).length;
      if (count >= 4) {
        setError(`Maksimal 4 kartu ${card['Card Name']} dalam satu deck!`);
        setTimeout(() => setError(''), 3000);
        return;
      }
    }
    
    setDeck([...deck, card]);
  };

  const removeFromDeckById = (cardId: number) => {
    const indexToRemove = deck.findIndex(c => c['Card ID'] === cardId);
    if (indexToRemove !== -1) {
      const newDeck = [...deck];
      newDeck.splice(indexToRemove, 1);
      setDeck(newDeck);
    }
  };

  const getSuperCategory = (stage: string) => {
    if (!stage) return 'POKEMON';
    if (stage.includes('Energy')) return 'ENERGY';
    if (stage.includes('Pokémon') || stage === 'Pokemon') return 'POKEMON';
    return 'TRAINER';
  };

  // Group deck cards
  const groupedDeck: Record<number, { card: Card, count: number }> = {};
  deck.forEach(card => {
    const id = card['Card ID'];
    if (groupedDeck[id]) {
      groupedDeck[id].count += 1;
    } else {
      groupedDeck[id] = { card, count: 1 };
    }
  });

  const pokemonList = Object.values(groupedDeck).filter(item => getSuperCategory(item.card["Stage (Pokémon)/Type (Energy and Trainer)"]) === 'POKEMON');
  const energyList = Object.values(groupedDeck).filter(item => getSuperCategory(item.card["Stage (Pokémon)/Type (Energy and Trainer)"]) === 'ENERGY');
  const trainerList = Object.values(groupedDeck).filter(item => getSuperCategory(item.card["Stage (Pokémon)/Type (Energy and Trainer)"]) === 'TRAINER');

  const countPokemon = pokemonList.reduce((acc, curr) => acc + curr.count, 0);
  const countEnergy = energyList.reduce((acc, curr) => acc + curr.count, 0);
  const countTrainer = trainerList.reduce((acc, curr) => acc + curr.count, 0);

  const saveDeck = async () => {
    if (deck.length !== 60) {
      setError('Deck harus berjumlah tepat 60 kartu!');
      setTimeout(() => setError(''), 3000);
      return;
    }
    
    const cardIds = deck.map(c => c['Card ID']);
    
    // --- Validasi Engine C++ ---
    try {
      setSuccess('Memvalidasi deck dengan C++ Engine...');
      const validateRes = await fetch('http://localhost:8001/validate_deck', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ deck: cardIds })
      });
      
      const validateData = await validateRes.json();
      if (!validateData.valid) {
        setSuccess('');
        setError('Deck tidak valid: ' + (validateData.reason || 'Ditolak oleh engine'));
        return;
      }
    } catch (e) {
      setSuccess('');
      setError('Gagal terhubung ke C++ Engine untuk validasi. Pastikan server.py berjalan!');
      return;
    }
    // ----------------------------
    
    const method = selectedDeckId ? 'PUT' : 'POST';
    const payload = selectedDeckId 
      ? { id: selectedDeckId, name: deckName, cards: cardIds }
      : { name: deckName, cards: cardIds };

    const res = await fetch('/api/deck', {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (res.ok) {
      const data = await res.json();
      setSuccess('Deck valid dan berhasil disimpan!');
      setTimeout(() => setSuccess(''), 3000);
      
      // Update savedDecks list
      if (!selectedDeckId && data.deck) {
        setSelectedDeckId(data.deck.id);
        setSavedDecks([data.deck, ...savedDecks]);
      } else if (selectedDeckId) {
        setSavedDecks(savedDecks.map(d => d.id === selectedDeckId ? { ...d, name: deckName, cards: JSON.stringify(cardIds) } : d));
      }
    } else {
      const data = await res.json();
      setError(data.error || 'Gagal menyimpan deck');
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#0f172a', color: 'white', fontFamily: 'sans-serif' }}>
      
      {/* TOP NAVBAR */}
      <div style={{ height: '70px', background: 'rgba(15, 23, 42, 0.95)', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', padding: '0 2rem', justifyContent: 'space-between', zIndex: 10, backdropFilter: 'blur(10px)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
          <button 
            onClick={() => router.push('/')} 
            style={{ background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: '1.1rem', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '0.5rem', transition: 'color 0.2s' }}
            onMouseEnter={e => e.currentTarget.style.color = '#38bdf8'}
            onMouseLeave={e => e.currentTarget.style.color = '#94a3b8'}
          >
            ← Kembali ke Beranda
          </button>
          <div style={{ width: '1px', height: '24px', background: 'rgba(255,255,255,0.1)' }}></div>
          <h1 style={{ margin: 0, fontSize: '1.5rem', background: 'linear-gradient(to right, #38bdf8, #818cf8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', fontWeight: '900', letterSpacing: '1px' }}>
            DECK BUILDER
          </h1>
        </div>
        
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button 
            onClick={() => router.push('/play/ai')} 
            style={{ padding: '0.6rem 1.2rem', background: 'rgba(255,255,255,0.05)', color: '#e2e8f0', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold', transition: 'all 0.2s' }}
            onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.1)'; e.currentTarget.style.borderColor = '#38bdf8'; }}
            onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.05)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'; }}
          >
            ⚔️ Lawan AI
          </button>
          <button 
            onClick={() => router.push('/play/multiplayer')} 
            style={{ padding: '0.6rem 1.2rem', background: 'linear-gradient(135deg, #6366f1, #4f46e5)', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold', transition: 'transform 0.2s', boxShadow: '0 4px 6px -1px rgba(99, 102, 241, 0.4)' }}
            onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
            onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
          >
            🌐 Multiplayer
          </button>
        </div>
      </div>

      {/* MAIN CONTENT AREA */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        
        {/* LEFT PANE: DECK LIST */}
      <div style={{ width: '350px', borderRight: '1px solid rgba(255,255,255,0.1)', padding: '1rem', display: 'flex', flexDirection: 'column', background: '#0b1121' }}>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h2 style={{ margin: 0, fontSize: '1.2rem' }}>Deck Builder ({deck.length}/60)</h2>
        </div>

        {/* Pemilihan Deck */}
        <select 
          value={selectedDeckId || 'new'} 
          onChange={e => loadDeck(e.target.value)}
          style={{ width: '100%', padding: '0.6rem', background: '#1e293b', border: '1px solid rgba(255,255,255,0.2)', color: 'white', borderRadius: '8px', marginBottom: '0.5rem', outline: 'none', cursor: 'pointer' }}
        >
          <option value="new">+ Buat Deck Baru</option>
          {savedDecks.map(d => (
            <option key={d.id} value={d.id}>{d.name}</option>
          ))}
        </select>

        <input 
          value={deckName}
          onChange={e => setDeckName(e.target.value)}
          placeholder="Nama Deck"
          style={{ width: '100%', padding: '0.6rem', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', borderRadius: '8px', marginBottom: '1rem', outline: 'none' }}
        />
        
        {error && <div style={{ background: 'rgba(239, 68, 68, 0.2)', color: '#fca5a5', padding: '0.5rem', borderRadius: '8px', marginBottom: '1rem', border: '1px solid #ef4444', fontSize: '0.9rem' }}>{error}</div>}
        {success && <div style={{ background: 'rgba(16, 185, 129, 0.2)', color: '#6ee7b7', padding: '0.5rem', borderRadius: '8px', marginBottom: '1rem', border: '1px solid #10b981', fontSize: '0.9rem' }}>{success}</div>}

        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '1rem', paddingRight: '0.5rem' }}>
          
          {/* POKEMON GROUP */}
          {pokemonList.length > 0 && (
            <div>
              <div style={{ background: 'rgba(255,255,255,0.05)', padding: '0.4rem 1rem', fontSize: '0.8rem', fontWeight: 'bold', color: '#94a3b8', letterSpacing: '1px' }}>POKÉMON ({countPokemon})</div>
              {pokemonList.map(item => (
                <div key={item.card['Card ID']} style={{ display: 'flex', alignItems: 'center', height: '40px', borderBottom: '1px solid rgba(255,255,255,0.05)', background: 'linear-gradient(90deg, rgba(30,41,59,1) 0%, rgba(30,41,59,0.4) 100%)', position: 'relative', overflow: 'hidden' }}>
                  <div style={{ position: 'absolute', top: 0, right: 0, bottom: 0, width: '60%', backgroundImage: `url(/assets/cards/${item.card['Card ID']}.png)`, backgroundSize: 'cover', backgroundPosition: 'center 20%', opacity: 0.15, zIndex: 0, maskImage: 'linear-gradient(to right, transparent, black)' }}></div>
                  <div style={{ display: 'flex', alignItems: 'center', zIndex: 1, flex: 1, paddingLeft: '0.5rem' }}>
                    <button onClick={() => removeFromDeckById(item.card['Card ID'])} style={{ background: 'transparent', border: 'none', color: '#cbd5e1', cursor: 'pointer', fontSize: '1.2rem', width: '24px' }}>-</button>
                    <button onClick={() => addToDeck(item.card)} style={{ background: 'transparent', border: 'none', color: '#cbd5e1', cursor: 'pointer', fontSize: '1.2rem', width: '24px' }}>+</button>
                    <span style={{ fontWeight: '900', minWidth: '24px', textAlign: 'center', color: 'white', textShadow: '0 1px 2px black' }}>{item.count}</span>
                    <span style={{ marginLeft: '0.5rem', color: '#f8fafc', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontSize: '0.9rem', fontWeight: '500' }}>{item.card['Card Name']}</span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* TRAINER GROUP */}
          {trainerList.length > 0 && (
            <div>
              <div style={{ background: 'rgba(255,255,255,0.05)', padding: '0.4rem 1rem', fontSize: '0.8rem', fontWeight: 'bold', color: '#94a3b8', letterSpacing: '1px' }}>TRAINER ({countTrainer})</div>
              {trainerList.map(item => (
                <div key={item.card['Card ID']} style={{ display: 'flex', alignItems: 'center', height: '40px', borderBottom: '1px solid rgba(255,255,255,0.05)', background: 'linear-gradient(90deg, rgba(30,41,59,1) 0%, rgba(30,41,59,0.4) 100%)', position: 'relative', overflow: 'hidden' }}>
                  <div style={{ position: 'absolute', top: 0, right: 0, bottom: 0, width: '60%', backgroundImage: `url(/assets/cards/${item.card['Card ID']}.png)`, backgroundSize: 'cover', backgroundPosition: 'center 20%', opacity: 0.15, zIndex: 0, maskImage: 'linear-gradient(to right, transparent, black)' }}></div>
                  <div style={{ display: 'flex', alignItems: 'center', zIndex: 1, flex: 1, paddingLeft: '0.5rem' }}>
                    <button onClick={() => removeFromDeckById(item.card['Card ID'])} style={{ background: 'transparent', border: 'none', color: '#cbd5e1', cursor: 'pointer', fontSize: '1.2rem', width: '24px' }}>-</button>
                    <button onClick={() => addToDeck(item.card)} style={{ background: 'transparent', border: 'none', color: '#cbd5e1', cursor: 'pointer', fontSize: '1.2rem', width: '24px' }}>+</button>
                    <span style={{ fontWeight: '900', minWidth: '24px', textAlign: 'center', color: 'white', textShadow: '0 1px 2px black' }}>{item.count}</span>
                    <span style={{ marginLeft: '0.5rem', color: '#f8fafc', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontSize: '0.9rem', fontWeight: '500' }}>{item.card['Card Name']}</span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* ENERGY GROUP */}
          {energyList.length > 0 && (
            <div>
              <div style={{ background: 'rgba(255,255,255,0.05)', padding: '0.4rem 1rem', fontSize: '0.8rem', fontWeight: 'bold', color: '#94a3b8', letterSpacing: '1px' }}>ENERGY ({countEnergy})</div>
              {energyList.map(item => (
                <div key={item.card['Card ID']} style={{ display: 'flex', alignItems: 'center', height: '40px', borderBottom: '1px solid rgba(255,255,255,0.05)', background: 'linear-gradient(90deg, rgba(30,41,59,1) 0%, rgba(30,41,59,0.4) 100%)', position: 'relative', overflow: 'hidden' }}>
                  <div style={{ position: 'absolute', top: 0, right: 0, bottom: 0, width: '60%', backgroundImage: `url(/assets/cards/${item.card['Card ID']}.png)`, backgroundSize: 'cover', backgroundPosition: 'center 20%', opacity: 0.15, zIndex: 0, maskImage: 'linear-gradient(to right, transparent, black)' }}></div>
                  <div style={{ display: 'flex', alignItems: 'center', zIndex: 1, flex: 1, paddingLeft: '0.5rem' }}>
                    <button onClick={() => removeFromDeckById(item.card['Card ID'])} style={{ background: 'transparent', border: 'none', color: '#cbd5e1', cursor: 'pointer', fontSize: '1.2rem', width: '24px' }}>-</button>
                    <button onClick={() => addToDeck(item.card)} style={{ background: 'transparent', border: 'none', color: '#cbd5e1', cursor: 'pointer', fontSize: '1.2rem', width: '24px' }}>+</button>
                    <span style={{ fontWeight: '900', minWidth: '24px', textAlign: 'center', color: 'white', textShadow: '0 1px 2px black' }}>{item.count}</span>
                    <span style={{ marginLeft: '0.5rem', color: '#f8fafc', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontSize: '0.9rem', fontWeight: '500' }}>{item.card['Card Name']}</span>
                  </div>
                </div>
              ))}
            </div>
          )}

        </div>

        <button onClick={saveDeck} style={{ marginTop: '1rem', padding: '1rem', background: '#3b82f6', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold', letterSpacing: '1px', transition: 'background 0.2s' }} onMouseEnter={e => e.currentTarget.style.background = '#2563eb'} onMouseLeave={e => e.currentTarget.style.background = '#3b82f6'}>
          SIMPAN DECK
        </button>
      </div>

      {/* RIGHT PANE: GALLERY */}
      <div style={{ flex: 1, padding: '1rem', display: 'flex', flexDirection: 'column' }}>
        <h2 style={{ margin: '0 0 1rem 0' }}>Galeri Kartu</h2>
        <input 
          placeholder="Cari nama kartu atau elemen (ex: Pikachu, Fire)"
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={{ width: '100%', padding: '0.75rem', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.2)', color: 'white', borderRadius: '8px', marginBottom: '1rem' }}
        />
        
        <div 
          style={{ flex: 1, overflowY: 'auto', display: 'flex', flexWrap: 'wrap', gap: '1rem', alignContent: 'flex-start' }}
          onScroll={(e) => {
            const { scrollTop, clientHeight, scrollHeight } = e.currentTarget;
            if (scrollHeight - scrollTop <= clientHeight + 150) {
              setVisibleCount(prev => prev + 50);
            }
          }}
        >
          {filteredCards.map(card => (
            <div key={card['Card ID']} onClick={() => setPreviewCard(card)} style={{ width: '120px', cursor: 'pointer', textAlign: 'center', transition: 'transform 0.2s' }} onMouseEnter={e => e.currentTarget.style.transform = 'scale(1.05)'} onMouseLeave={e => e.currentTarget.style.transform = 'scale(1)'}>
              <div style={{ width: '120px', height: '167px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <img 
                  src={`/assets/cards/${card['Card ID']}.png`} 
                  alt={card['Card Name']} 
                  loading="lazy"
                  style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.3)' }} 
                  onError={(e) => { e.currentTarget.style.display = 'none'; e.currentTarget.parentElement!.innerHTML = `<span style="padding: 5px; font-size: 12px;">Missing Image:<br/>${card['Card Name']}</span>`; }}
                />
              </div>
              <div style={{ fontSize: '0.8rem', marginTop: '0.5rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{card['Card Name']}</div>
            </div>
          ))}
        </div>
      </div>
      
      {/* MODAL PREVIEW KARTU */}
      {previewCard && (
        <div 
          style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.85)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50, backdropFilter: 'blur(8px)' }} 
          onClick={() => setPreviewCard(null)}
        >
          <div 
            style={{ background: '#1e293b', padding: '2rem', borderRadius: '24px', display: 'flex', gap: '2rem', maxWidth: '800px', width: '90%', boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)', border: '1px solid rgba(255,255,255,0.1)' }} 
            onClick={e => e.stopPropagation()}
          >
            {/* Sisi Kiri: Gambar Kartu Besar */}
            <div style={{ width: '300px', flexShrink: 0, background: 'rgba(255,255,255,0.05)', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '418px' }}>
              <img 
                src={`/assets/cards/${previewCard['Card ID']}.png`} 
                alt={previewCard['Card Name']} 
                style={{ width: '100%', borderRadius: '16px', boxShadow: '0 10px 25px rgba(0,0,0,0.5)' }} 
                onError={(e) => { e.currentTarget.style.display = 'none'; e.currentTarget.parentElement!.innerHTML = `<div style="padding: 20px; text-align: center; font-size: 1.2rem;">Missing Image:<br/><strong>${previewCard['Card Name']}</strong></div>`; }}
              />
            </div>

            {/* Sisi Kanan: Detail & Tombol */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              <h2 style={{ fontSize: '2.5rem', margin: '0 0 1rem 0', background: 'linear-gradient(to right, #38bdf8, #818cf8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                {previewCard['Card Name']}
              </h2>
              
              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1.5rem', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)', marginBottom: '1rem' }}>
                <p style={{ color: '#e2e8f0', fontSize: '1.1rem', marginBottom: '0.5rem' }}><strong style={{ color: '#94a3b8', width: '100px', display: 'inline-block' }}>Tipe</strong>: {previewCard['Type'] || '-'}</p>
                <p style={{ color: '#e2e8f0', fontSize: '1.1rem', marginBottom: '0.5rem' }}><strong style={{ color: '#94a3b8', width: '100px', display: 'inline-block' }}>HP</strong>: {previewCard['HP'] || '-'}</p>
                <p style={{ color: '#e2e8f0', fontSize: '1.1rem', marginBottom: '0' }}><strong style={{ color: '#94a3b8', width: '100px', display: 'inline-block' }}>Kategori</strong>: {previewCard["Stage (Pokémon)/Type (Energy and Trainer)"] || '-'}</p>
              </div>

              {/* Blok Kemampuan / Serangan / Efek */}
              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '1.5rem', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)', marginBottom: '1rem', flex: 1, overflowY: 'auto', maxHeight: '200px' }}>
                <h3 style={{ color: '#38bdf8', marginTop: 0, marginBottom: '0.5rem', fontSize: '1.2rem' }}>Efek & Serangan</h3>
                
                {previewCard['Move Name'] && previewCard['Move Name'] !== 'NaN' ? (
                  <div style={{ marginBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '1rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                      <strong style={{ fontSize: '1.1rem', color: '#f8fafc' }}>{String(previewCard['Move Name']).replace(/\n/g, ' / ')}</strong>
                      {previewCard['Damage'] && previewCard['Damage'] !== 'NaN' && (
                        <span style={{ background: '#ef4444', color: 'white', padding: '0.2rem 0.5rem', borderRadius: '4px', fontWeight: 'bold' }}>{previewCard['Damage']} DMG</span>
                      )}
                    </div>
                    {previewCard['Cost'] && previewCard['Cost'] !== 'NaN' && (
                      <div style={{ fontSize: '0.9rem', color: '#fbbf24', marginBottom: '0.5rem' }}>Cost: {previewCard['Cost']}</div>
                    )}
                    {previewCard['Effect Explanation'] && previewCard['Effect Explanation'] !== 'NaN' && (
                      <p style={{ fontSize: '1rem', color: '#cbd5e1', lineHeight: '1.5', margin: 0 }}>{String(previewCard['Effect Explanation'])}</p>
                    )}
                  </div>
                ) : null}

                {previewCard['Rule'] && previewCard['Rule'] !== 'NaN' && (
                  <div style={{ background: 'rgba(239, 68, 68, 0.1)', borderLeft: '4px solid #ef4444', padding: '0.75rem', borderRadius: '0 8px 8px 0' }}>
                    <strong style={{ color: '#ef4444', display: 'block', marginBottom: '0.25rem' }}>Aturan Khusus (Rule):</strong>
                    <span style={{ fontSize: '0.95rem', color: '#cbd5e1' }}>{String(previewCard['Rule'])}</span>
                  </div>
                )}
                
                {(!previewCard['Move Name'] || previewCard['Move Name'] === 'NaN') && (!previewCard['Rule'] || previewCard['Rule'] === 'NaN') && (!previewCard['Effect Explanation'] || previewCard['Effect Explanation'] === 'NaN') && (
                  <p style={{ color: '#64748b', fontStyle: 'italic', margin: 0 }}>Tidak ada serangan atau efek khusus.</p>
                )}
              </div>
              
              <div style={{ marginTop: 'auto', display: 'flex', gap: '1rem' }}>
                <button 
                  onClick={() => { addToDeck(previewCard); setPreviewCard(null); }} 
                  style={{ padding: '1rem 2rem', background: 'linear-gradient(135deg, #10b981, #059669)', color: 'white', border: 'none', borderRadius: '12px', cursor: 'pointer', fontWeight: 'bold', fontSize: '1.1rem', flex: 1, transition: 'transform 0.2s', boxShadow: '0 4px 6px -1px rgba(16, 185, 129, 0.4)' }}
                  onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
                  onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
                >
                  + Tambahkan ke Deck
                </button>
                <button 
                  onClick={() => setPreviewCard(null)} 
                  style={{ padding: '1rem 2rem', background: 'rgba(255,255,255,0.05)', color: '#94a3b8', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', cursor: 'pointer', fontWeight: 'bold', fontSize: '1.1rem', transition: 'background 0.2s' }}
                  onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.1)'; e.currentTarget.style.color = 'white'; }}
                  onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.05)'; e.currentTarget.style.color = '#94a3b8'; }}
                >
                  Tutup
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
      </div>
      
    </div>
  );
}
