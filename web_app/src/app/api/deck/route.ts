import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';
import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET || 'rahasia-pokemon-tcg-super-kuat';

const getUserFromReq = (req: Request) => {
  const cookieHeader = req.headers.get('cookie');
  if (!cookieHeader) return null;
  const token = cookieHeader.split('; ').find(row => row.startsWith('token='))?.split('=')[1];
  if (!token) return null;
  try {
    return jwt.verify(token, JWT_SECRET) as { id: string, username: string };
  } catch {
    return null;
  }
};

export async function GET(req: Request) {
  const user = getUserFromReq(req);
  if (!user) return NextResponse.json({ error: 'Belum login' }, { status: 401 });

  try {
    const decks = await prisma.deck.findMany({
      where: { userId: user.id },
      orderBy: { updatedAt: 'desc' }
    });
    return NextResponse.json({ decks });
  } catch (error) {
    return NextResponse.json({ error: 'Terjadi kesalahan' }, { status: 500 });
  }
}

export async function POST(req: Request) {
  const user = getUserFromReq(req);
  if (!user) return NextResponse.json({ error: 'Belum login' }, { status: 401 });

  try {
    const { name, cards } = await req.json();
    if (!name || !cards || !Array.isArray(cards)) return NextResponse.json({ error: 'Data tidak lengkap' }, { status: 400 });

    const newDeck = await prisma.deck.create({
      data: {
        name,
        cards: JSON.stringify(cards),
        userId: user.id
      }
    });
    return NextResponse.json({ message: 'Deck berhasil dibuat', deck: newDeck });
  } catch (error) {
    return NextResponse.json({ error: 'Terjadi kesalahan internal' }, { status: 500 });
  }
}

export async function PUT(req: Request) {
  const user = getUserFromReq(req);
  if (!user) return NextResponse.json({ error: 'Belum login' }, { status: 401 });

  try {
    const { id, name, cards } = await req.json();
    if (!id || !name || !cards || !Array.isArray(cards)) return NextResponse.json({ error: 'Data tidak lengkap' }, { status: 400 });

    const updatedDeck = await prisma.deck.updateMany({
      where: { id: id, userId: user.id }, // pastikan hanya milik user ini
      data: { name, cards: JSON.stringify(cards) }
    });

    if (updatedDeck.count === 0) return NextResponse.json({ error: 'Deck tidak ditemukan atau Anda tidak berhak' }, { status: 404 });

    return NextResponse.json({ message: 'Deck berhasil diperbarui' });
  } catch (error) {
    return NextResponse.json({ error: 'Terjadi kesalahan internal' }, { status: 500 });
  }
}
