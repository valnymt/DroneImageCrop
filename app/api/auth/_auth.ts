import { eq } from "drizzle-orm";
import { getDb } from "../../../db";
import { sessions, users } from "../../../db/schema";

const SESSION_COOKIE = "agrisight_session";
const SESSION_MAX_AGE = 60 * 60 * 24 * 30;
const PBKDF2_ITERATIONS = 240_000;

type User = typeof users.$inferSelect;

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function hexToBytes(value: string): Uint8Array {
  if (!/^[0-9a-f]+$/i.test(value) || value.length % 2 !== 0) throw new Error("Invalid salt");
  return Uint8Array.from(value.match(/.{2}/g)!, (pair) => Number.parseInt(pair, 16));
}

async function derivePasswordHash(password: string, salt: Uint8Array): Promise<string> {
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(password), "PBKDF2", false, ["deriveBits"]);
  const bits = await crypto.subtle.deriveBits(
    { name: "PBKDF2", salt, iterations: PBKDF2_ITERATIONS, hash: "SHA-256" },
    key,
    256,
  );
  return `${bytesToHex(salt)}$${bytesToHex(new Uint8Array(bits))}`;
}

export async function hashPassword(password: string): Promise<string> {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  return derivePasswordHash(password, salt);
}

export async function verifyPassword(password: string, stored: string): Promise<boolean> {
  try {
    const [saltHex, expected] = stored.split("$", 2);
    const actual = await derivePasswordHash(password, hexToBytes(saltHex));
    return timingSafeEqual(actual.split("$")[1], expected);
  } catch {
    return false;
  }
}

function timingSafeEqual(left: string, right: string): boolean {
  if (left.length !== right.length) return false;
  let difference = 0;
  for (let index = 0; index < left.length; index += 1) {
    difference |= left.charCodeAt(index) ^ right.charCodeAt(index);
  }
  return difference === 0;
}

export function normalizeEmail(value: string): string {
  return value.trim().toLowerCase();
}

export function validEmail(value: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
}

export function sessionCookie(token: string): string {
  return `${SESSION_COOKIE}=${token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${SESSION_MAX_AGE}`;
}

export function clearSessionCookie(): string {
  return `${SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0`;
}

function sessionToken(request: Request): string | null {
  const cookieHeader = request.headers.get("Cookie") ?? "";
  const match = cookieHeader.match(new RegExp(`(?:^|;\\s*)${SESSION_COOKIE}=([^;]+)`));
  return match ? match[1] : null;
}

export async function createSession(userId: number): Promise<string> {
  const token = bytesToHex(crypto.getRandomValues(new Uint8Array(32)));
  await getDb().insert(sessions).values({ token, userId });
  return token;
}

export async function getSessionUser(request: Request): Promise<User | null> {
  const token = sessionToken(request);
  if (!token) return null;
  const [session] = await getDb().select().from(sessions).where(eq(sessions.token, token)).limit(1);
  if (!session) return null;
  const [user] = await getDb().select().from(users).where(eq(users.id, session.userId)).limit(1);
  return user ?? null;
}

export function publicUser(user: User) {
  return { id: String(user.id), email: user.email, username: user.username, occupation: user.occupation };
}

export { SESSION_COOKIE, sessionToken };
