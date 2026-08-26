import { eq } from "drizzle-orm";
import { getDb } from "../../../../db";
import { users } from "../../../../db/schema";
import { createSession, hashPassword, normalizeEmail, publicUser, sessionCookie, validEmail } from "../_auth";

export async function POST(request: Request) {
  const body = await request.json().catch(() => ({})) as { email?: unknown; username?: unknown; occupation?: unknown; password?: unknown };
  const email = typeof body.email === "string" ? normalizeEmail(body.email) : "";
  const username = typeof body.username === "string" ? body.username.trim() : "";
  const occupation = typeof body.occupation === "string" ? body.occupation.trim() : "";
  const password = typeof body.password === "string" ? body.password : "";

  if (!validEmail(email)) return Response.json({ detail: "Enter a valid email address." }, { status: 422 });
  if (!username || username.length > 80) return Response.json({ detail: "Enter a username (maximum 80 characters)." }, { status: 422 });
  if (!occupation || occupation.length > 120) return Response.json({ detail: "Enter an occupation (maximum 120 characters)." }, { status: 422 });
  if (password.length < 8) return Response.json({ detail: "Password must be at least 8 characters." }, { status: 422 });

  const db = getDb();
  const existingRows = await db.select({ id: users.id }).from(users).where(eq(users.email, email)).limit(1);
  if (existingRows.length) return Response.json({ detail: "An account already exists with this email." }, { status: 409 });

  const [user] = await db.insert(users).values({ email, username, occupation, passwordHash: await hashPassword(password) }).returning();
  const token = await createSession(user.id);
  return new Response(JSON.stringify({ user: publicUser(user) }), {
    status: 201,
    headers: { "Content-Type": "application/json", "Set-Cookie": sessionCookie(token) },
  });
}
