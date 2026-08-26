import { eq } from "drizzle-orm";
import { getDb } from "../../../../db";
import { users } from "../../../../db/schema";
import { createSession, normalizeEmail, publicUser, sessionCookie, verifyPassword } from "../_auth";

export async function POST(request: Request) {
  const body = await request.json().catch(() => ({})) as { email?: unknown; password?: unknown };
  const email = typeof body.email === "string" ? normalizeEmail(body.email) : "";
  const password = typeof body.password === "string" ? body.password : "";
  const [user] = await getDb().select().from(users).where(eq(users.email, email)).limit(1);

  if (!user || !password || !(await verifyPassword(password, user.passwordHash))) {
    return Response.json({ detail: "Invalid email or password." }, { status: 401 });
  }

  const token = await createSession(user.id);
  return new Response(JSON.stringify({ user: publicUser(user) }), {
    headers: { "Content-Type": "application/json", "Set-Cookie": sessionCookie(token) },
  });
}
