import { eq } from "drizzle-orm";
import { getDb } from "../../../../db";
import { sessions } from "../../../../db/schema";
import { clearSessionCookie, sessionCookie, sessionToken } from "../_auth";

export async function POST(request: Request) {
  const token = sessionToken(request);
  if (token) await getDb().delete(sessions).where(eq(sessions.token, token));
  return new Response(null, { status: 204, headers: { "Set-Cookie": clearSessionCookie() } });
}
