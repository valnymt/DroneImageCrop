import { getSessionUser, publicUser } from "../_auth";

export async function GET(request: Request) {
  const user = await getSessionUser(request);
  return Response.json({ user: user ? publicUser(user) : null });
}
