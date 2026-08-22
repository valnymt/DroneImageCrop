// Applies pending drizzle/*.sql migrations to the local D1 database that
// `vinext dev` / the Cloudflare Vite plugin emulates via Miniflare. Local
// dev has no separate "deploy" step to apply migrations at, unlike the
// hosting platform (see app/api/analyses/route.ts's error message), so run
// this once after `npm run db:generate` and after first starting the dev
// server (which creates the local D1 state directory).
import { DatabaseSync } from "node:sqlite";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const root = decodeURIComponent(new URL("..", import.meta.url).pathname.replace(/^\/([a-zA-Z]:)/, "$1"));
const d1Dir = join(root, ".wrangler", "state", "v3", "d1", "miniflare-D1DatabaseObject");
const migrationsDir = join(root, "drizzle");

if (!existsSync(d1Dir)) {
  console.error(
    `No local D1 state found at ${d1Dir}.\n` +
    "Run `npm run dev` once first (it creates the local D1 database on startup), then re-run this script."
  );
  process.exit(1);
}

const sqliteFiles = readdirSync(d1Dir).filter((f) => f.endsWith(".sqlite") && f !== "metadata.sqlite");
if (sqliteFiles.length === 0) {
  console.error(`No .sqlite database file found under ${d1Dir}.`);
  process.exit(1);
}

const migrationFiles = existsSync(migrationsDir)
  ? readdirSync(migrationsDir).filter((f) => f.endsWith(".sql")).sort()
  : [];
if (migrationFiles.length === 0) {
  console.error("No migrations found in drizzle/. Run `npm run db:generate` first.");
  process.exit(1);
}

for (const sqliteFile of sqliteFiles) {
  const dbPath = join(d1Dir, sqliteFile);
  const db = new DatabaseSync(dbPath);
  db.exec("CREATE TABLE IF NOT EXISTS _local_migrations (name TEXT PRIMARY KEY, applied_at TEXT DEFAULT CURRENT_TIMESTAMP)");
  const applied = new Set(db.prepare("SELECT name FROM _local_migrations").all().map((r) => r.name));

  for (const file of migrationFiles) {
    if (applied.has(file)) continue;
    const sql = readFileSync(join(migrationsDir, file), "utf-8");
    db.exec(sql);
    db.prepare("INSERT INTO _local_migrations (name) VALUES (?)").run(file);
    console.log(`Applied ${file} to ${sqliteFile}`);
  }
  db.close();
}

console.log("Local D1 database is up to date.");
