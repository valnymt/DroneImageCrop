import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AgriSight — Drone Crop Intelligence",
  description: "AI-powered aerial crop monitoring, plant detection, vegetation health and yield estimation.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
