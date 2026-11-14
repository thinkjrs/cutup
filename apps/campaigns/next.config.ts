import type { NextConfig } from "next";

// The Rust functions under api/ answer /api/*, so the app has no API routes of its own.
const nextConfig: NextConfig = {};

export default nextConfig;
