import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

/**
 * Vendor chunks are named explicitly.
 *
 * Without this, Rollup puts the modules shared by two lazily-loaded pages into
 * an anonymous shared chunk and names it after whichever source module it
 * encountered first. That is how a 385 kB chunk came to be called
 * `GenericIntelligencePanel` when its actual contents were pdfjs-dist — the
 * bundle report was blaming a 240-line component for a PDF engine. Naming the
 * heavy dependencies keeps the build output honest about what costs what.
 */
function manualChunks(id) {
  if (!id.includes("node_modules")) return undefined;
  if (/[\\/]node_modules[\\/](pdfjs-dist|react-pdf)[\\/]/.test(id)) return "vendor-pdf";
  if (/[\\/]node_modules[\\/](recharts|d3-|victory-|internmap|robust-predicates|delaunator)/.test(id))
    return "vendor-charts";
  if (/[\\/]node_modules[\\/](cytoscape|react-cytoscapejs|layout-base|cose-base|heap-js)/.test(id))
    return "vendor-graph";
  if (/[\\/]node_modules[\\/](@mui|@emotion)[\\/]/.test(id)) return "vendor-mui";
  if (/[\\/]node_modules[\\/](react|react-dom|scheduler|@tanstack)[\\/]/.test(id))
    return "vendor-react";
  return undefined;
}

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: { manualChunks },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    css: true,
    include: ["src/**/*.test.{ts,tsx}"],
  },
  server: {
    port: Number(process.env.FRONTEND_PORT || 5174),
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
