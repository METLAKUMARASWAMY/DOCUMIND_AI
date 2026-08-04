import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    css: true
  },
  server: {
    port: Number(process.env.FRONTEND_PORT || 5174),
    proxy: {
      "/api": "http://localhost:8000"
    }
  }
});
