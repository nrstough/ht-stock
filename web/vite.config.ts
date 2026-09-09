import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// The app is served by `python -m ht.serve`, which also answers /api. In development Vite
// proxies there so the frontend never needs a second origin and no CORS story exists.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8765' } },
  build: { outDir: 'dist', emptyOutDir: true },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test-setup.ts'],
  },
})
