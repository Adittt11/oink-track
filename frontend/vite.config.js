import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const backendTarget = process.env.VITE_BACKEND_URL || 'http://127.0.0.1:8001';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': backendTarget,
      '/health': backendTarget,
      '/stream': backendTarget,
    },
  },
});
