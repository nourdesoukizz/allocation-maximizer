import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Avoid requiring @types/node by declaring process as any for TS
declare const process: any;

export default defineConfig({
  // Allow building under a subpath (e.g., /scx/) by setting VITE_BASE
  // Example: VITE_BASE=/scx/ npm run build
  base: (process?.env?.VITE_BASE as string) || '/',
  plugins: [react()],
  server: {
    port: 3000,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'http://localhost:8004',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api/, ''),
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.log('proxy error', err);
          });
          proxy.on('proxyReq', (proxyReq, req, res) => {
            console.log('Sending Request to the Target:', req.method, req.url);
          });
          proxy.on('proxyRes', (proxyRes, req, res) => {
            console.log('Received Response from the Target:', proxyRes.statusCode, req.url);
          });
        },
      },
    },
  },
});
