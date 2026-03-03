import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  timeout: 30000,
  retries: 1,
  use: {
    baseURL: 'http://localhost:3000',
    headless: true,
    screenshot: 'only-on-failure',
  },
  webServer: [
    {
      command: './target/release/codemem serve --api --port 4242',
      port: 4242,
      timeout: 30000,
      reuseExistingServer: true,
      cwd: '..',
    },
    {
      command: 'npx vite --port 3000',
      port: 3000,
      timeout: 15000,
      reuseExistingServer: true,
    },
  ],
  projects: [
    { name: 'chromium', use: { browserName: 'chromium' } },
  ],
})
