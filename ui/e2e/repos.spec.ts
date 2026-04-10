import { test, expect } from '@playwright/test'

// ═══════════════════════════════════════════════════════════════════════
// Layer 1: Repos API Tests — the Repos page was removed from the UI,
// so we only test the REST API endpoints
// ═══════════════════════════════════════════════════════════════════════

const API = 'http://localhost:4242'

test.describe('Repos API', () => {
  test('GET /api/repos returns list', async ({ request }) => {
    const res = await request.get(`${API}/api/repos`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(Array.isArray(data)).toBe(true)
  })
})
