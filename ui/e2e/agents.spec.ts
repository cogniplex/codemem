import { test, expect } from '@playwright/test'

// ═══════════════════════════════════════════════════════════════════════
// Layer 1: Agent API Tests — the Agents page was removed from the UI,
// so we only test the REST API endpoints
// ═══════════════════════════════════════════════════════════════════════

const API = 'http://localhost:4242'

test.describe('Agent API', () => {
  test('GET /api/agents/recipes returns recipe list', async ({ request }) => {
    const res = await request.get(`${API}/api/agents/recipes`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(Array.isArray(data)).toBe(true)
    expect(data.length).toBeGreaterThan(0)
    const first = data[0]
    expect(first).toHaveProperty('id')
    expect(first).toHaveProperty('name')
    expect(first).toHaveProperty('steps')
  })
})
