import { test, expect } from '@playwright/test'

// ═══════════════════════════════════════════════════════════════════════
// Layer 1: API Integration Tests — verify REST endpoints return correct shapes
// ═══════════════════════════════════════════════════════════════════════

const API = 'http://localhost:4242'

test.describe('API: Stats & Health', () => {
  test('GET /api/stats returns counts', async ({ request }) => {
    const res = await request.get(`${API}/api/stats`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(data).toHaveProperty('memory_count')
    expect(data).toHaveProperty('node_count')
    expect(data).toHaveProperty('edge_count')
    expect(data).toHaveProperty('session_count')
    expect(data).toHaveProperty('namespace_count')
    expect(typeof data.memory_count).toBe('number')
  })

  test('GET /api/health returns component statuses', async ({ request }) => {
    const res = await request.get(`${API}/api/health`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(data.storage).toHaveProperty('status')
    expect(data.vector).toHaveProperty('status')
    expect(data.graph).toHaveProperty('status')
    expect(data.embeddings).toHaveProperty('status')
  })

  test('GET /api/metrics returns metrics', async ({ request }) => {
    const res = await request.get(`${API}/api/metrics`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(data).toHaveProperty('tool_calls_total')
    expect(data).toHaveProperty('latency_percentiles')
  })
})

test.describe('API: Memories', () => {
  test('GET /api/memories returns paginated list', async ({ request }) => {
    const res = await request.get(`${API}/api/memories?limit=5`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(data).toHaveProperty('memories')
    expect(data).toHaveProperty('total')
    expect(data).toHaveProperty('offset')
    expect(data).toHaveProperty('limit')
    expect(Array.isArray(data.memories)).toBe(true)
    expect(data.limit).toBeLessThanOrEqual(5)
  })

  test('POST + GET + DELETE memory lifecycle', async ({ request }) => {
    // Create
    const createRes = await request.post(`${API}/api/memories`, {
      data: {
        content: 'Playwright test memory',
        memory_type: 'context',
        importance: 0.5,
        tags: ['test'],
      },
    })
    expect(createRes.status()).toBe(201)
    const { id } = await createRes.json()
    expect(id).toBeTruthy()

    // Read
    const getRes = await request.get(`${API}/api/memories/${id}`)
    expect(getRes.ok()).toBe(true)
    const memory = await getRes.json()
    expect(memory.content).toBe('Playwright test memory')
    expect(memory.memory_type).toBe('context')

    // Delete
    const delRes = await request.delete(`${API}/api/memories/${id}`)
    expect(delRes.ok()).toBe(true)

    // Confirm deleted
    const gone = await request.get(`${API}/api/memories/${id}`)
    expect(gone.status()).toBe(404)
  })

  test('GET /api/search returns scored results', async ({ request }) => {
    const res = await request.get(`${API}/api/search?q=test&k=5`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(data).toHaveProperty('results')
    expect(data).toHaveProperty('query')
    expect(data.query).toBe('test')
  })
})

test.describe('API: Graph', () => {
  test('GET /api/graph/subgraph returns nodes and edges', async ({ request }) => {
    const res = await request.get(`${API}/api/graph/subgraph?max_nodes=50`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(data).toHaveProperty('nodes')
    expect(data).toHaveProperty('edges')
    expect(Array.isArray(data.nodes)).toBe(true)
    expect(Array.isArray(data.edges)).toBe(true)
  })

  test('GET /api/graph/communities returns mapping', async ({ request }) => {
    const res = await request.get(`${API}/api/graph/communities`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(data).toHaveProperty('communities')
    expect(data).toHaveProperty('num_communities')
  })

  test('GET /api/graph/pagerank returns scores', async ({ request }) => {
    test.setTimeout(60000)
    const res = await request.get(`${API}/api/graph/pagerank?top=5`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(data).toHaveProperty('scores')
    expect(Array.isArray(data.scores)).toBe(true)
  })
})

test.describe('API: Repos', () => {
  test('GET /api/repos returns list', async ({ request }) => {
    const res = await request.get(`${API}/api/repos`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(Array.isArray(data)).toBe(true)
  })
})

test.describe('API: Namespaces', () => {
  test('GET /api/namespaces returns list', async ({ request }) => {
    const res = await request.get(`${API}/api/namespaces`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(Array.isArray(data)).toBe(true)
  })
})

test.describe('API: Sessions', () => {
  test('GET /api/sessions returns list', async ({ request }) => {
    const res = await request.get(`${API}/api/sessions`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(Array.isArray(data)).toBe(true)
  })
})

test.describe('API: Timeline & Distribution', () => {
  test('GET /api/timeline returns buckets', async ({ request }) => {
    const res = await request.get(`${API}/api/timeline`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(Array.isArray(data)).toBe(true)
  })

  test('GET /api/distribution returns type counts', async ({ request }) => {
    const res = await request.get(`${API}/api/distribution`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(data).toHaveProperty('type_counts')
    expect(data).toHaveProperty('total')
  })
})

test.describe('API: Consolidation', () => {
  test('GET /api/consolidation/status returns all cycles', async ({ request }) => {
    const res = await request.get(`${API}/api/consolidation/status`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(data).toHaveProperty('cycles')
    expect(data.cycles.length).toBeGreaterThanOrEqual(5)
    const names = data.cycles.map((c: { cycle: string }) => c.cycle)
    expect(names).toContain('decay')
    expect(names).toContain('creative')
    expect(names).toContain('cluster')
    expect(names).toContain('summarize')
    expect(names).toContain('forget')
  })

  test('POST /api/consolidation/decay runs successfully', async ({ request }) => {
    const res = await request.post(`${API}/api/consolidation/decay`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(data).toHaveProperty('message')
  })

  test('POST /api/consolidation/invalid returns 400', async ({ request }) => {
    const res = await request.post(`${API}/api/consolidation/invalid`)
    expect(res.status()).toBe(400)
  })
})

test.describe('API: Agent Recipes', () => {
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

test.describe('API: Config', () => {
  test('GET /api/config returns config object', async ({ request }) => {
    const res = await request.get(`${API}/api/config`)
    expect(res.ok()).toBe(true)
    const data = await res.json()
    expect(typeof data).toBe('object')
  })
})
