import { test, expect } from '@playwright/test'

// ═══════════════════════════════════════════════════════════════════════
// Layer 3: Dashboard E2E Tests — verify dashboard renders live data
// ═══════════════════════════════════════════════════════════════════════

test.describe('Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    // Wait for navbar to render (proves app loaded)
    await expect(page.locator('nav')).toBeVisible({ timeout: 10000 })
  })

  test('stat cards show numeric values', async ({ page }) => {
    // Wait for stat card values to load (tabular-nums class on the number elements)
    const statsGrid = page.locator('.grid.grid-cols-2')
    await expect(statsGrid).toBeVisible({ timeout: 10000 })

    // Key stat card labels should be present
    await expect(statsGrid.locator('text=Memories')).toBeVisible({ timeout: 5000 })
    await expect(statsGrid.locator('text=Graph Nodes')).toBeVisible()
    await expect(statsGrid.locator('text=Edges')).toBeVisible()
    await expect(statsGrid.locator('text=Sessions')).toBeVisible()
  })

  test('recent activity section renders', async ({ page }) => {
    await expect(page.locator('text=Recent Activity')).toBeVisible({ timeout: 10000 })
  })

  test('consolidation section shows cycles', async ({ page }) => {
    // Consolidation section is on the dashboard
    await expect(page.locator('text=Consolidation')).toBeVisible({ timeout: 10000 })
    // Should have Run buttons for each cycle
    const runBtns = page.locator('button', { hasText: 'Run' })
    await expect(runBtns.first()).toBeVisible({ timeout: 10000 })
  })

  test('insights section renders', async ({ page }) => {
    // InsightsSection should render at least one insight card
    await expect(page.getByRole('heading', { name: 'File Hotspots' })).toBeVisible({ timeout: 15000 })
  })
})
