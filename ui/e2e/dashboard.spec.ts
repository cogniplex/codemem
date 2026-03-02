import { test, expect } from '@playwright/test'

// ═══════════════════════════════════════════════════════════════════════
// Layer 3: Dashboard E2E Tests — verify dashboard renders live data
// ═══════════════════════════════════════════════════════════════════════

test.describe('Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    // Wait for sidebar to render (proves app loaded)
    await expect(page.locator('aside')).toBeVisible({ timeout: 10000 })
  })

  test('stat cards show numeric values', async ({ page }) => {
    // Wait for stat card numbers to load (not just labels)
    // The stat card numbers are inside <p> tags with class containing "tabular-nums"
    await expect(page.locator('p.tabular-nums').first()).toBeVisible({ timeout: 10000 })

    // Wait for actual data — numbers should be present
    await expect(page.locator('p.tabular-nums').first()).not.toHaveText('', { timeout: 10000 })

    // All 4 stat card labels should be present in the stat cards grid
    const statsGrid = page.locator('.grid.grid-cols-2')
    await expect(statsGrid.locator('text=Memories')).toBeVisible({ timeout: 5000 })
    await expect(statsGrid.locator('text=Graph Nodes')).toBeVisible()
    await expect(statsGrid.locator('text=Edges')).toBeVisible()
    await expect(statsGrid.locator('text=Sessions')).toBeVisible()
  })

  test('recent activity shows memory entries', async ({ page }) => {
    await expect(page.locator('text=Recent Activity')).toBeVisible({ timeout: 10000 })

    // Should have at least one memory entry with a type badge
    const typeBadges = page.locator('text=context').or(page.locator('text=insight'))
    await expect(typeBadges.first()).toBeVisible({ timeout: 10000 })
  })

  test('type distribution chart renders', async ({ page }) => {
    await expect(page.locator('text=Type Distribution')).toBeVisible({ timeout: 10000 })
    // The PieChart SVG is inside the card container, not inside the header
    // Go up to the card (rounded-lg border) and find any SVG descendant
    const card = page.locator('text=Type Distribution').locator('xpath=ancestor::div[contains(@class,"rounded-lg")]')
    await expect(card.locator('svg').first()).toBeVisible({ timeout: 15000 })
  })

  test('quick action buttons are present', async ({ page }) => {
    await expect(page.locator('button', { hasText: 'Index Repo' })).toBeVisible({ timeout: 10000 })
    await expect(page.locator('button', { hasText: 'Run Consolidation' })).toBeVisible()
  })

  test('Run Consolidation button triggers mutation', async ({ page }) => {
    const btn = page.locator('button', { hasText: 'Run Consolidation' })
    await expect(btn).toBeVisible({ timeout: 10000 })
    await expect(btn).toBeEnabled()

    // Intercept the API call
    const responsePromise = page.waitForResponse(
      (res) => res.url().includes('/api/consolidation/') && res.request().method() === 'POST',
    )

    await btn.click()
    const response = await responsePromise
    expect(response.ok()).toBe(true)
  })
})
