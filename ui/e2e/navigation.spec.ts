import { test, expect } from '@playwright/test'

// ═══════════════════════════════════════════════════════════════════════
// Layer 2: UI Navigation & Layout Tests — verify pages load and render
// ═══════════════════════════════════════════════════════════════════════

test.describe('Shell Layout', () => {
  test('page loads with navbar and main content', async ({ page }) => {
    await page.goto('/')
    const nav = page.locator('nav')
    await expect(nav).toBeVisible({ timeout: 10000 })
    // Check brand name in the navbar
    await expect(nav.locator('text=codemem')).toBeVisible()
    // Check all nav tab labels exist
    for (const label of ['Dashboard', 'Graph', 'Memories']) {
      await expect(nav.locator(`button`, { hasText: label })).toBeVisible()
    }
    // Main content area exists
    await expect(page.locator('main')).toBeVisible()
  })

  test('namespace picker is visible', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('nav')).toBeVisible({ timeout: 10000 })
    // Namespace dropdown button shows "All namespaces" by default
    await expect(page.locator('nav button', { hasText: 'All namespaces' })).toBeVisible()
  })

  test('search button opens search modal', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('nav')).toBeVisible({ timeout: 10000 })
    const searchBtn = page.locator('nav button', { hasText: 'Search' })
    await expect(searchBtn).toBeVisible()
    await searchBtn.click()
    // Search modal should appear with an input
    await expect(page.locator('input[placeholder*="Search"]').or(page.locator('[role="dialog"] input'))).toBeVisible({ timeout: 5000 })
  })
})

test.describe('Navigation between views', () => {
  test('clicking nav tabs switches views', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('nav')).toBeVisible({ timeout: 10000 })

    // Dashboard is default — should show metric cards
    await expect(page.locator('text=Memories').first()).toBeVisible({ timeout: 10000 })

    // Navigate to Memories
    await page.locator('nav button', { hasText: 'Memories' }).click()
    // Wait for memory browser heading
    await expect(page.locator('h2', { hasText: 'Memories' })).toBeVisible({ timeout: 5000 })

    // Navigate to Graph
    await page.locator('nav button', { hasText: 'Graph' }).click()
    // Graph view should show loading state or graph content
    await expect(
      page.locator('text=Loading graph')
        .or(page.locator('text=No graph nodes'))
        .or(page.locator('canvas').first())
    ).toBeVisible({ timeout: 10000 })

    // Back to Dashboard
    await page.locator('nav button', { hasText: 'Dashboard' }).click()
    await expect(page.locator('text=Recent Activity')).toBeVisible({ timeout: 5000 })
  })
})
