import { test, expect } from '@playwright/test'

// ═══════════════════════════════════════════════════════════════════════
// Layer 3: Settings E2E Tests — consolidation, health, scoring weights
// ═══════════════════════════════════════════════════════════════════════

test.describe('Settings', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.click('text=Settings')
    await expect(page.locator('h2', { hasText: 'Settings' })).toBeVisible({ timeout: 5000 })
  })

  test('scoring weights section renders sliders', async ({ page }) => {
    await expect(page.locator('text=Scoring Weights').first()).toBeVisible({ timeout: 10000 })
    // Should have sliders for scoring components
    const sliders = page.locator('input[type="range"]')
    await expect(sliders.first()).toBeVisible({ timeout: 10000 })
    const count = await sliders.count()
    expect(count).toBeGreaterThanOrEqual(8)
  })

  test('consolidation section shows all 5 cycles', async ({ page }) => {
    await expect(page.locator('text=Consolidation Cycles')).toBeVisible({ timeout: 10000 })
    await expect(page.locator('text=decay')).toBeVisible()
    await expect(page.locator('text=creative')).toBeVisible()
    await expect(page.locator('text=cluster')).toBeVisible()
    await expect(page.locator('text=summarize')).toBeVisible()
    await expect(page.locator('text=forget')).toBeVisible()
  })

  test('consolidation Run buttons trigger API calls', async ({ page }) => {
    await expect(page.locator('text=Consolidation Cycles')).toBeVisible({ timeout: 10000 })

    // Find the first Run button
    const runBtn = page.locator('button', { hasText: 'Run' }).first()
    await expect(runBtn).toBeVisible()
    await expect(runBtn).toBeEnabled()

    const responsePromise = page.waitForResponse(
      (res) => res.url().includes('/api/consolidation/') && res.request().method() === 'POST',
    )

    await runBtn.click()
    const response = await responsePromise
    expect(response.ok()).toBe(true)
  })

  test('health section shows component statuses', async ({ page }) => {
    await expect(page.locator('text=System Health')).toBeVisible({ timeout: 10000 })
    await expect(page.locator('text=Storage')).toBeVisible()
    await expect(page.locator('text=Vector Index')).toBeVisible()
    await expect(page.locator('text=Graph Engine')).toBeVisible()
    await expect(page.locator('text=Embeddings')).toBeVisible()
    // All should be healthy
    const healthBadges = page.locator('text=Healthy')
    await expect(healthBadges.first()).toBeVisible()
  })
})
