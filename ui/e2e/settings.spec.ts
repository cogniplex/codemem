import { test, expect } from '@playwright/test'

// ═══════════════════════════════════════════════════════════════════════
// Layer 3: Consolidation E2E Tests — consolidation is now on the Dashboard
// (Settings page no longer exists as a separate view)
// ═══════════════════════════════════════════════════════════════════════

test.describe('Consolidation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('nav')).toBeVisible({ timeout: 10000 })
  })

  test('consolidation section on dashboard shows cycle names', async ({ page }) => {
    await expect(page.locator('text=Consolidation')).toBeVisible({ timeout: 10000 })
    // Each cycle should be listed
    for (const cycle of ['decay', 'creative', 'cluster', 'summarize', 'forget']) {
      await expect(page.locator(`text=${cycle}`).first()).toBeVisible({ timeout: 5000 })
    }
  })

  test('consolidation Run buttons trigger API calls', async ({ page }) => {
    test.setTimeout(60000)
    await expect(page.locator('text=Consolidation')).toBeVisible({ timeout: 10000 })

    const runBtn = page.locator('button', { hasText: 'Run' }).first()
    await expect(runBtn).toBeVisible({ timeout: 10000 })
    await expect(runBtn).toBeEnabled()

    // Set up listener before clicking
    const responsePromise = page.waitForResponse(
      (res) => res.url().includes('/api/consolidation/') && res.request().method() === 'POST',
    )

    await runBtn.click()
    const response = await responsePromise
    expect(response.ok()).toBe(true)
  })
})
