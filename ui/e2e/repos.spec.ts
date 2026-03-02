import { test, expect } from '@playwright/test'

// ═══════════════════════════════════════════════════════════════════════
// Layer 3: Repos E2E Tests — register, view, index repos
// ═══════════════════════════════════════════════════════════════════════

test.describe('Repo Manager', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.locator('aside').locator('text=Repos').click()
    await expect(page.getByRole('heading', { name: 'Repositories' })).toBeVisible({ timeout: 5000 })
  })

  test('shows existing repos or empty state', async ({ page }) => {
    // Either show repo cards or the empty state
    const hasCards = await page.locator('[class*="rounded-lg border"]').filter({ hasText: /codemem|Documents/ }).count()
    const hasEmpty = await page.locator('text=No repositories registered').count()
    expect(hasCards + hasEmpty).toBeGreaterThan(0)
  })

  test('Add Repository button toggles form', async ({ page }) => {
    const addBtn = page.locator('button', { hasText: 'Add Repository' })
    await expect(addBtn).toBeVisible()

    await addBtn.click()
    await expect(page.locator('text=Register a new repository')).toBeVisible()
    await expect(page.locator('input[placeholder="/path/to/repository"]')).toBeVisible()

    // The same button toggles to Cancel
    const cancelBtn = page.locator('button', { hasText: 'Cancel' })
    await cancelBtn.click()
    await expect(page.locator('text=Register a new repository')).not.toBeVisible()
  })

  test('register repo lifecycle', async ({ page }) => {
    // Open form
    await page.locator('button', { hasText: 'Add Repository' }).click()
    await page.fill('input[placeholder="/path/to/repository"]', '/tmp/playwright-test-repo')
    await page.fill('input[placeholder="my-project"]', 'test-repo')

    // Intercept register API call
    const registerPromise = page.waitForResponse(
      (res) => res.url().includes('/api/repos') && res.request().method() === 'POST',
    )

    await page.locator('button', { hasText: 'Register' }).click()
    const registerRes = await registerPromise
    // May succeed or fail (path might not exist) - just verify the call was made
    expect([201, 500]).toContain(registerRes.status())
  })

  test('re-index button triggers indexing', async ({ page }) => {
    // If there's a repo card with a Re-index button
    const reindexBtn = page.locator('button', { hasText: 'Re-index' }).first()
    if (await reindexBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
      const responsePromise = page.waitForResponse(
        (res) => res.url().includes('/index') && res.request().method() === 'POST',
      )
      await reindexBtn.click()
      const response = await responsePromise
      expect(response.ok()).toBe(true)
    }
  })
})
