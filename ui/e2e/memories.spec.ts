import { test, expect } from '@playwright/test'

// ═══════════════════════════════════════════════════════════════════════
// Layer 3: Memory Browser E2E Tests — list, filter, search, detail panel
// ═══════════════════════════════════════════════════════════════════════

test.describe('Memory Browser', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('nav')).toBeVisible({ timeout: 10000 })
    await page.locator('nav button', { hasText: 'Memories' }).click()
    await expect(page.locator('h2', { hasText: 'Memories' })).toBeVisible({ timeout: 5000 })
  })

  test('shows memory list or empty state', async ({ page }) => {
    // Wait for loading to finish — skeletons have animate-pulse class
    await expect(page.locator('.animate-pulse').first()).not.toBeVisible({ timeout: 15000 }).catch(() => {})
    // Now check: either memory card buttons or the empty state icon exist
    const cards = await page.locator('.space-y-2 button').count()
    const empty = await page.locator('text=No memories yet').or(page.locator('text=No results found')).count()
    expect(cards + empty).toBeGreaterThan(0)
  })

  test('search input exists', async ({ page }) => {
    await expect(page.locator('input[placeholder*="Search memories"]')).toBeVisible()
  })

  test('type filter pills exist', async ({ page }) => {
    // "All" filter pill should be visible
    await expect(page.locator('button', { hasText: 'All' }).first()).toBeVisible()
    // At least one type filter pill exists
    await expect(page.getByRole('button', { name: 'decision' })).toBeVisible({ timeout: 5000 })
  })

  test('pagination controls exist when memories present', async ({ page }) => {
    const hasMemories = await page.locator('button.w-full.rounded-xl').count()
    if (hasMemories > 0) {
      // Pagination shows page numbers
      await expect(page.locator('text=/\\d+ \\/ \\d+/')).toBeVisible({ timeout: 5000 })
    }
  })
})
