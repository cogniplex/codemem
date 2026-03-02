import { test, expect } from '@playwright/test'

// ═══════════════════════════════════════════════════════════════════════
// Layer 3: Agent Runner E2E Tests — recipe selection and execution
// ═══════════════════════════════════════════════════════════════════════

test.describe('Agent Runner', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.locator('aside').locator('text=Agents').click()
    await expect(page.locator('text=Agent Runner')).toBeVisible({ timeout: 5000 })
  })

  test('recipe selector loads recipes', async ({ page }) => {
    // Find the recipe select inside the main content area (not the header namespace select)
    const main = page.locator('main')
    const recipeSelect = main.locator('select').first()
    await expect(recipeSelect).toBeVisible()

    // Wait for recipes to load — 5 options: 1 placeholder + 4 recipes
    await expect(recipeSelect.locator('option')).toHaveCount(5, { timeout: 10000 })
  })

  test('selecting recipe shows description and steps', async ({ page }) => {
    const main = page.locator('main')
    const recipeSelect = main.locator('select').first()
    // Wait for recipes to load
    await expect(recipeSelect.locator('option')).not.toHaveCount(1, { timeout: 10000 })

    await recipeSelect.selectOption({ index: 1 }) // Pick first real recipe

    // Should show step pills (tool names)
    await expect(
      page
        .locator('text=index_codebase')
        .or(page.locator('text=get_pagerank'))
        .or(page.locator('text=consolidate_decay'))
    ).toBeVisible({ timeout: 5000 })
  })

  test('Run button is disabled without recipe selection', async ({ page }) => {
    const runBtn = page.locator('button', { hasText: 'Run' }).first()
    await expect(runBtn).toBeVisible()
    await expect(runBtn).toBeDisabled()
  })

  test('Run button sends SSE request', async ({ page }) => {
    const main = page.locator('main')
    const recipeSelect = main.locator('select').first()
    // Wait for recipes to load
    await expect(recipeSelect.locator('option')).not.toHaveCount(1, { timeout: 10000 })

    // Select graph-analysis recipe (no repo needed)
    await recipeSelect.selectOption('graph-analysis')

    const runBtn = page.locator('button', { hasText: 'Run' }).first()
    await expect(runBtn).toBeEnabled()

    // Intercept the POST to verify it fires
    const responsePromise = page.waitForResponse(
      (res) => res.url().includes('/api/agents/run') && res.request().method() === 'POST',
    )

    await runBtn.click()
    const response = await responsePromise
    // SSE response should be 200
    expect(response.status()).toBe(200)
  })
})
