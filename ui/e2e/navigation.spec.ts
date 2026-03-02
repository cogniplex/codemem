import { test, expect } from '@playwright/test'

// ═══════════════════════════════════════════════════════════════════════
// Layer 2: UI Navigation & Layout Tests — verify pages load and render
// ═══════════════════════════════════════════════════════════════════════

test.describe('Shell Layout', () => {
  test('page loads with sidebar and header', async ({ page }) => {
    await page.goto('/')
    const sidebar = page.locator('aside')
    await expect(sidebar).toBeVisible()
    // Check brand name in the sidebar header
    await expect(sidebar.locator('text=codemem')).toBeVisible()
    // Check all nav items exist in sidebar
    for (const label of ['Dashboard', 'Repos', 'Graph', 'Memories', 'Timeline', 'Agents', 'Settings']) {
      await expect(sidebar.locator(`text=${label}`)).toBeVisible()
    }
    // Header with namespace picker
    await expect(page.locator('header')).toBeVisible()
  })

  test('namespace picker is visible', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('text=Namespace')).toBeVisible()
    await expect(page.locator('header select')).toBeVisible()
  })

  test('sidebar can be collapsed', async ({ page }) => {
    await page.goto('/')
    const sidebar = page.locator('aside')
    await expect(sidebar).toBeVisible()

    // Click the collapse/toggle button in the sidebar header
    const collapseBtn = sidebar.locator('div').first().locator('button')
    await collapseBtn.click()

    // Wait for CSS transition (duration-200 = 200ms)
    await page.waitForTimeout(300)

    // Sidebar should be narrow (w-16 = 64px)
    const width = await sidebar.evaluate((el) => el.offsetWidth)
    expect(width).toBeLessThanOrEqual(80)

    // Nav labels should be hidden
    await expect(sidebar.locator('span:text("Dashboard")')).not.toBeVisible()
  })
})

test.describe('Navigation between views', () => {
  test('clicking sidebar items switches views', async ({ page }) => {
    await page.goto('/')
    const sidebar = page.locator('aside')

    // Dashboard is default
    await expect(page.locator('text=Recent Activity')).toBeVisible({ timeout: 10000 })

    // Navigate to Repos
    await sidebar.locator('text=Repos').click()
    await expect(page.getByRole('heading', { name: 'Repositories' })).toBeVisible({ timeout: 5000 })

    // Navigate to Memories
    await sidebar.locator('text=Memories').click()
    // Wait for memory list or empty state
    await expect(page.locator('table').or(page.locator('text=No memories'))).toBeVisible({ timeout: 5000 })

    // Navigate to Timeline
    await sidebar.locator('text=Timeline').click()
    await expect(page.locator('text=Sessions')).toBeVisible({ timeout: 5000 })

    // Navigate to Agents
    await sidebar.locator('text=Agents').click()
    await expect(page.locator('text=Agent Runner')).toBeVisible({ timeout: 5000 })

    // Navigate to Settings
    await sidebar.locator('text=Settings').click()
    await expect(page.getByRole('heading', { name: 'Scoring Weights' })).toBeVisible({ timeout: 5000 })

    // Back to Dashboard
    await sidebar.locator('text=Dashboard').click()
    await expect(page.locator('text=Recent Activity')).toBeVisible({ timeout: 5000 })
  })
})
