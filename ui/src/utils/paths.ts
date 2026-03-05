/**
 * Trim absolute paths for display. Shows just relative paths or
 * the last few components of long paths.
 */

/** Shorten a label for display. For absolute paths, show the last 2-3 components. */
export function trimLabel(label: string): string {
  if (!label) return label

  if (label.startsWith('/')) {
    const parts = label.split('/')
    if (parts.length > 3) {
      return parts.slice(-3).join('/')
    }
  }

  return label
}
