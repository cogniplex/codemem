/**
 * Trim absolute paths for display. Strips workspace prefix from labels
 * and shows just relative paths or directory names.
 */

/** Strip leading path prefix from a label, showing relative path */
export function trimLabel(label: string, namespace?: string | null): string {
  if (!label) return label

  // If namespace is a path and label starts with it, strip the prefix
  if (namespace && label.startsWith(namespace)) {
    const stripped = label.slice(namespace.length).replace(/^\//, '')
    if (stripped) return stripped
  }

  // For any absolute path, show just the last 2-3 components
  if (label.startsWith('/')) {
    const parts = label.split('/')
    if (parts.length > 3) {
      return parts.slice(-3).join('/')
    }
  }

  return label
}

/** Show just the directory name for a namespace path */
export function trimNamespace(ns: string): string {
  if (!ns) return ns
  if (ns.startsWith('/')) {
    const parts = ns.split('/').filter(Boolean)
    return parts[parts.length - 1] ?? ns
  }
  return ns
}
