// Pre-assigned colors for known memory types
const KNOWN_TYPE_COLORS: Record<string, { bg: string; text: string; dot: string; border: string }> = {
  decision: { bg: 'bg-violet-500/10', text: 'text-violet-400', dot: 'bg-violet-400', border: 'border-violet-500/20' },
  pattern: { bg: 'bg-cyan-500/10', text: 'text-cyan-400', dot: 'bg-cyan-400', border: 'border-cyan-500/20' },
  preference: { bg: 'bg-amber-500/10', text: 'text-amber-400', dot: 'bg-amber-400', border: 'border-amber-500/20' },
  style: { bg: 'bg-pink-500/10', text: 'text-pink-400', dot: 'bg-pink-400', border: 'border-pink-500/20' },
  habit: { bg: 'bg-emerald-500/10', text: 'text-emerald-400', dot: 'bg-emerald-400', border: 'border-emerald-500/20' },
  insight: { bg: 'bg-blue-500/10', text: 'text-blue-400', dot: 'bg-blue-400', border: 'border-blue-500/20' },
  context: { bg: 'bg-zinc-500/10', text: 'text-zinc-400', dot: 'bg-zinc-500', border: 'border-zinc-500/20' },
}

// Palette for unknown/agent-created types
const DYNAMIC_PALETTE = [
  { bg: 'bg-rose-500/10', text: 'text-rose-400', dot: 'bg-rose-400', border: 'border-rose-500/20' },
  { bg: 'bg-orange-500/10', text: 'text-orange-400', dot: 'bg-orange-400', border: 'border-orange-500/20' },
  { bg: 'bg-lime-500/10', text: 'text-lime-400', dot: 'bg-lime-400', border: 'border-lime-500/20' },
  { bg: 'bg-teal-500/10', text: 'text-teal-400', dot: 'bg-teal-400', border: 'border-teal-500/20' },
  { bg: 'bg-indigo-500/10', text: 'text-indigo-400', dot: 'bg-indigo-400', border: 'border-indigo-500/20' },
  { bg: 'bg-fuchsia-500/10', text: 'text-fuchsia-400', dot: 'bg-fuchsia-400', border: 'border-fuchsia-500/20' },
  { bg: 'bg-sky-500/10', text: 'text-sky-400', dot: 'bg-sky-400', border: 'border-sky-500/20' },
  { bg: 'bg-yellow-500/10', text: 'text-yellow-400', dot: 'bg-yellow-400', border: 'border-yellow-500/20' },
]

const dynamicCache = new Map<string, (typeof DYNAMIC_PALETTE)[number]>()

/** Get consistent colors for any memory type — known types get pre-assigned colors, unknown get stable random ones. */
export function getTypeColors(type: string) {
  if (KNOWN_TYPE_COLORS[type]) return KNOWN_TYPE_COLORS[type]
  if (dynamicCache.has(type)) return dynamicCache.get(type)!
  // Stable hash-based index
  let hash = 0
  for (let i = 0; i < type.length; i++) hash = ((hash << 5) - hash + type.charCodeAt(i)) | 0
  const entry = DYNAMIC_PALETTE[Math.abs(hash) % DYNAMIC_PALETTE.length]
  dynamicCache.set(type, entry)
  return entry
}

/** All known type names (for filter pills etc.) */
export const KNOWN_TYPES = Object.keys(KNOWN_TYPE_COLORS)
