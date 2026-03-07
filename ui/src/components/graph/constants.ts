export const EDGE_COLORS: Record<string, string> = {
  // Structural — blue/cyan
  CONTAINS: '#3b82f670', PART_OF: '#3b82f670', IMPORTS: '#06b6d4b0',
  // Execution
  CALLS: '#52525b50', EXTENDS: '#34d39990', IMPLEMENTS: '#22c55eb0', INHERITS: '#4ade8090',
  // Semantic — purple/violet
  RELATES_TO: '#8b5cf690', SIMILAR_TO: '#a78bfa90', SHARES_THEME: '#c084fc90',
  EXPLAINS: '#a78bfa90', EXEMPLIFIES: '#c084fc90', SUMMARIZES: '#8b5cf690',
  // Temporal — amber/orange
  LEADS_TO: '#f59e0b90', PRECEDED_BY: '#fbbf2490', EVOLVED_INTO: '#fb923c90', DERIVED_FROM: '#f9731690',
  // Dependency — teal
  DEPENDS_ON: '#14b8a690',
  // Negative — red/rose
  CONTRADICTS: '#ef4444a0', INVALIDATED_BY: '#f87171a0', BLOCKS: '#dc2626a0', SUPERSEDES: '#fb7185a0',
  // Reinforcement — lime
  REINFORCES: '#a3e635a0',
  // Co-change
  CO_CHANGED: '#94a3b840',
}

export const ALL_RELATIONSHIPS = new Set(Object.keys(EDGE_COLORS))

export const KIND_COLORS: Record<string, string> = {
  function: '#8b5cf6',   // violet-500
  method: '#a78bfa',     // violet-400
  class: '#06b6d4',      // cyan-500
  file: '#10b981',       // emerald-500
  module: '#f59e0b',     // amber-500
  package: '#d97706',    // amber-600
  variable: '#ef4444',   // red-500
  type: '#3b82f6',       // blue-500
  interface: '#a855f7',  // purple-500
  trait: '#f97316',      // orange-500
  struct: '#14b8a6',     // teal-500
  enum: '#d946ef',       // fuchsia-500
  memory: '#6366f1',     // indigo-500
  constant: '#facc15',   // yellow-400
  endpoint: '#f43f5e',   // rose-500
  test: '#64748b',       // slate-500
}
