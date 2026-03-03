export const EDGE_COLORS: Record<string, string> = {
  // Structural — blue/cyan
  CONTAINS: '#3b82f650', PART_OF: '#3b82f650', IMPORTS: '#06b6d4a0',
  // Execution — subtle so CALLS (97% of edges) doesn't blind you
  CALLS: '#52525b30', EXTENDS: '#34d39960', IMPLEMENTS: '#22c55ea0', INHERITS: '#4ade8060',
  // Semantic — purple/violet
  RELATES_TO: '#8b5cf660', SIMILAR_TO: '#a78bfa60', SHARES_THEME: '#c084fc60',
  EXPLAINS: '#a78bfa60', EXEMPLIFIES: '#c084fc60', SUMMARIZES: '#8b5cf660',
  // Temporal — amber/orange
  LEADS_TO: '#f59e0b60', PRECEDED_BY: '#fbbf2460', EVOLVED_INTO: '#fb923c60', DERIVED_FROM: '#f9731660',
  // Dependency — teal
  DEPENDS_ON: '#14b8a660',
  // Negative — red/rose
  CONTRADICTS: '#ef444480', INVALIDATED_BY: '#f8717180', BLOCKS: '#dc262680', SUPERSEDES: '#fb718580',
  // Reinforcement — lime
  REINFORCES: '#a3e63580',
  // Co-change
  CO_CHANGED: '#94a3b860',
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
