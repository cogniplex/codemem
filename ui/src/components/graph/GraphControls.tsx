import { Search, SlidersHorizontal } from 'lucide-react'

const NODE_KINDS = [
  'function', 'method', 'class', 'file', 'module', 'package',
  'variable', 'type', 'interface', 'trait', 'struct', 'enum',
  'memory', 'constant', 'endpoint', 'test',
]

const KIND_COLORS: Record<string, string> = {
  function: '#8b5cf6',
  method: '#a78bfa',
  class: '#06b6d4',
  file: '#10b981',
  module: '#f59e0b',
  package: '#d97706',
  variable: '#ef4444',
  type: '#3b82f6',
  interface: '#a855f7',
  trait: '#f97316',
  struct: '#14b8a6',
  enum: '#d946ef',
  memory: '#6366f1',
  constant: '#facc15',
  endpoint: '#f43f5e',
  test: '#64748b',
}

interface Props {
  kinds: Set<string>
  onToggleKind: (kind: string) => void
  maxNodes: number
  onMaxNodesChange: (n: number) => void
  showCommunities: boolean
  onToggleCommunities: () => void
  showEdges: boolean
  onToggleEdges: () => void
  searchLabel: string
  onSearchChange: (v: string) => void
}

export function GraphControls({
  kinds,
  onToggleKind,
  maxNodes,
  onMaxNodesChange,
  showCommunities,
  onToggleCommunities,
  showEdges,
  onToggleEdges,
  searchLabel,
  onSearchChange,
}: Props) {
  return (
    <div className="absolute left-4 top-4 z-10 flex w-60 flex-col gap-3 rounded-lg border border-zinc-800 bg-zinc-900/95 p-4 shadow-xl backdrop-blur-sm">
      <div className="flex items-center gap-2 text-sm font-medium text-zinc-300">
        <SlidersHorizontal size={14} />
        Controls
      </div>

      {/* Search */}
      <div className="relative">
        <Search size={14} className="absolute left-2.5 top-2.5 text-zinc-500" />
        <input
          type="text"
          value={searchLabel}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Find node..."
          className="w-full rounded-md border border-zinc-700 bg-zinc-800 py-1.5 pl-8 pr-3 text-sm text-zinc-200 placeholder-zinc-500 focus:border-violet-500 focus:outline-none"
        />
      </div>

      {/* Max nodes slider */}
      <div>
        <label className="mb-1 flex items-center justify-between text-xs text-zinc-400">
          <span>Max nodes</span>
          <span className="tabular-nums text-zinc-300">{maxNodes}</span>
        </label>
        <input
          type="range"
          min={50}
          max={3000}
          step={50}
          value={maxNodes}
          onChange={(e) => onMaxNodesChange(Number(e.target.value))}
          className="w-full accent-violet-500"
        />
      </div>

      {/* Toggles */}
      <div className="flex flex-col gap-1.5">
        <label className="flex cursor-pointer items-center gap-2 text-sm text-zinc-300">
          <input
            type="checkbox"
            checked={showEdges}
            onChange={onToggleEdges}
            className="rounded border-zinc-600 bg-zinc-800 accent-violet-500"
          />
          Show edges
        </label>
        <label className="flex cursor-pointer items-center gap-2 text-sm text-zinc-300">
          <input
            type="checkbox"
            checked={showCommunities}
            onChange={onToggleCommunities}
            className="rounded border-zinc-600 bg-zinc-800 accent-violet-500"
          />
          Show communities
        </label>
      </div>

      {/* Kind filters */}
      <div>
        <p className="mb-1.5 text-xs font-medium text-zinc-400">Node kinds</p>
        <div className="flex max-h-44 flex-col gap-1 overflow-y-auto">
          {NODE_KINDS.map((kind) => (
            <label
              key={kind}
              className="flex cursor-pointer items-center gap-2 text-xs text-zinc-300"
            >
              <input
                type="checkbox"
                checked={kinds.has(kind)}
                onChange={() => onToggleKind(kind)}
                className="rounded border-zinc-600 bg-zinc-800 accent-violet-500"
              />
              <span
                className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ backgroundColor: KIND_COLORS[kind] ?? '#71717a' }}
              />
              {kind}
            </label>
          ))}
        </div>
      </div>
    </div>
  )
}
