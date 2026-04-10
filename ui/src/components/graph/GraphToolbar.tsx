import { useState, useRef, useEffect } from 'react'
import {
  Search,
  ChevronDown,
  Layers,
  GitBranch,
  SlidersHorizontal,
  PanelLeft,
  PanelLeftClose,
} from 'lucide-react'
import { KIND_COLORS, EDGE_COLORS } from './constants'

interface Props {
  // Search
  searchLabel: string
  onSearchChange: (v: string) => void
  // Node kinds
  kinds: Set<string>
  onToggleKind: (kind: string) => void
  // Max nodes
  maxNodes: number
  onMaxNodesChange: (n: number) => void
  // Toggles
  showCommunities: boolean
  onToggleCommunities: () => void
  showEdges: boolean
  onToggleEdges: () => void
  // Relationships
  activeRelationships: Set<string>
  edgeCounts: Record<string, number>
  onToggleRelationship: (rel: string) => void
  // File tree
  showFileTree: boolean
  onToggleFileTree: () => void
}

const NODE_KINDS = [
  'function', 'method', 'class', 'file', 'module', 'package',
  'type', 'interface', 'memory', 'constant', 'endpoint', 'test',
  'trait', 'enum', 'external',
]

export function GraphToolbar({
  searchLabel, onSearchChange,
  kinds, onToggleKind,
  maxNodes, onMaxNodesChange,
  showCommunities, onToggleCommunities,
  showEdges, onToggleEdges,
  activeRelationships, edgeCounts, onToggleRelationship,
  showFileTree, onToggleFileTree,
}: Props) {
  return (
    <div className="flex h-10 shrink-0 items-center gap-2 border-b border-zinc-800/60 bg-zinc-900/50 px-3">
      {/* File tree toggle */}
      <button
        onClick={onToggleFileTree}
        className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
        title={showFileTree ? 'Hide file tree' : 'Show file tree'}
      >
        {showFileTree ? <PanelLeftClose size={14} /> : <PanelLeft size={14} />}
      </button>

      <div className="mx-1 h-4 w-px bg-zinc-800" />

      {/* Search */}
      <div className="relative">
        <Search size={12} className="absolute left-2 top-1/2 -translate-y-1/2 text-zinc-600" />
        <input
          type="text"
          value={searchLabel}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Find node..."
          className="w-36 rounded-md border border-zinc-800 bg-zinc-950/50 py-1 pl-7 pr-2 text-xs text-zinc-300 placeholder-zinc-600 outline-none transition-colors focus:border-zinc-600 focus:w-48"
        />
      </div>

      {/* Node kinds dropdown */}
      <DropdownFilter
        label="Nodes"
        icon={<Layers size={12} />}
        items={NODE_KINDS.map((k) => ({
          id: k,
          label: k,
          active: kinds.has(k),
          color: KIND_COLORS[k],
        }))}
        onToggle={onToggleKind}
      />

      {/* Edge types dropdown */}
      <DropdownFilter
        label="Edges"
        icon={<GitBranch size={12} />}
        items={Object.entries(edgeCounts)
          .sort((a, b) => b[1] - a[1])
          .map(([rel, count]) => ({
            id: rel,
            label: `${rel} (${count})`,
            active: activeRelationships.has(rel),
            color: (EDGE_COLORS[rel] ?? '#52525b').slice(0, 7),
          }))}
        onToggle={onToggleRelationship}
      />

      <div className="mx-1 h-4 w-px bg-zinc-800" />

      {/* Max nodes */}
      <div className="flex items-center gap-1.5">
        <SlidersHorizontal size={12} className="text-zinc-600" />
        <input
          type="range"
          min={50}
          max={3000}
          step={50}
          value={maxNodes}
          onChange={(e) => onMaxNodesChange(Number(e.target.value))}
          className="w-20 accent-violet-500"
        />
        <span className="min-w-[2.5rem] text-[10px] tabular-nums text-zinc-500">{maxNodes}</span>
      </div>

      <div className="mx-1 h-4 w-px bg-zinc-800" />

      {/* Toggles */}
      <TogglePill label="Edges" active={showEdges} onClick={onToggleEdges} />
      <TogglePill label="Communities" active={showCommunities} onClick={onToggleCommunities} />
    </div>
  )
}

function TogglePill({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md px-2 py-0.5 text-[11px] font-medium transition-colors ${
        active
          ? 'bg-violet-500/15 text-violet-400'
          : 'text-zinc-600 hover:text-zinc-400'
      }`}
    >
      {label}
    </button>
  )
}

function DropdownFilter({
  label,
  icon,
  items,
  onToggle,
}: {
  label: string
  icon: React.ReactNode
  items: { id: string; label: string; active: boolean; color?: string }[]
  onToggle: (id: string) => void
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  const activeCount = items.filter((i) => i.active).length

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-300"
      >
        {icon}
        <span>{label}</span>
        <span className="text-zinc-600">{activeCount}/{items.length}</span>
        <ChevronDown size={10} className={`transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="absolute left-0 top-full z-50 mt-1 max-h-64 w-52 overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-900 p-2 shadow-xl">
          {items.map(({ id, label, active, color }) => (
            <label
              key={id}
              className="flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
            >
              <input
                type="checkbox"
                checked={active}
                onChange={() => onToggle(id)}
                className="rounded border-zinc-600 bg-zinc-800 accent-violet-500"
              />
              {color && (
                <span
                  className="inline-block h-2 w-2 shrink-0 rounded-full"
                  style={{ backgroundColor: color }}
                />
              )}
              <span className="truncate">{label}</span>
            </label>
          ))}
        </div>
      )}
    </div>
  )
}
