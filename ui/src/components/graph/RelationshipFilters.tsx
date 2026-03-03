import { EDGE_COLORS } from './constants'

interface Props {
  activeRelationships: Set<string>
  edgeCounts: Record<string, number>
  onToggle: (rel: string) => void
}

export function RelationshipFilters({ activeRelationships, edgeCounts, onToggle }: Props) {
  const sorted = Object.entries(edgeCounts).sort((a, b) => b[1] - a[1])

  if (sorted.length === 0) return null

  return (
    <div className="absolute bottom-4 left-4 right-4 z-10">
      <div className="flex flex-wrap items-center gap-1.5 rounded-lg border border-zinc-800 bg-zinc-900/90 px-3 py-2 backdrop-blur-sm">
        <span className="mr-1 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
          Edges
        </span>
        {sorted.map(([rel, count]) => {
          const active = activeRelationships.has(rel)
          const color = EDGE_COLORS[rel] ?? '#52525b60'
          // Strip alpha from hex for the dot
          const dotColor = color.length > 7 ? color.slice(0, 7) : color
          return (
            <button
              key={rel}
              onClick={() => onToggle(rel)}
              className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] transition-opacity ${
                active ? 'opacity-100' : 'opacity-30'
              } hover:opacity-80`}
            >
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ backgroundColor: dotColor }}
              />
              <span className="text-zinc-300">{rel}</span>
              <span className="text-zinc-500">{count}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
