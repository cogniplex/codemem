import type { BrowseNodeItem } from '../../api/types'
import { KIND_COLORS } from './constants'
import { trimLabel } from '../../utils/paths'

interface Props {
  node: BrowseNodeItem
  selected: boolean
  onClick: () => void
}

export function BrowseNodeCard({ node, selected, onClick }: Props) {
  const color = KIND_COLORS[node.kind] ?? '#71717a'
  const centralityPct = Math.min(node.centrality * 100, 100)

  return (
    <button
      onClick={onClick}
      className={`w-full rounded-lg border p-3 text-left transition-colors ${
        selected
          ? 'border-violet-500/50 bg-zinc-800/80'
          : 'border-zinc-800 bg-zinc-900 hover:border-zinc-700 hover:bg-zinc-800/50'
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          {/* Kind badge */}
          <span
            className="mb-1 inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium"
            style={{ backgroundColor: color + '20', color }}
          >
            <span
              className="inline-block h-1.5 w-1.5 rounded-full"
              style={{ backgroundColor: color }}
            />
            {node.kind}
          </span>

          {/* Label */}
          <p className="mt-1 truncate text-sm font-medium text-zinc-200">
            {trimLabel(node.label)}
          </p>

          {/* Namespace */}
          {node.namespace && (
            <p className="mt-0.5 truncate text-xs text-zinc-500">
              {node.namespace}
            </p>
          )}
        </div>

        {/* Degree dot */}
        <div className="flex flex-col items-end gap-1">
          <span className="text-xs tabular-nums text-zinc-400">
            {node.degree}
          </span>
        </div>
      </div>

      {/* Centrality bar */}
      <div className="mt-2 h-1 overflow-hidden rounded-full bg-zinc-800">
        <div
          className="h-full rounded-full transition-all"
          style={{
            width: `${Math.max(centralityPct, 2)}%`,
            backgroundColor: color,
            opacity: 0.6,
          }}
        />
      </div>
    </button>
  )
}
