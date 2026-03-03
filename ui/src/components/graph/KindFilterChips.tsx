import { KIND_COLORS } from './constants'

interface Props {
  kinds: Record<string, number>
  activeKind: string | null
  onSelect: (kind: string | null) => void
}

export function KindFilterChips({ kinds, activeKind, onSelect }: Props) {
  const sorted = Object.entries(kinds).sort((a, b) => b[1] - a[1])

  return (
    <div className="flex flex-wrap gap-1.5">
      <button
        onClick={() => onSelect(null)}
        className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
          activeKind === null
            ? 'bg-zinc-700 text-zinc-100'
            : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200'
        }`}
      >
        All
      </button>
      {sorted.map(([kind, count]) => {
        const color = KIND_COLORS[kind] ?? '#71717a'
        const active = activeKind === kind
        return (
          <button
            key={kind}
            onClick={() => onSelect(active ? null : kind)}
            className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
              active
                ? 'text-zinc-100'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
            style={{
              backgroundColor: active ? color + '30' : undefined,
              borderColor: active ? color : undefined,
              border: active ? `1px solid ${color}60` : '1px solid transparent',
            }}
          >
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ backgroundColor: color }}
            />
            {kind}
            <span className="text-zinc-500">{count}</span>
          </button>
        )
      })}
    </div>
  )
}
