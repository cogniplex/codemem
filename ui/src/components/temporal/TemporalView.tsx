import { useState, useMemo } from 'react'
import { Clock, GitCommit, FileText, Loader2 } from 'lucide-react'
import { useTemporalChanges } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'

function daysAgo(n: number): string {
  const d = new Date()
  d.setDate(d.getDate() - n)
  return d.toISOString()
}

export function TemporalView() {
  const namespace = useNamespaceStore((s) => s.active)
  const [range, setRange] = useState<'7d' | '30d' | '90d'>('30d')

  const from = useMemo(() => {
    const days = range === '7d' ? 7 : range === '30d' ? 30 : 90
    return daysAgo(days)
  }, [range])
  const to = useMemo(() => new Date().toISOString(), [])

  const { data, isLoading, error } = useTemporalChanges(
    from,
    to,
    namespace ?? undefined,
  )

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Clock size={20} className="text-zinc-400" />
          <h2 className="text-lg font-semibold text-zinc-100">Temporal</h2>
        </div>

        <div className="flex items-center gap-1">
          {(['7d', '30d', '90d'] as const).map((r) => (
            <button
              key={r}
              onClick={() => setRange(r)}
              className={`rounded-md px-3 py-1 text-sm transition-colors ${
                range === r
                  ? 'bg-zinc-800 text-zinc-100'
                  : 'text-zinc-500 hover:bg-zinc-800/50 hover:text-zinc-300'
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      {/* Stats */}
      {data && (
        <div className="grid grid-cols-3 gap-4">
          <StatCard
            label="Commits"
            value={data.commits}
            icon={<GitCommit size={16} className="text-violet-400" />}
          />
          <StatCard
            label="Files Changed"
            value={new Set(data.entries.flatMap((e) => e.files ?? [])).size}
            icon={<FileText size={16} className="text-cyan-400" />}
          />
          <StatCard
            label="Authors"
            value={new Set(data.entries.map((e) => e.author)).size}
            icon={<Clock size={16} className="text-amber-400" />}
          />
        </div>
      )}

      {/* Content */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 size={20} className="animate-spin text-zinc-500" />
          <span className="ml-2 text-sm text-zinc-400">Loading temporal data...</span>
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-850 p-6 text-center text-sm text-zinc-500">
          Failed to load temporal data. Run <code className="text-zinc-400">codemem analyze</code> to populate the temporal graph.
        </div>
      )}

      {data && data.entries.length === 0 && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-850 p-6 text-center text-sm text-zinc-500">
          No commits found in this range. Try a wider range or a different namespace.
        </div>
      )}

      {data && data.entries.length > 0 && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-850">
          <div className="border-b border-zinc-800 px-5 py-3">
            <h3 className="text-sm font-medium text-zinc-300">
              Recent Commits ({data.entries.length})
            </h3>
          </div>
          <div className="divide-y divide-zinc-800/50">
            {data.entries.slice(0, 50).map((entry) => (
              <CommitRow key={entry.sha} entry={entry} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function StatCard({
  label,
  value,
  icon,
}: {
  label: string
  value: number
  icon: React.ReactNode
}) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-850 p-4">
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-xs text-zinc-500">{label}</span>
      </div>
      <p className="mt-1 text-2xl font-semibold tabular-nums text-zinc-100">{value}</p>
    </div>
  )
}

function CommitRow({ entry }: { entry: import('../../api/types').TemporalEntry }) {
  const [expanded, setExpanded] = useState(false)
  const date = new Date(entry.timestamp)
  const relDate = formatRelative(date)
  const files = entry.files ?? []
  const symbols = entry.symbols ?? []

  return (
    <div className="px-5 py-3">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-start gap-3 text-left"
      >
        <GitCommit size={14} className="mt-0.5 shrink-0 text-zinc-600" />
        <div className="min-w-0 flex-1">
          <p className="text-sm text-zinc-200 line-clamp-2">
            {entry.message || entry.sha.slice(0, 8)}
          </p>
          <div className="mt-1 flex items-center gap-3 text-xs text-zinc-500">
            <span className="font-mono">{entry.sha.slice(0, 7)}</span>
            <span>{entry.author}</span>
            <span>{relDate}</span>
            {files.length > 0 && (
              <span className="text-zinc-600">{files.length} file(s)</span>
            )}
          </div>
        </div>
      </button>

      {expanded && (files.length > 0 || symbols.length > 0) && (
        <div className="ml-8 mt-2 space-y-1">
          {files.map((f) => (
            <div key={f} className="flex items-center gap-2 text-xs">
              <FileText size={10} className="text-zinc-600" />
              <span className="truncate text-zinc-400">{f}</span>
            </div>
          ))}
          {symbols.length > 0 && (
            <p className="text-xs text-zinc-600">
              {symbols.length} symbol(s) affected
            </p>
          )}
        </div>
      )}
    </div>
  )
}

function formatRelative(date: Date): string {
  const now = Date.now()
  const diff = now - date.getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 30) return `${days}d ago`
  return date.toLocaleDateString()
}
