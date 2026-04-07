import { useState, useMemo } from 'react'
import {
  Clock,
  FileWarning,
  GitBranch,
  TrendingUp,
  Loader2,
  AlertTriangle,
} from 'lucide-react'
import { useStaleFiles, useDrift } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'

function daysAgo(n: number): string {
  const d = new Date()
  d.setDate(d.getDate() - n)
  return d.toISOString()
}

export function TemporalView() {
  const namespace = useNamespaceStore((s) => s.active)
  const [staleDays, setStaleDays] = useState(30)
  const [driftRange, setDriftRange] = useState<'30d' | '90d' | '180d'>('90d')

  const driftFrom = useMemo(() => {
    const days = driftRange === '30d' ? 30 : driftRange === '90d' ? 90 : 180
    return daysAgo(days)
  }, [driftRange])
  const driftTo = useMemo(() => new Date().toISOString(), [])

  const { data: staleData, isLoading: staleLoading } = useStaleFiles(
    namespace ?? undefined,
    staleDays,
  )
  const { data: driftData, isLoading: driftLoading } = useDrift(
    driftFrom,
    driftTo,
    namespace ?? undefined,
  )

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Clock size={20} className="text-zinc-400" />
        <h2 className="text-lg font-semibold text-zinc-100">Temporal Analysis</h2>
      </div>

      {/* Drift Section */}
      <div className="rounded-xl border border-zinc-800/60 bg-zinc-900">
        <div className="flex items-center justify-between border-b border-zinc-800/60 px-5 py-3">
          <div className="flex items-center gap-2">
            <GitBranch size={14} className="text-violet-400" />
            <h3 className="text-sm font-medium text-zinc-300">Architectural Drift</h3>
          </div>
          <div className="flex items-center gap-1">
            {(['30d', '90d', '180d'] as const).map((r) => (
              <button
                key={r}
                onClick={() => setDriftRange(r)}
                className={`rounded px-2 py-0.5 text-xs transition-colors ${
                  driftRange === r
                    ? 'bg-zinc-700 text-zinc-100'
                    : 'text-zinc-500 hover:text-zinc-300'
                }`}
              >
                {r}
              </button>
            ))}
          </div>
        </div>

        {driftLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 size={16} className="animate-spin text-zinc-500" />
          </div>
        ) : driftData ? (
          <div className="p-5 space-y-4">
            {/* Drift metrics */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <MetricBox label="New Cross-Module Edges" value={driftData.new_cross_module_edges} />
              <MetricBox label="Files Added" value={driftData.added_files} />
              <MetricBox label="Files Removed" value={driftData.removed_files} />
              <MetricBox
                label="Coupling Pairs"
                value={driftData.coupling_increases?.length ?? 0}
              />
            </div>

            {/* Hotspot files */}
            {driftData.hotspot_files?.length > 0 && (
              <div>
                <p className="mb-2 text-xs font-medium text-zinc-400">
                  <TrendingUp size={12} className="mr-1 inline" />
                  Hotspot Files (most modified)
                </p>
                <div className="space-y-1">
                  {driftData.hotspot_files.map((f) => (
                    <div
                      key={f}
                      className="rounded bg-zinc-900 px-3 py-1.5 text-xs font-mono text-zinc-300"
                    >
                      {f}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Coupling increases */}
            {driftData.coupling_increases?.length > 0 && (
              <div>
                <p className="mb-2 text-xs font-medium text-zinc-400">
                  Top Coupling Pairs
                </p>
                <div className="space-y-1">
                  {driftData.coupling_increases.map(([a, b, count], i) => (
                    <div
                      key={i}
                      className="flex items-center gap-2 rounded bg-zinc-900 px-3 py-1.5 text-xs"
                    >
                      <span className="truncate font-mono text-zinc-300">{stripPrefix(a)}</span>
                      <span className="text-zinc-600">&harr;</span>
                      <span className="truncate font-mono text-zinc-300">{stripPrefix(b)}</span>
                      <span className="ml-auto shrink-0 text-zinc-500">{count}x</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <EmptyState message="Run codemem analyze with --days to populate temporal data." />
        )}
      </div>

      {/* Stale Files Section */}
      <div className="rounded-xl border border-zinc-800/60 bg-zinc-900">
        <div className="flex items-center justify-between border-b border-zinc-800/60 px-5 py-3">
          <div className="flex items-center gap-2">
            <FileWarning size={14} className="text-amber-400" />
            <h3 className="text-sm font-medium text-zinc-300">Stale Files</h3>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-zinc-500">Stale after</label>
            <select
              value={staleDays}
              onChange={(e) => setStaleDays(Number(e.target.value))}
              className="rounded border border-zinc-700 bg-zinc-900 px-2 py-0.5 text-xs text-zinc-300 outline-none"
            >
              <option value={14}>14 days</option>
              <option value={30}>30 days</option>
              <option value={60}>60 days</option>
              <option value={90}>90 days</option>
            </select>
          </div>
        </div>

        {staleLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 size={16} className="animate-spin text-zinc-500" />
          </div>
        ) : staleData && staleData.files.length > 0 ? (
          <div className="divide-y divide-zinc-800/50">
            <div className="px-5 py-2 text-xs text-zinc-500">
              {staleData.stale_files} file(s) not modified in {staleData.stale_days}+ days
            </div>
            {staleData.files.slice(0, 30).map((f) => (
              <div
                key={f.file_path}
                className="flex items-center gap-3 px-5 py-2"
              >
                <span className="min-w-0 flex-1 truncate text-xs font-mono text-zinc-300">
                  {f.file_path}
                </span>
                <span className="shrink-0 text-xs tabular-nums text-zinc-500">
                  centrality: {f.centrality.toFixed(3)}
                </span>
                <span className="shrink-0 text-xs tabular-nums text-zinc-600">
                  {f.incoming_edges} dep{f.incoming_edges !== 1 ? 's' : ''}
                </span>
                {f.last_modified && (
                  <span className="shrink-0 text-xs text-zinc-600">
                    {new Date(f.last_modified).toLocaleDateString()}
                  </span>
                )}
              </div>
            ))}
          </div>
        ) : staleData ? (
          <EmptyState message={`No files stale for ${staleDays}+ days.`} />
        ) : (
          <EmptyState message="Run codemem analyze to populate file data." />
        )}
      </div>
    </div>
  )
}

function MetricBox({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-900 p-3">
      <p className="text-xs text-zinc-500">{label}</p>
      <p className="mt-1 text-xl font-semibold tabular-nums text-zinc-100">{value}</p>
    </div>
  )
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-2 px-5 py-6 text-sm text-zinc-500">
      <AlertTriangle size={14} className="text-zinc-600" />
      {message}
    </div>
  )
}

function stripPrefix(id: string): string {
  return id.replace(/^file:|^sym:/, '')
}
