import { useState, useMemo } from 'react'
import {
  Clock,
  FileWarning,
  GitBranch,
  TrendingUp,
  Loader2,
  AlertTriangle,
  ArrowLeftRight,
  FileText,
  Layers,
  Minus,
  Plus,
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

  const { data: staleData, isLoading: staleLoading } = useStaleFiles(namespace ?? undefined, staleDays)
  const { data: driftData, isLoading: driftLoading } = useDrift(driftFrom, driftTo, namespace ?? undefined)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="rounded-lg bg-violet-500/10 p-2">
          <Clock size={18} className="text-violet-400" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-zinc-100">Temporal Analysis</h2>
          <p className="text-[12px] text-zinc-500">Architecture drift, file staleness, and change patterns</p>
        </div>
      </div>

      {/* Drift Summary Cards */}
      {driftData && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <DriftMetric
            label="Cross-Module Edges"
            value={driftData.new_cross_module_edges}
            icon={<ArrowLeftRight size={14} />}
            accent="text-violet-400"
          />
          <DriftMetric
            label="Files Added"
            value={driftData.added_files}
            icon={<Plus size={14} />}
            accent="text-emerald-400"
          />
          <DriftMetric
            label="Files Removed"
            value={driftData.removed_files}
            icon={<Minus size={14} />}
            accent="text-red-400"
          />
          <DriftMetric
            label="Coupling Pairs"
            value={driftData.coupling_increases?.length ?? 0}
            icon={<Layers size={14} />}
            accent="text-amber-400"
          />
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Drift Section */}
        <div className="rounded-xl border border-zinc-800/50 bg-zinc-900">
          <div className="flex items-center justify-between border-b border-zinc-800/40 px-5 py-3.5">
            <div className="flex items-center gap-2">
              <GitBranch size={14} className="text-violet-400" />
              <h3 className="text-[13px] font-medium text-zinc-100">Architectural Drift</h3>
            </div>
            <div className="flex items-center gap-0.5 rounded-lg border border-zinc-800/50 bg-zinc-950/50 p-0.5">
              {(['30d', '90d', '180d'] as const).map((r) => (
                <button
                  key={r}
                  onClick={() => setDriftRange(r)}
                  className={`rounded-md px-2.5 py-1 text-[11px] font-medium transition-all ${
                    driftRange === r
                      ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                      : 'text-zinc-500 hover:text-zinc-300'
                  }`}
                >
                  {r}
                </button>
              ))}
            </div>
          </div>

          {driftLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 size={16} className="animate-spin text-zinc-600" />
            </div>
          ) : driftData ? (
            <div className="p-5 space-y-5">
              {/* Hotspot files */}
              {driftData.hotspot_files?.length > 0 && (
                <div>
                  <div className="mb-2.5 flex items-center gap-1.5">
                    <TrendingUp size={12} className="text-amber-400" />
                    <p className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                      Hotspot Files
                    </p>
                  </div>
                  <div className="space-y-1">
                    {driftData.hotspot_files.map((f, i) => (
                      <div
                        key={f}
                        className="flex items-center gap-2.5 rounded-lg bg-zinc-950/50 px-3 py-2"
                      >
                        <span className="text-[11px] tabular-nums text-zinc-600">{i + 1}</span>
                        <FileText size={12} className="text-zinc-600" />
                        <span className="truncate text-[12px] font-mono text-zinc-300">{f}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Coupling */}
              {driftData.coupling_increases?.length > 0 && (
                <div>
                  <p className="mb-2.5 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                    Top Coupling Pairs
                  </p>
                  <div className="space-y-1">
                    {driftData.coupling_increases.map(([a, b, count], i) => (
                      <div
                        key={i}
                        className="flex items-center gap-2 rounded-lg bg-zinc-950/50 px-3 py-2 text-[12px]"
                      >
                        <span className="min-w-0 flex-1 truncate font-mono text-zinc-400">{strip(a)}</span>
                        <ArrowLeftRight size={10} className="shrink-0 text-zinc-700" />
                        <span className="min-w-0 flex-1 truncate font-mono text-zinc-400">{strip(b)}</span>
                        <span className="shrink-0 rounded-md bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-medium tabular-nums text-amber-400">
                          {count}x
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {!driftData.hotspot_files?.length && !driftData.coupling_increases?.length && (
                <EmptyBlock message="No drift detected in this period." />
              )}
            </div>
          ) : (
            <EmptyBlock message="Run codemem analyze to populate temporal data." />
          )}
        </div>

        {/* Stale Files Section */}
        <div className="rounded-xl border border-zinc-800/50 bg-zinc-900">
          <div className="flex items-center justify-between border-b border-zinc-800/40 px-5 py-3.5">
            <div className="flex items-center gap-2">
              <FileWarning size={14} className="text-amber-400" />
              <h3 className="text-[13px] font-medium text-zinc-100">Stale Files</h3>
            </div>
            <div className="flex items-center gap-0.5 rounded-lg border border-zinc-800/50 bg-zinc-950/50 p-0.5">
              {[14, 30, 60, 90].map((d) => (
                <button
                  key={d}
                  onClick={() => setStaleDays(d)}
                  className={`rounded-md px-2.5 py-1 text-[11px] font-medium transition-all ${
                    staleDays === d
                      ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                      : 'text-zinc-500 hover:text-zinc-300'
                  }`}
                >
                  {d}d
                </button>
              ))}
            </div>
          </div>

          {staleLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 size={16} className="animate-spin text-zinc-600" />
            </div>
          ) : staleData && staleData.files.length > 0 ? (
            <div className="max-h-[500px] overflow-y-auto">
              <div className="px-5 py-2">
                <p className="text-[11px] text-zinc-600">
                  {staleData.stale_files} file(s) unchanged for {staleData.stale_days}+ days
                </p>
              </div>
              <div className="divide-y divide-zinc-800/30 px-2 pb-2">
                {staleData.files.slice(0, 30).map((f) => (
                  <div
                    key={f.file_path}
                    className="flex items-center gap-3 rounded-lg px-3 py-2.5 transition-colors hover:bg-zinc-800/20"
                  >
                    <FileText size={13} className="shrink-0 text-zinc-600" />
                    <span className="min-w-0 flex-1 truncate text-[12px] font-mono text-zinc-300">
                      {f.file_path}
                    </span>
                    <div className="flex shrink-0 items-center gap-3">
                      <span className="text-[10px] tabular-nums text-zinc-600" title="Centrality">
                        c:{f.centrality.toFixed(3)}
                      </span>
                      <span className="text-[10px] tabular-nums text-zinc-600" title="Dependencies">
                        {f.incoming_edges} dep{f.incoming_edges !== 1 ? 's' : ''}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <EmptyBlock
              message={staleData ? `No files stale for ${staleDays}+ days.` : 'Run codemem analyze first.'}
            />
          )}
        </div>
      </div>
    </div>
  )
}

function DriftMetric({
  label, value, icon, accent,
}: {
  label: string; value: number; icon: React.ReactNode; accent: string
}) {
  return (
    <div className="rounded-xl border border-zinc-800/50 bg-zinc-900 p-4">
      <div className="flex items-center gap-2">
        <span className={accent}>{icon}</span>
        <span className="text-[11px] font-medium text-zinc-500">{label}</span>
      </div>
      <p className="mt-2 text-2xl font-bold tabular-nums text-zinc-50">{value}</p>
    </div>
  )
}

function EmptyBlock({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-2 px-5 py-10 text-[13px] text-zinc-500">
      <AlertTriangle size={14} className="text-zinc-600" />
      {message}
    </div>
  )
}

function strip(id: string): string {
  return id.replace(/^file:|^sym:/, '')
}
