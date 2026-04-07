import { useState, useMemo } from 'react'
import {
  GitBranch, FileWarning, TrendingUp, Loader2, AlertTriangle,
  ArrowLeftRight, FileText,
} from 'lucide-react'
import { useStaleFiles, useDrift } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { Card } from '../shared/Card'

function daysAgo(n: number): string {
  const d = new Date(); d.setDate(d.getDate() - n); return d.toISOString()
}

function Segmented({ options, value, onChange }: { options: string[]; value: string; onChange: (v: string) => void }) {
  return (
    <div className="flex items-center gap-0.5 rounded-lg border border-zinc-800/40 bg-zinc-950 p-0.5">
      {options.map((o) => (
        <button key={o} onClick={() => onChange(o)}
          className={`rounded-md px-2.5 py-1 text-[11px] font-medium transition-all ${value === o ? 'bg-zinc-800 text-zinc-100 shadow-sm' : 'text-zinc-500 hover:text-zinc-300'}`}
        >{o}</button>
      ))}
    </div>
  )
}

export function TemporalSection() {
  const namespace = useNamespaceStore((s) => s.active)
  const [staleDays, setStaleDays] = useState(30)
  const [driftRange, setDriftRange] = useState('90d')

  const driftFrom = useMemo(() => daysAgo(driftRange === '30d' ? 30 : driftRange === '90d' ? 90 : 180), [driftRange])
  const driftTo = useMemo(() => new Date().toISOString(), [])

  const { data: staleData, isLoading: staleLoading } = useStaleFiles(namespace ?? undefined, staleDays)
  const { data: driftData, isLoading: driftLoading } = useDrift(driftFrom, driftTo, namespace ?? undefined)

  return (
    <div className="space-y-4">
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Drift */}
        <Card
          title="Architectural Drift"
          icon={<GitBranch size={14} className="text-violet-400" />}
          actions={<Segmented options={['30d', '90d', '180d']} value={driftRange} onChange={setDriftRange} />}
        >
          {driftLoading ? (
            <div className="flex items-center justify-center py-8"><Loader2 size={16} className="animate-spin text-zinc-600" /></div>
          ) : driftData ? (
            <div className="space-y-5">
              {driftData.hotspot_files?.length > 0 && (
                <div>
                  <div className="mb-2 flex items-center gap-1.5">
                    <TrendingUp size={12} className="text-amber-400" />
                    <p className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Hotspot Files</p>
                  </div>
                  <div className="space-y-1">
                    {driftData.hotspot_files.map((f, i) => (
                      <div key={f} className="flex items-center gap-2.5 rounded-lg border border-zinc-700/30 bg-zinc-900 px-3 py-2">
                        <span className="text-[11px] tabular-nums text-zinc-600">{i + 1}</span>
                        <FileText size={12} className="text-zinc-600" />
                        <span className="truncate text-[12px] font-mono text-zinc-300">{f}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {driftData.coupling_increases?.length > 0 && (
                <div>
                  <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Top Coupling</p>
                  <div className="space-y-1">
                    {driftData.coupling_increases.map(([a, b, count], i) => (
                      <div key={i} className="flex items-center gap-2 rounded-lg border border-zinc-700/30 bg-zinc-900 px-3 py-2 text-[12px]">
                        <span className="min-w-0 flex-1 truncate font-mono text-zinc-400">{strip(a)}</span>
                        <ArrowLeftRight size={10} className="shrink-0 text-zinc-700" />
                        <span className="min-w-0 flex-1 truncate font-mono text-zinc-400">{strip(b)}</span>
                        <span className="shrink-0 rounded-md bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-medium tabular-nums text-amber-400">{count}x</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {!driftData.hotspot_files?.length && !driftData.coupling_increases?.length && (
                <Empty message="No drift detected in this period." />
              )}
            </div>
          ) : (
            <Empty message="Run codemem analyze to populate temporal data." />
          )}
        </Card>

        {/* Stale Files */}
        <Card
          title="Stale Files"
          icon={<FileWarning size={14} className="text-amber-400" />}
          actions={<Segmented options={['14d', '30d', '60d', '90d']} value={`${staleDays}d`} onChange={(v) => setStaleDays(parseInt(v))} />}
          padded={false}
        >
          {staleLoading ? (
            <div className="flex items-center justify-center py-8"><Loader2 size={16} className="animate-spin text-zinc-600" /></div>
          ) : staleData && staleData.files.length > 0 ? (
            <div className="max-h-[400px] overflow-y-auto">
              <div className="px-5 py-2">
                <p className="text-[11px] text-zinc-600">{staleData.stale_files} file(s) unchanged for {staleData.stale_days}+ days</p>
              </div>
              <div className="divide-y divide-zinc-800/20 px-2 pb-2">
                {staleData.files.slice(0, 20).map((f) => (
                  <div key={f.file_path} className="flex items-center gap-3 rounded-lg px-3 py-2 transition-colors hover:bg-zinc-900/40">
                    <FileText size={13} className="shrink-0 text-zinc-600" />
                    <span className="min-w-0 flex-1 truncate text-[12px] font-mono text-zinc-300">{f.file_path}</span>
                    <span className="text-[10px] tabular-nums text-zinc-600">c:{f.centrality.toFixed(3)}</span>
                    <span className="text-[10px] tabular-nums text-zinc-600">{f.incoming_edges} dep{f.incoming_edges !== 1 ? 's' : ''}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="p-5"><Empty message={staleData ? `No files stale for ${staleDays}+ days.` : 'Run codemem analyze first.'} /></div>
          )}
        </Card>
      </div>
    </div>
  )
}

function Empty({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-2 text-[13px] text-zinc-500">
      <AlertTriangle size={14} className="text-zinc-600" />
      {message}
    </div>
  )
}

function strip(id: string): string { return id.replace(/^file:|^sym:/, '') }
