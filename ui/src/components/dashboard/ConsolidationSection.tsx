import { Play, Loader2, Recycle } from 'lucide-react'
import { useConsolidationStatus, useRunConsolidation } from '../../api/hooks'
import { Card } from '../shared/Card'

export function ConsolidationSection() {
  const { data, isLoading } = useConsolidationStatus()
  const runConsolidation = useRunConsolidation()

  return (
    <Card
      title="Consolidation"
      icon={<Recycle size={14} className="text-zinc-500" />}
      padded={false}
    >
      <div className="flex divide-x divide-zinc-800/20">
        {isLoading ? (
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="flex flex-1 items-center justify-between gap-3 px-5 py-3">
              <span className="h-4 w-16 animate-pulse rounded-md bg-zinc-700/30" />
              <span className="h-6 w-12 animate-pulse rounded-md bg-zinc-700/20" />
            </div>
          ))
        ) : !data?.cycles.length ? (
          <p className="flex-1 px-5 py-4 text-center text-[13px] text-zinc-600">No consolidation data</p>
        ) : (
          data.cycles.map((c) => (
            <div key={c.cycle} className="flex flex-1 items-center justify-between gap-3 px-5 py-3 transition-colors hover:bg-zinc-700/10">
              <div className="min-w-0">
                <p className="text-[13px] font-medium capitalize text-zinc-200">{c.cycle}</p>
                <p className="text-[11px] text-zinc-600">
                  {c.last_run
                    ? `${new Date(c.last_run).toLocaleDateString()} · ${c.affected_count}`
                    : 'Never run'}
                </p>
              </div>
              <button
                onClick={() => runConsolidation.mutate(c.cycle)}
                disabled={runConsolidation.isPending}
                className="flex shrink-0 items-center gap-1.5 rounded-lg border border-zinc-700/40 bg-zinc-800/40 px-2.5 py-1 text-[11px] font-medium text-zinc-300 transition-colors hover:border-zinc-600 hover:bg-zinc-700/40 disabled:opacity-50"
              >
                {runConsolidation.isPending ? <Loader2 size={11} className="animate-spin" /> : <Play size={11} />}
                Run
              </button>
            </div>
          ))
        )}
      </div>
    </Card>
  )
}
