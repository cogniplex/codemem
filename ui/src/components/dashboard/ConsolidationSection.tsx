import { Play, Loader2 } from 'lucide-react'
import { useConsolidationStatus, useRunConsolidation } from '../../api/hooks'

export function ConsolidationSection() {
  const { data, isLoading } = useConsolidationStatus()
  const runConsolidation = useRunConsolidation()

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-850">
      <div className="border-b border-zinc-800 px-5 py-3">
        <h3 className="text-sm font-medium text-zinc-300">
          Consolidation Cycles
        </h3>
      </div>
      <div className="divide-y divide-zinc-800/50">
        {isLoading ? (
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="flex items-center gap-3 px-5 py-3">
              <span className="h-4 w-20 animate-pulse rounded bg-zinc-800" />
              <span className="ml-auto h-4 w-28 animate-pulse rounded bg-zinc-800" />
            </div>
          ))
        ) : !data?.cycles.length ? (
          <p className="px-5 py-6 text-center text-sm text-zinc-500">
            No consolidation data
          </p>
        ) : (
          data.cycles.map((c) => (
            <div
              key={c.cycle}
              className="flex items-center gap-4 px-5 py-3"
            >
              <span className="text-sm font-medium capitalize text-zinc-200">
                {c.cycle}
              </span>
              <span className="text-xs text-zinc-500">
                {c.last_run
                  ? `Last run: ${new Date(c.last_run).toLocaleDateString()}`
                  : 'Never run'}
              </span>
              <span className="text-xs text-zinc-600">
                {c.affected_count} affected
              </span>
              <button
                onClick={() => runConsolidation.mutate(c.cycle)}
                disabled={runConsolidation.isPending}
                className="ml-auto flex items-center gap-1.5 rounded bg-zinc-800 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-700 disabled:opacity-50"
              >
                {runConsolidation.isPending ? (
                  <Loader2 size={12} className="animate-spin" />
                ) : (
                  <Play size={12} />
                )}
                Run
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
