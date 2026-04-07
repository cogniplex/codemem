import { GitCommit, Users, FileText } from 'lucide-react'
import { useActivityInsights } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { MetricCard } from './MetricCard'
import type { MemoryItem } from '../../api/types'

function InsightFeed({ insights }: { insights: MemoryItem[] }) {
  if (insights.length === 0) {
    return (
      <p className="py-8 text-center text-[13px] text-zinc-600">
        No activity insights yet. Run the enrichment pipeline to generate insights.
      </p>
    )
  }

  return (
    <div className="space-y-2">
      {insights.map((item) => (
        <div
          key={item.id}
          className="rounded-xl border border-zinc-800/40 bg-zinc-900/50 px-4 py-3 transition-colors hover:bg-zinc-800/20"
        >
          <div className="flex items-start justify-between gap-3">
            <p className="text-[13px] leading-relaxed text-zinc-300">{item.content}</p>
            <span className="shrink-0 rounded-md bg-violet-500/10 px-2 py-0.5 text-[11px] font-medium tabular-nums text-violet-400">
              {(item.importance * 100).toFixed(0)}%
            </span>
          </div>
          <p className="mt-1 text-[11px] text-zinc-600">
            {new Date(item.created_at).toLocaleDateString()}
          </p>
        </div>
      ))}
    </div>
  )
}

export function ActivityTab() {
  const namespace = useNamespaceStore((s) => s.active)
  const { data, isLoading } = useActivityInsights(
    namespace ? { namespace } : undefined,
  )

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <MetricCard label="Annotated Files" value={data?.git_summary.total_annotated_files ?? 0} icon={FileText} color="text-violet-400" isLoading={isLoading} />
        <MetricCard label="Insights Generated" value={data?.insights.length ?? 0} icon={GitCommit} color="text-cyan-400" isLoading={isLoading} />
        <MetricCard label="Contributors" value={data?.git_summary.top_authors.length ?? 0} icon={Users} color="text-emerald-400" isLoading={isLoading} />
      </div>

      {data?.git_summary.top_authors && data.git_summary.top_authors.length > 0 && (
        <div className="overflow-hidden rounded-xl border border-zinc-800/50 bg-zinc-900">
          <div className="bg-zinc-800/40 px-5 py-3">
            <h3 className="text-[13px] font-medium text-zinc-100">Top Contributors</h3>
          </div>
          <div className="flex flex-wrap gap-2 p-4">
            {data.git_summary.top_authors.map((author) => (
              <span
                key={author}
                className="rounded-lg border border-zinc-700/40 bg-zinc-800/50 px-3 py-1 text-[12px] text-zinc-300"
              >
                {author}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="overflow-hidden rounded-xl border border-zinc-800/50 bg-zinc-900">
        <div className="bg-zinc-800/40 px-5 py-3">
          <h3 className="text-[13px] font-medium text-zinc-100">Activity Insights</h3>
        </div>
        <div className="p-4">
          <InsightFeed insights={data?.insights ?? []} />
        </div>
      </div>
    </div>
  )
}
