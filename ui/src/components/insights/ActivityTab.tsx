import { GitCommit, Users, FileText } from 'lucide-react'
import { useActivityInsights } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { MetricCard } from './MetricCard'
import type { MemoryItem } from '../../api/types'

function InsightFeed({ insights }: { insights: MemoryItem[] }) {
  if (insights.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-zinc-500">
        No activity insights yet. Run the enrichment pipeline to generate insights.
      </p>
    )
  }

  return (
    <div className="space-y-2">
      {insights.map((item) => (
        <div
          key={item.id}
          className="rounded-md border border-zinc-800 bg-zinc-900/50 px-4 py-3"
        >
          <div className="flex items-start justify-between gap-3">
            <p className="text-sm text-zinc-300">{item.content}</p>
            <span className="shrink-0 rounded-full bg-violet-500/15 px-2 py-0.5 text-xs font-medium text-violet-400">
              {(item.importance * 100).toFixed(0)}%
            </span>
          </div>
          <p className="mt-1 text-xs text-zinc-600">
            {new Date(item.created_at).toLocaleDateString()}
          </p>
        </div>
      ))}
    </div>
  )
}

export function ActivityTab() {
  const namespace = useNamespaceStore((s) => s.activeNamespace)
  const { data, isLoading } = useActivityInsights(
    namespace ? { namespace } : undefined,
  )

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <MetricCard
          label="Annotated Files"
          value={data?.git_summary.total_annotated_files ?? 0}
          icon={FileText}
          color="text-violet-400"
          isLoading={isLoading}
        />
        <MetricCard
          label="Insights Generated"
          value={data?.insights.length ?? 0}
          icon={GitCommit}
          color="text-cyan-400"
          isLoading={isLoading}
        />
        <MetricCard
          label="Contributors"
          value={data?.git_summary.top_authors.length ?? 0}
          icon={Users}
          color="text-emerald-400"
          isLoading={isLoading}
        />
      </div>

      {data?.git_summary.top_authors && data.git_summary.top_authors.length > 0 && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
          <h3 className="mb-3 text-sm font-medium text-zinc-300">Top Contributors</h3>
          <div className="flex flex-wrap gap-2">
            {data.git_summary.top_authors.map((author) => (
              <span
                key={author}
                className="rounded-full border border-zinc-700 bg-zinc-800 px-3 py-1 text-xs text-zinc-300"
              >
                {author}
              </span>
            ))}
          </div>
        </div>
      )}

      <div>
        <h3 className="mb-3 text-sm font-medium text-zinc-300">Activity Insights</h3>
        <InsightFeed insights={data?.insights ?? []} />
      </div>
    </div>
  )
}
