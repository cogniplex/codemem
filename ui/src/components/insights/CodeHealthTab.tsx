import { FileSearch, Network, Layers } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { useCodeHealthInsights } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { MetricCard } from './MetricCard'
import type { MemoryItem } from '../../api/types'

function InsightFeed({ insights }: { insights: MemoryItem[] }) {
  if (insights.length === 0) return null

  return (
    <div className="space-y-2">
      {insights.map((item) => (
        <div
          key={item.id}
          className="rounded-md border border-zinc-800 bg-zinc-900/50 px-4 py-3"
        >
          <div className="flex items-start justify-between gap-3">
            <p className="text-sm text-zinc-300">{item.content}</p>
            <span className="shrink-0 rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs font-medium text-emerald-400">
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

const barColors = ['#818cf8', '#a78bfa', '#c4b5fd', '#ddd6fe', '#ede9fe', '#f5f3ff']

export function CodeHealthTab() {
  const namespace = useNamespaceStore((s) => s.activeNamespace)
  const { data, isLoading } = useCodeHealthInsights(
    namespace ? { namespace } : undefined,
  )

  const hotspotData = (data?.file_hotspots ?? []).slice(0, 8).map((h) => ({
    name: h.description.split(' — ')[0] ?? h.description,
    count: h.frequency,
  }))

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <MetricCard
          label="File Hotspots"
          value={data?.file_hotspots.length ?? 0}
          icon={FileSearch}
          color="text-amber-400"
          isLoading={isLoading}
        />
        <MetricCard
          label="PageRank Leaders"
          value={data?.pagerank_leaders.length ?? 0}
          icon={Network}
          color="text-cyan-400"
          isLoading={isLoading}
        />
        <MetricCard
          label="Communities"
          value={data?.community_count ?? 0}
          icon={Layers}
          color="text-violet-400"
          isLoading={isLoading}
        />
      </div>

      {hotspotData.length > 0 && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
          <h3 className="mb-3 text-sm font-medium text-zinc-300">File Hotspots</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={hotspotData} layout="vertical" margin={{ left: 120 }}>
              <XAxis type="number" tick={{ fill: '#71717a', fontSize: 12 }} />
              <YAxis
                dataKey="name"
                type="category"
                tick={{ fill: '#a1a1aa', fontSize: 11 }}
                width={120}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#18181b',
                  border: '1px solid #3f3f46',
                  borderRadius: '6px',
                  color: '#e4e4e7',
                }}
              />
              <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                {hotspotData.map((_, i) => (
                  <Cell key={i} fill={barColors[i % barColors.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {(data?.pagerank_leaders ?? []).length > 0 && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
          <h3 className="mb-3 text-sm font-medium text-zinc-300">PageRank Leaders</h3>
          <div className="space-y-1.5">
            {data!.pagerank_leaders.slice(0, 8).map((entry, i) => (
              <div key={entry.node_id} className="flex items-center gap-3 text-sm">
                <span className="w-5 text-right text-xs text-zinc-600">{i + 1}</span>
                <span className="flex-1 truncate text-zinc-300">{entry.label}</span>
                <span className="tabular-nums text-zinc-500">{entry.score.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div>
        <h3 className="mb-3 text-sm font-medium text-zinc-300">Code Health Insights</h3>
        <InsightFeed insights={data?.insights ?? []} />
      </div>
    </div>
  )
}
