import { Link2, Layers, Zap } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { usePerformanceInsights } from '../../api/hooks'
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
            <span className="shrink-0 rounded-full bg-cyan-500/15 px-2 py-0.5 text-xs font-medium text-cyan-400">
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

const barColors = ['#f87171', '#fb923c', '#fbbf24', '#a3e635', '#34d399', '#22d3ee']

export function PerformanceTab() {
  const namespace = useNamespaceStore((s) => s.active)
  const { data, isLoading } = usePerformanceInsights(
    namespace ? { namespace } : undefined,
  )

  const couplingData = (data?.high_coupling_nodes ?? []).slice(0, 8).map((n) => ({
    name: n.label.length > 30 ? n.label.slice(-30) : n.label,
    score: n.coupling_score,
  }))

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <MetricCard
          label="High-Coupling Nodes"
          value={data?.high_coupling_nodes.length ?? 0}
          icon={Link2}
          color="text-red-400"
          isLoading={isLoading}
        />
        <MetricCard
          label="Dependency Depth"
          value={data?.max_depth ?? 0}
          icon={Layers}
          color="text-amber-400"
          isLoading={isLoading}
        />
        <MetricCard
          label="Critical Path Files"
          value={data?.critical_path.length ?? 0}
          icon={Zap}
          color="text-cyan-400"
          isLoading={isLoading}
        />
      </div>

      {couplingData.length > 0 && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
          <h3 className="mb-3 text-sm font-medium text-zinc-300">High-Coupling Nodes</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={couplingData} layout="vertical" margin={{ left: 120 }}>
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
              <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                {couplingData.map((_, i) => (
                  <Cell key={i} fill={barColors[i % barColors.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {(data?.critical_path ?? []).length > 0 && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
          <h3 className="mb-3 text-sm font-medium text-zinc-300">Critical Path (PageRank)</h3>
          <div className="space-y-1.5">
            {data!.critical_path.slice(0, 8).map((entry, i) => (
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
        <h3 className="mb-3 text-sm font-medium text-zinc-300">Performance Insights</h3>
        <InsightFeed insights={data?.insights ?? []} />
      </div>
    </div>
  )
}
