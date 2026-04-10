import { Link2, Layers, Zap } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { usePerformanceInsights } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { Card, MetricCard } from '../shared/Card'
import type { MemoryItem } from '../../api/types'

function InsightFeed({ insights }: { insights: MemoryItem[] }) {
  if (insights.length === 0) return <p className="py-6 text-center text-[13px] text-zinc-600">No performance insights yet.</p>

  return (
    <div className="space-y-2">
      {insights.map((item) => (
        <div key={item.id} className="rounded-lg border border-zinc-700/30 bg-zinc-900 px-4 py-3 transition-colors hover:bg-zinc-800/80">
          <div className="flex items-start justify-between gap-3">
            <p className="min-w-0 break-words text-[13px] leading-relaxed text-zinc-300">{item.content}</p>
            <span className="shrink-0 rounded-md bg-cyan-500/10 px-2 py-0.5 text-[11px] font-medium tabular-nums text-cyan-400">
              {(item.importance * 100).toFixed(0)}%
            </span>
          </div>
          <p className="mt-1 text-[11px] text-zinc-600">{new Date(item.created_at).toLocaleDateString()}</p>
        </div>
      ))}
    </div>
  )
}

const barColors = ['#f87171', '#fb923c', '#fbbf24', '#a3e635', '#34d399', '#22d3ee']

export function PerformanceTab() {
  const namespace = useNamespaceStore((s) => s.active)
  const { data, isLoading } = usePerformanceInsights(namespace ? { namespace } : undefined)

  const couplingData = (data?.high_coupling_nodes ?? []).slice(0, 8).map((n) => ({
    name: n.label.length > 30 ? n.label.slice(-30) : n.label,
    score: n.coupling_score,
  }))

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <MetricCard label="High-Coupling Nodes" value={data?.high_coupling_nodes.length ?? 0} icon={<Link2 size={15} className="text-red-400" />} isLoading={isLoading} />
        <MetricCard label="Dependency Depth" value={data?.max_depth ?? 0} icon={<Layers size={15} className="text-amber-400" />} isLoading={isLoading} />
        <MetricCard label="Critical Path Files" value={data?.critical_path.length ?? 0} icon={<Zap size={15} className="text-cyan-400" />} isLoading={isLoading} />
      </div>

      {couplingData.length > 0 && (
        <Card title="High-Coupling Nodes">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={couplingData} layout="vertical" margin={{ left: 120 }}>
              <XAxis type="number" tick={{ fill: '#71717a', fontSize: 12 }} />
              <YAxis dataKey="name" type="category" tick={{ fill: '#a1a1aa', fontSize: 11 }} width={120} />
              <Tooltip contentStyle={{ backgroundColor: '#18181b', border: '1px solid #3f3f46', borderRadius: '6px', color: '#e4e4e7' }} />
              <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                {couplingData.map((_, i) => <Cell key={i} fill={barColors[i % barColors.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      )}

      {(data?.critical_path ?? []).length > 0 && (
        <Card title="Critical Path (PageRank)" padded={false}>
          <div className="divide-y divide-zinc-800/20 px-5 py-2">
            {data!.critical_path.slice(0, 8).map((entry, i) => (
              <div key={entry.node_id} className="flex items-center gap-3 py-2 text-[13px]">
                <span className="w-5 text-right text-[11px] tabular-nums text-zinc-600">{i + 1}</span>
                <span className="flex-1 truncate text-zinc-300">{entry.label}</span>
                <span className="tabular-nums text-zinc-500">{entry.score.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      <Card title="Performance Insights">
        <InsightFeed insights={data?.insights ?? []} />
      </Card>
    </div>
  )
}
