import { FileSearch, Network, Layers } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { useCodeHealthInsights } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { Card, MetricCard } from '../shared/Card'
import type { MemoryItem } from '../../api/types'

function InsightFeed({ insights }: { insights: MemoryItem[] }) {
  if (insights.length === 0) return <p className="py-6 text-center text-[13px] text-zinc-600">No insights yet.</p>

  return (
    <div className="space-y-2">
      {insights.map((item) => (
        <div key={item.id} className="rounded-lg border border-zinc-700/30 bg-zinc-900 px-4 py-3 transition-colors hover:bg-zinc-800/80">
          <div className="flex items-start justify-between gap-3">
            <p className="min-w-0 break-words text-[13px] leading-relaxed text-zinc-300">{item.content}</p>
            <span className="shrink-0 rounded-md bg-emerald-500/10 px-2 py-0.5 text-[11px] font-medium tabular-nums text-emerald-400">
              {(item.importance * 100).toFixed(0)}%
            </span>
          </div>
          <p className="mt-1 text-[11px] text-zinc-600">{new Date(item.created_at).toLocaleDateString()}</p>
        </div>
      ))}
    </div>
  )
}

const barColors = ['#818cf8', '#a78bfa', '#c4b5fd', '#ddd6fe', '#ede9fe', '#f5f3ff']

export function CodeHealthTab() {
  const namespace = useNamespaceStore((s) => s.active)
  const { data, isLoading } = useCodeHealthInsights(namespace ? { namespace } : undefined)

  const hotspotData = (data?.file_hotspots ?? []).slice(0, 8).map((h) => ({
    name: h.description.split(' — ')[0] ?? h.description,
    count: h.frequency,
  }))

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <MetricCard label="File Hotspots" value={data?.file_hotspots.length ?? 0} icon={<FileSearch size={15} className="text-amber-400" />} isLoading={isLoading} />
        <MetricCard label="PageRank Leaders" value={data?.pagerank_leaders.length ?? 0} icon={<Network size={15} className="text-cyan-400" />} isLoading={isLoading} />
        <MetricCard label="Communities" value={data?.community_count ?? 0} icon={<Layers size={15} className="text-violet-400" />} isLoading={isLoading} />
      </div>

      {hotspotData.length > 0 && (
        <Card title="File Hotspots">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={hotspotData} layout="vertical" margin={{ left: 120 }}>
              <XAxis type="number" tick={{ fill: '#71717a', fontSize: 12 }} />
              <YAxis dataKey="name" type="category" tick={{ fill: '#a1a1aa', fontSize: 11 }} width={120} />
              <Tooltip contentStyle={{ backgroundColor: '#18181b', border: '1px solid #3f3f46', borderRadius: '6px', color: '#e4e4e7' }} />
              <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                {hotspotData.map((_, i) => <Cell key={i} fill={barColors[i % barColors.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      )}

      {(data?.pagerank_leaders ?? []).length > 0 && (
        <Card title="PageRank Leaders" padded={false}>
          <div className="divide-y divide-zinc-800/20 px-5 py-2">
            {data!.pagerank_leaders.slice(0, 8).map((entry, i) => (
              <div key={entry.node_id} className="flex items-center gap-3 py-2 text-[13px]">
                <span className="w-5 text-right text-[11px] tabular-nums text-zinc-600">{i + 1}</span>
                <span className="flex-1 truncate text-zinc-300">{entry.label}</span>
                <span className="tabular-nums text-zinc-500">{entry.score.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      <Card title="Code Health Insights">
        <InsightFeed insights={data?.insights ?? []} />
      </Card>
    </div>
  )
}
