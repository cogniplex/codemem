import {
  GitCommit, FileText, Users, FileSearch, Network, Layers,
  ShieldAlert, Link2, Zap,
} from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import {
  useActivityInsights, useCodeHealthInsights,
  useSecurityInsights, usePerformanceInsights,
} from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { Card, MetricCard } from '../shared/Card'
import type { MemoryItem } from '../../api/types'

function InsightFeed({ insights, accent }: { insights: MemoryItem[]; accent: string }) {
  if (insights.length === 0) return <p className="py-4 text-center text-[13px] text-zinc-600">No data yet.</p>

  return (
    <div className="max-h-[320px] space-y-1.5 overflow-y-auto">
      {insights.slice(0, 8).map((item) => (
        <div key={item.id} className="rounded-lg border border-zinc-700/30 bg-zinc-900 px-3.5 py-2.5 transition-colors hover:bg-zinc-800/80">
          <p className="min-w-0 break-words text-[12px] leading-relaxed text-zinc-300 line-clamp-2">{item.content}</p>
          <div className="mt-1.5 flex items-center justify-between">
            <span className="text-[10px] text-zinc-600">{new Date(item.created_at).toLocaleDateString()}</span>
            <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium tabular-nums ${accent}`}>
              {(item.importance * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      ))}
    </div>
  )
}

function severityBadge(tags: string[]) {
  if (tags.some((t) => t.includes('severity:high'))) return { text: 'HIGH', cls: 'bg-red-500/10 text-red-400' }
  if (tags.some((t) => t.includes('severity:medium'))) return { text: 'MED', cls: 'bg-amber-500/10 text-amber-400' }
  return { text: 'INFO', cls: 'bg-zinc-700/30 text-zinc-400' }
}

const chartColors1 = ['#818cf8', '#a78bfa', '#c4b5fd', '#ddd6fe', '#ede9fe', '#f5f3ff']
const chartColors2 = ['#f87171', '#fb923c', '#fbbf24', '#a3e635', '#34d399', '#22d3ee']

export function InsightsSection() {
  const namespace = useNamespaceStore((s) => s.active)
  const ns = namespace ? { namespace } : undefined

  const { data: activity } = useActivityInsights(ns)
  const { data: health } = useCodeHealthInsights(ns)
  const { data: security } = useSecurityInsights(ns)
  const { data: perf } = usePerformanceInsights(ns)

  const hotspotData = (health?.file_hotspots ?? []).slice(0, 6).map((h) => ({
    name: h.description.split(' — ')[0] ?? h.description, count: h.frequency,
  }))
  const couplingData = (perf?.high_coupling_nodes ?? []).slice(0, 6).map((n) => ({
    name: n.label.length > 20 ? '...' + n.label.slice(-20) : n.label, score: n.coupling_score,
  }))

  return (
    <div className="space-y-6">
      {/* ── Insight metrics ── */}
      <div className="grid grid-cols-3 gap-3 lg:grid-cols-6">
        <MetricCard label="Files Analyzed" value={activity?.git_summary.total_annotated_files ?? 0} icon={<FileText size={13} className="text-violet-400" />} />
        <MetricCard label="Contributors" value={activity?.git_summary.top_authors.length ?? 0} icon={<Users size={13} className="text-emerald-400" />} />
        <MetricCard label="Hotspots" value={health?.file_hotspots.length ?? 0} icon={<FileSearch size={13} className="text-amber-400" />} />
        <MetricCard label="Communities" value={health?.community_count ?? 0} icon={<Layers size={13} className="text-violet-400" />} />
        <MetricCard label="Security" value={security?.sensitive_file_count ?? 0} icon={<ShieldAlert size={13} className="text-red-400" />} />
        <MetricCard label="Coupling" value={perf?.high_coupling_nodes.length ?? 0} icon={<Link2 size={13} className="text-amber-400" />} />
      </div>

      {/* ── Charts + ranked lists ── */}
      <div className="grid gap-6 lg:grid-cols-2">
        {hotspotData.length > 0 ? (
          <Card title="File Hotspots" icon={<FileSearch size={14} className="text-amber-400" />}>
            <ResponsiveContainer width="100%" height={170}>
              <BarChart data={hotspotData} layout="vertical" margin={{ left: 90 }}>
                <XAxis type="number" tick={{ fill: '#71717a', fontSize: 11 }} />
                <YAxis dataKey="name" type="category" tick={{ fill: '#a1a1aa', fontSize: 10 }} width={90} />
                <Tooltip contentStyle={{ backgroundColor: '#18181b', border: '1px solid #3f3f46', borderRadius: '6px', color: '#e4e4e7', fontSize: '12px' }} />
                <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                  {hotspotData.map((_, i) => <Cell key={i} fill={chartColors1[i % chartColors1.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        ) : (
          <Card title="File Hotspots" icon={<FileSearch size={14} className="text-amber-400" />}>
            <p className="py-6 text-center text-[13px] text-zinc-600">No hotspot data.</p>
          </Card>
        )}

        {couplingData.length > 0 ? (
          <Card title="High-Coupling Nodes" icon={<Link2 size={14} className="text-red-400" />}>
            <ResponsiveContainer width="100%" height={170}>
              <BarChart data={couplingData} layout="vertical" margin={{ left: 90 }}>
                <XAxis type="number" tick={{ fill: '#71717a', fontSize: 11 }} />
                <YAxis dataKey="name" type="category" tick={{ fill: '#a1a1aa', fontSize: 10 }} width={90} />
                <Tooltip contentStyle={{ backgroundColor: '#18181b', border: '1px solid #3f3f46', borderRadius: '6px', color: '#e4e4e7', fontSize: '12px' }} />
                <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                  {couplingData.map((_, i) => <Cell key={i} fill={chartColors2[i % chartColors2.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        ) : (
          <Card title="High-Coupling Nodes" icon={<Link2 size={14} className="text-red-400" />}>
            <p className="py-6 text-center text-[13px] text-zinc-600">No coupling data.</p>
          </Card>
        )}
      </div>

      {/* ── Insight feeds — 2x2 grid ── */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Activity" icon={<GitCommit size={14} className="text-cyan-400" />}>
          <InsightFeed insights={activity?.insights ?? []} accent="bg-violet-500/10 text-violet-400" />
        </Card>

        <Card title="Security Findings" icon={<ShieldAlert size={14} className="text-red-400" />}>
          {(security?.insights ?? []).length === 0 ? (
            <p className="py-4 text-center text-[13px] text-zinc-600">No security findings.</p>
          ) : (
            <div className="space-y-1.5">
              {(security?.insights ?? []).slice(0, 8).map((item) => {
                const b = severityBadge(item.tags)
                return (
                  <div key={item.id} className="rounded-lg border border-zinc-700/30 bg-zinc-900 px-3.5 py-2.5 transition-colors hover:bg-zinc-800/80">
                    <p className="min-w-0 break-words text-[12px] leading-relaxed text-zinc-300 line-clamp-2">{item.content}</p>
                    <div className="mt-1.5 flex items-center justify-between">
                      <span className="text-[10px] text-zinc-600">{new Date(item.created_at).toLocaleDateString()}</span>
                      <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${b.cls}`}>{b.text}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </Card>

        <Card title="Code Health" icon={<Network size={14} className="text-emerald-400" />}>
          <InsightFeed insights={health?.insights ?? []} accent="bg-emerald-500/10 text-emerald-400" />
        </Card>

        <Card title="Performance" icon={<Zap size={14} className="text-amber-400" />}>
          <InsightFeed insights={perf?.insights ?? []} accent="bg-cyan-500/10 text-cyan-400" />
        </Card>
      </div>

      {/* ── Ranked lists ── */}
      {((health?.pagerank_leaders ?? []).length > 0 || (perf?.critical_path ?? []).length > 0) && (
        <div className="grid gap-6 lg:grid-cols-2">
          {(health?.pagerank_leaders ?? []).length > 0 && (
            <Card title="PageRank Leaders" icon={<Network size={14} className="text-cyan-400" />} padded={false}>
              <div className="divide-y divide-zinc-800/20 px-5 py-1">
                {health!.pagerank_leaders.slice(0, 6).map((e, i) => (
                  <div key={e.node_id} className="flex items-center gap-3 py-2 text-[12px]">
                    <span className="w-4 text-right text-[11px] tabular-nums text-zinc-600">{i + 1}</span>
                    <span className="flex-1 truncate text-zinc-300">{e.label}</span>
                    <span className="tabular-nums text-zinc-500">{e.score.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}
          {(perf?.critical_path ?? []).length > 0 && (
            <Card title="Critical Path" icon={<Zap size={14} className="text-amber-400" />} padded={false}>
              <div className="divide-y divide-zinc-800/20 px-5 py-1">
                {perf!.critical_path.slice(0, 6).map((e, i) => (
                  <div key={e.node_id} className="flex items-center gap-3 py-2 text-[12px]">
                    <span className="w-4 text-right text-[11px] tabular-nums text-zinc-600">{i + 1}</span>
                    <span className="flex-1 truncate text-zinc-300">{e.label}</span>
                    <span className="tabular-nums text-zinc-500">{e.score.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
