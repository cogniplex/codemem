import { useMemo } from 'react'
import {
  Brain, GitFork, Network, Timer,
  FileText, Users, FileSearch, Layers, ShieldAlert, Link2,
  ArrowLeftRight, Plus, Minus,
} from 'lucide-react'
import {
  useStats, useActivityInsights, useCodeHealthInsights,
  useSecurityInsights, usePerformanceInsights, useDrift,
} from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { MetricCard } from '../shared/Card'
import { RecentActivity } from './RecentActivity'
import { ConsolidationSection } from './ConsolidationSection'
import { TemporalSection } from './TemporalSection'
import { InsightsSection } from './InsightsSection'

function daysAgo(n: number): string {
  const d = new Date(); d.setDate(d.getDate() - n); return d.toISOString()
}

export function DashboardView() {
  const namespace = useNamespaceStore((s) => s.active)
  const ns = namespace ? { namespace } : undefined
  const { data: stats, isLoading: sL } = useStats()
  const { data: activity, isLoading: aL } = useActivityInsights(ns)
  const { data: health, isLoading: hL } = useCodeHealthInsights(ns)
  const { data: security, isLoading: secL } = useSecurityInsights(ns)
  const { data: perf, isLoading: pL } = usePerformanceInsights(ns)

  const driftFrom = useMemo(() => daysAgo(90), [])
  const driftTo = useMemo(() => new Date().toISOString(), [])
  const { data: drift, isLoading: dL } = useDrift(driftFrom, driftTo, namespace ?? undefined)

  return (
    <div className="mx-auto max-w-7xl space-y-8">
      {/* ── All metrics — one place ── */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-5 xl:grid-cols-7">
        <MetricCard label="Memories" value={stats?.memory_count ?? 0} icon={<Brain size={13} className="text-violet-400" />} isLoading={sL} />
        <MetricCard label="Graph Nodes" value={stats?.node_count ?? 0} icon={<Network size={13} className="text-cyan-400" />} isLoading={sL} />
        <MetricCard label="Edges" value={stats?.edge_count ?? 0} icon={<GitFork size={13} className="text-emerald-400" />} isLoading={sL} />
        <MetricCard label="Sessions" value={stats?.session_count ?? 0} icon={<Timer size={13} className="text-amber-400" />} isLoading={sL} />
        <MetricCard label="Contributors" value={activity?.git_summary.top_authors.length ?? 0} icon={<Users size={13} className="text-emerald-400" />} isLoading={aL} />
        <MetricCard label="Files Analyzed" value={activity?.git_summary.total_annotated_files ?? 0} icon={<FileText size={13} className="text-violet-400" />} isLoading={aL} />
        <MetricCard label="Hotspots" value={health?.file_hotspots.length ?? 0} icon={<FileSearch size={13} className="text-amber-400" />} isLoading={hL} />
        <MetricCard label="Communities" value={health?.community_count ?? 0} icon={<Layers size={13} className="text-violet-400" />} isLoading={hL} />
        <MetricCard label="Security" value={security?.sensitive_file_count ?? 0} icon={<ShieldAlert size={13} className="text-red-400" />} isLoading={secL} />
        <MetricCard label="High Coupling" value={perf?.high_coupling_nodes.length ?? 0} icon={<Link2 size={13} className="text-amber-400" />} isLoading={pL} />
        <MetricCard label="Cross-Module" value={drift?.new_cross_module_edges ?? 0} icon={<ArrowLeftRight size={13} className="text-violet-400" />} isLoading={dL} />
        <MetricCard label="Files Added" value={drift?.added_files ?? 0} icon={<Plus size={13} className="text-emerald-400" />} isLoading={dL} />
        <MetricCard label="Files Removed" value={drift?.removed_files ?? 0} icon={<Minus size={13} className="text-red-400" />} isLoading={dL} />
        <MetricCard label="Co-Changed" value={drift?.coupling_increases?.length ?? 0} icon={<Layers size={13} className="text-amber-400" />} isLoading={dL} />
      </div>

      {/* ── Consolidation (horizontal) ── */}
      <ConsolidationSection />

      {/* ── Recent Activity (full width) ── */}
      <RecentActivity />

      {/* ── Insights: charts + feeds ── */}
      <InsightsSection />

      {/* ── Temporal: drift + stale ── */}
      <TemporalSection />
    </div>
  )
}
