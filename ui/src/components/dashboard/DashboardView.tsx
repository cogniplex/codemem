import { Brain, GitFork, Network, Timer, FileText, Users, FileSearch, Layers, ShieldAlert, Link2 } from 'lucide-react'
import {
  useStats, useActivityInsights, useCodeHealthInsights,
  useSecurityInsights, usePerformanceInsights,
} from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { MetricCard } from '../shared/Card'
import { RecentActivity } from './RecentActivity'
import { ConsolidationSection } from './ConsolidationSection'
import { TemporalSection } from './TemporalSection'
import { InsightsSection } from './InsightsSection'

export function DashboardView() {
  const namespace = useNamespaceStore((s) => s.active)
  const ns = namespace ? { namespace } : undefined
  const { data: stats, isLoading: statsLoading } = useStats()
  const { data: activity, isLoading: actLoading } = useActivityInsights(ns)
  const { data: health, isLoading: healthLoading } = useCodeHealthInsights(ns)
  const { data: security, isLoading: secLoading } = useSecurityInsights(ns)
  const { data: perf, isLoading: perfLoading } = usePerformanceInsights(ns)

  return (
    <div className="mx-auto max-w-7xl space-y-8">
      {/* ── All metrics at the top ── */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-5 lg:grid-cols-5">
        <MetricCard label="Memories" value={stats?.memory_count ?? 0} icon={<Brain size={13} className="text-violet-400" />} isLoading={statsLoading} />
        <MetricCard label="Graph Nodes" value={stats?.node_count ?? 0} icon={<Network size={13} className="text-cyan-400" />} isLoading={statsLoading} />
        <MetricCard label="Edges" value={stats?.edge_count ?? 0} icon={<GitFork size={13} className="text-emerald-400" />} isLoading={statsLoading} />
        <MetricCard label="Sessions" value={stats?.session_count ?? 0} icon={<Timer size={13} className="text-amber-400" />} isLoading={statsLoading} />
        <MetricCard label="Contributors" value={activity?.git_summary.top_authors.length ?? 0} icon={<Users size={13} className="text-emerald-400" />} isLoading={actLoading} />
        <MetricCard label="Files Analyzed" value={activity?.git_summary.total_annotated_files ?? 0} icon={<FileText size={13} className="text-violet-400" />} isLoading={actLoading} />
        <MetricCard label="Hotspots" value={health?.file_hotspots.length ?? 0} icon={<FileSearch size={13} className="text-amber-400" />} isLoading={healthLoading} />
        <MetricCard label="Communities" value={health?.community_count ?? 0} icon={<Layers size={13} className="text-violet-400" />} isLoading={healthLoading} />
        <MetricCard label="Security" value={security?.sensitive_file_count ?? 0} icon={<ShieldAlert size={13} className="text-red-400" />} isLoading={secLoading} />
        <MetricCard label="Coupling" value={perf?.high_coupling_nodes.length ?? 0} icon={<Link2 size={13} className="text-amber-400" />} isLoading={perfLoading} />
      </div>

      {/* ── Activity + Consolidation ── */}
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <RecentActivity />
        </div>
        <ConsolidationSection />
      </div>

      {/* ── Charts + Insight feeds ── */}
      <InsightsSection />

      {/* ── Temporal: Drift + Stale ── */}
      <TemporalSection />
    </div>
  )
}
