import { Brain, GitFork, Network, Timer } from 'lucide-react'
import { useStats } from '../../api/hooks'
import { MetricCard } from '../shared/Card'
import { RecentActivity } from './RecentActivity'
import { ConsolidationSection } from './ConsolidationSection'
import { TemporalSection } from './TemporalSection'
import { InsightsSection } from './InsightsSection'

export function DashboardView() {
  const { data: stats, isLoading } = useStats()

  return (
    <div className="mx-auto max-w-7xl space-y-8">
      {/* ── Core stats ── */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <MetricCard label="Memories" value={stats?.memory_count ?? 0} icon={<Brain size={13} className="text-violet-400" />} isLoading={isLoading} />
        <MetricCard label="Graph Nodes" value={stats?.node_count ?? 0} icon={<Network size={13} className="text-cyan-400" />} isLoading={isLoading} />
        <MetricCard label="Edges" value={stats?.edge_count ?? 0} icon={<GitFork size={13} className="text-emerald-400" />} isLoading={isLoading} />
        <MetricCard label="Sessions" value={stats?.session_count ?? 0} icon={<Timer size={13} className="text-amber-400" />} isLoading={isLoading} />
      </div>

      {/* ── Activity + Consolidation ── */}
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <RecentActivity />
        </div>
        <ConsolidationSection />
      </div>

      {/* ── Insights: charts + feeds ── */}
      <InsightsSection />

      {/* ── Temporal: drift + stale ── */}
      <TemporalSection />
    </div>
  )
}
