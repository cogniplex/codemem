import { StatsCards } from './StatsCards'
import { RecentActivity } from './RecentActivity'
import { ConsolidationSection } from './ConsolidationSection'
import { TemporalSection } from './TemporalSection'
import { InsightsSection } from './InsightsSection'

export function DashboardView() {
  return (
    <div className="mx-auto max-w-7xl space-y-10">
      {/* Row 1: Stats + Quick metrics from insights (all numbers at the top) */}
      <StatsCards />

      {/* Row 2: Two-column — left: recent activity, right: consolidation stacked with contributors */}
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <RecentActivity />
        </div>
        <div className="space-y-6">
          <ConsolidationSection />
        </div>
      </div>

      {/* Row 3: Insights — charts + feeds */}
      <InsightsSection />

      {/* Row 4: Temporal — drift + stale */}
      <TemporalSection />
    </div>
  )
}
