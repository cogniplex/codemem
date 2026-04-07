import { StatsCards } from './StatsCards'
import { RecentActivity } from './RecentActivity'
import { TypeDistribution } from './TypeDistribution'
import { ConsolidationSection } from './ConsolidationSection'

export function DashboardView() {
  return (
    <div className="space-y-6">
      <StatsCards />

      <div className="grid gap-6 lg:grid-cols-2">
        <RecentActivity />
        <TypeDistribution />
      </div>

      <ConsolidationSection />
    </div>
  )
}
