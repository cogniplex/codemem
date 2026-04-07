import { StatsCards } from './StatsCards'
import { RecentActivity } from './RecentActivity'
import { ConsolidationSection } from './ConsolidationSection'

export function DashboardView() {
  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <StatsCards />

      <div className="grid gap-6 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <RecentActivity />
        </div>
        <div className="lg:col-span-2">
          <ConsolidationSection />
        </div>
      </div>
    </div>
  )
}
