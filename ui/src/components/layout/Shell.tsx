import { Navbar } from './Navbar'
import { SearchModal } from './SearchModal'
import { useUiStore } from '../../stores/ui'
import { DashboardView } from '../dashboard/DashboardView'
import { GraphView } from '../graph/GraphView'
import { MemoryBrowser } from '../memories/MemoryBrowser'
import { InsightsView } from '../insights/InsightsView'
import { TemporalView } from '../temporal/TemporalView'

const views: Record<string, React.FC> = {
  dashboard: DashboardView,
  graph: GraphView,
  memories: MemoryBrowser,
  insights: InsightsView,
  temporal: TemporalView,
}

const fullBleedViews = new Set(['graph'])

export function Shell() {
  const { activeView } = useUiStore()
  const View = views[activeView] ?? views.dashboard
  const isFullBleed = fullBleedViews.has(activeView)

  return (
    <div className="flex h-screen flex-col bg-zinc-950 text-zinc-300 antialiased">
      <Navbar />
      <main className={`flex-1 overflow-y-auto ${isFullBleed ? '' : 'p-6'}`}>
        <View />
      </main>
      <SearchModal />
    </div>
  )
}
