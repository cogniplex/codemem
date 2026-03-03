import { Sidebar } from './Sidebar'
import { Header } from './Header'
import { SearchModal } from './SearchModal'
import { useUiStore } from '../../stores/ui'
import { DashboardView } from '../dashboard/DashboardView'
import { GraphView } from '../graph/GraphView'
import { MemoryBrowser } from '../memories/MemoryBrowser'
import { RepoManager } from '../repos/RepoManager'
import { TimelineView } from '../timeline/TimelineView'
import { AgentRunner } from '../agents/AgentRunner'
import { InsightsView } from '../insights/InsightsView'
import { SettingsView } from '../settings/SettingsView'

const views: Record<string, React.FC> = {
  dashboard: DashboardView,
  repos: RepoManager,
  graph: GraphView,
  memories: MemoryBrowser,
  timeline: TimelineView,
  agents: AgentRunner,
  insights: InsightsView,
  settings: SettingsView,
}

export function Shell() {
  const { activeView } = useUiStore()
  const View = views[activeView] ?? views.dashboard

  return (
    <div className="flex h-screen bg-zinc-900 text-zinc-200">
      <Sidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto p-6">
          <View />
        </main>
      </div>
      <SearchModal />
    </div>
  )
}
