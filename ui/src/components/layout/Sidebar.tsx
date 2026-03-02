import {
  LayoutDashboard,
  GitFork,
  Network,
  Brain,
  CalendarClock,
  Bot,
  Lightbulb,
  Settings,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react'
import { useUiStore } from '../../stores/ui'

const navItems = [
  { key: 'dashboard' as const, label: 'Dashboard', icon: LayoutDashboard },
  { key: 'repos' as const, label: 'Repos', icon: GitFork },
  { key: 'graph' as const, label: 'Graph', icon: Network },
  { key: 'memories' as const, label: 'Memories', icon: Brain },
  { key: 'timeline' as const, label: 'Timeline', icon: CalendarClock },
  { key: 'agents' as const, label: 'Agents', icon: Bot },
  { key: 'insights' as const, label: 'Insights', icon: Lightbulb },
  { key: 'settings' as const, label: 'Settings', icon: Settings },
]

export function Sidebar() {
  const { sidebarCollapsed, toggleSidebar, activeView, setActiveView } = useUiStore()

  return (
    <aside
      className={`flex flex-col border-r border-zinc-800 bg-zinc-950 transition-all duration-200 ${
        sidebarCollapsed ? 'w-16' : 'w-56'
      }`}
    >
      <div className="flex h-14 items-center gap-2 border-b border-zinc-800 px-4">
        {!sidebarCollapsed && (
          <span className="text-sm font-semibold tracking-wide text-zinc-100">codemem</span>
        )}
        <button
          onClick={toggleSidebar}
          className="ml-auto rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
        >
          {sidebarCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>

      <nav className="flex-1 space-y-1 p-2">
        {navItems.map(({ key, label, icon: Icon }) => {
          const active = activeView === key
          return (
            <button
              key={key}
              onClick={() => setActiveView(key)}
              className={`flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors ${
                active
                  ? 'bg-zinc-800 text-zinc-100'
                  : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'
              }`}
            >
              <Icon size={18} />
              {!sidebarCollapsed && <span>{label}</span>}
            </button>
          )
        })}
      </nav>
    </aside>
  )
}
