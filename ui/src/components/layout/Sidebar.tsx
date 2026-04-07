import {
  LayoutDashboard,
  Network,
  Brain,
  Lightbulb,
  Clock,
  PanelLeftClose,
  PanelLeftOpen,
} from 'lucide-react'
import { useUiStore } from '../../stores/ui'

const navItems = [
  { key: 'dashboard' as const, label: 'Dashboard', icon: LayoutDashboard },
  { key: 'graph' as const, label: 'Graph', icon: Network },
  { key: 'memories' as const, label: 'Memories', icon: Brain },
  { key: 'insights' as const, label: 'Insights', icon: Lightbulb },
  { key: 'temporal' as const, label: 'Temporal', icon: Clock },
]

export function Sidebar() {
  const { sidebarCollapsed, toggleSidebar, activeView, setActiveView } = useUiStore()

  return (
    <aside
      className={`flex flex-col border-r border-zinc-800/60 bg-zinc-950 transition-all duration-200 ${
        sidebarCollapsed ? 'w-[52px]' : 'w-52'
      }`}
    >
      {/* Logo */}
      <div className="flex h-14 items-center border-b border-zinc-800/60 px-3">
        {!sidebarCollapsed && (
          <span className="text-[13px] font-semibold tracking-wider text-zinc-200">
            codemem
          </span>
        )}
        <button
          onClick={toggleSidebar}
          className={`rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800/50 hover:text-zinc-300 ${
            sidebarCollapsed ? 'mx-auto' : 'ml-auto'
          }`}
        >
          {sidebarCollapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}
        </button>
      </div>

      {/* Nav */}
      <nav className="flex-1 space-y-0.5 px-2 py-3">
        {navItems.map(({ key, label, icon: Icon }) => {
          const active = activeView === key
          return (
            <button
              key={key}
              onClick={() => setActiveView(key)}
              className={`group relative flex w-full items-center gap-3 rounded-lg px-3 py-2 text-[13px] font-medium transition-colors ${
                active
                  ? 'bg-zinc-800/70 text-zinc-100'
                  : 'text-zinc-500 hover:bg-zinc-800/40 hover:text-zinc-300'
              }`}
            >
              {active && (
                <span className="absolute left-0 top-1/2 h-4 w-[3px] -translate-y-1/2 rounded-r-full bg-violet-500" />
              )}
              <Icon size={16} strokeWidth={active ? 2 : 1.5} />
              {!sidebarCollapsed && <span>{label}</span>}
            </button>
          )
        })}
      </nav>
    </aside>
  )
}
