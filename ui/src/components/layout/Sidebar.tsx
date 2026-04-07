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
import { useStats } from '../../api/hooks'

const navItems = [
  { key: 'dashboard' as const, label: 'Dashboard', icon: LayoutDashboard },
  { key: 'graph' as const, label: 'Graph', icon: Network },
  { key: 'memories' as const, label: 'Memories', icon: Brain },
  { key: 'insights' as const, label: 'Insights', icon: Lightbulb },
  { key: 'temporal' as const, label: 'Temporal', icon: Clock },
]

export function Sidebar() {
  const { sidebarCollapsed, toggleSidebar, activeView, setActiveView } = useUiStore()
  const { data: stats } = useStats()

  return (
    <aside
      className={`flex flex-col bg-zinc-950 transition-all duration-200 ${
        sidebarCollapsed ? 'w-[56px]' : 'w-56'
      }`}
    >
      {/* Logo area */}
      <div className="flex h-12 items-center px-3">
        {!sidebarCollapsed && (
          <div className="flex items-center gap-2.5 pl-1">
            <div className="flex h-6 w-6 items-center justify-center rounded-md bg-violet-500/15">
              <Brain size={13} className="text-violet-400" />
            </div>
            <span className="text-[13px] font-semibold tracking-wide text-zinc-100">
              codemem
            </span>
          </div>
        )}
        <button
          onClick={toggleSidebar}
          className={`rounded-md p-1.5 text-zinc-600 transition-colors hover:bg-zinc-800/50 hover:text-zinc-400 ${
            sidebarCollapsed ? 'mx-auto' : 'ml-auto'
          }`}
        >
          {sidebarCollapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}
        </button>
      </div>

      {/* Nav */}
      <nav className="flex-1 space-y-0.5 px-2 pt-2">
        {navItems.map(({ key, label, icon: Icon }) => {
          const active = activeView === key
          return (
            <button
              key={key}
              onClick={() => setActiveView(key)}
              className={`group relative flex w-full items-center gap-3 rounded-lg px-3 py-[9px] text-[13px] transition-all duration-150 ${
                active
                  ? 'bg-zinc-800/80 font-medium text-zinc-100'
                  : 'text-zinc-500 hover:bg-zinc-800/40 hover:text-zinc-300'
              }`}
            >
              {active && (
                <span className="absolute left-0 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-r-full bg-violet-500" />
              )}
              <Icon size={16} strokeWidth={active ? 2 : 1.5} />
              {!sidebarCollapsed && <span>{label}</span>}
            </button>
          )
        })}
      </nav>

      {/* Bottom stats */}
      {!sidebarCollapsed && stats && (
        <div className="border-t border-zinc-800/40 px-4 py-3">
          <div className="space-y-1.5">
            <StatRow label="Memories" value={stats.memory_count} />
            <StatRow label="Nodes" value={stats.node_count} />
            <StatRow label="Edges" value={stats.edge_count} />
          </div>
        </div>
      )}
    </aside>
  )
}

function StatRow({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[11px] text-zinc-600">{label}</span>
      <span className="text-[11px] tabular-nums text-zinc-500">{value.toLocaleString()}</span>
    </div>
  )
}
