import { Brain, GitFork, Network, Timer } from 'lucide-react'
import { useStats } from '../../api/hooks'

const cards = [
  { key: 'memory_count', label: 'Memories', icon: Brain, headerBg: 'bg-violet-950/60', bodyBg: 'bg-violet-900/15', iconColor: 'text-violet-400' },
  { key: 'node_count', label: 'Graph Nodes', icon: Network, headerBg: 'bg-cyan-950/60', bodyBg: 'bg-cyan-900/15', iconColor: 'text-cyan-400' },
  { key: 'edge_count', label: 'Edges', icon: GitFork, headerBg: 'bg-emerald-950/60', bodyBg: 'bg-emerald-900/15', iconColor: 'text-emerald-400' },
  { key: 'session_count', label: 'Sessions', icon: Timer, headerBg: 'bg-amber-950/60', bodyBg: 'bg-amber-900/15', iconColor: 'text-amber-400' },
] as const

export function StatsCards() {
  const { data, isLoading } = useStats()

  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      {cards.map(({ key, label, icon: Icon, headerBg, bodyBg, iconColor }) => (
        <div key={key} className="overflow-hidden rounded-xl border border-zinc-800/40">
          <div className={`flex items-center gap-2 px-4 py-2 ${headerBg}`}>
            <Icon size={13} className={iconColor} strokeWidth={1.5} />
            <span className="text-[13px] font-medium text-zinc-300">{label}</span>
          </div>
          <div className={`px-4 py-3 ${bodyBg}`}>
            <p className="text-3xl font-bold tabular-nums tracking-tight text-zinc-50">
              {isLoading ? (
                <span className="inline-block h-8 w-20 animate-pulse rounded-lg bg-zinc-800/40" />
              ) : (
                (data?.[key] ?? 0).toLocaleString()
              )}
            </p>
          </div>
        </div>
      ))}
    </div>
  )
}
