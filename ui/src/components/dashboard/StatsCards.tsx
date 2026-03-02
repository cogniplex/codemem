import { Brain, GitFork, Network, Timer } from 'lucide-react'
import { useStats } from '../../api/hooks'

const cards = [
  { key: 'memory_count', label: 'Memories', icon: Brain, color: 'text-violet-400' },
  { key: 'node_count', label: 'Graph Nodes', icon: Network, color: 'text-cyan-400' },
  { key: 'edge_count', label: 'Edges', icon: GitFork, color: 'text-emerald-400' },
  { key: 'session_count', label: 'Sessions', icon: Timer, color: 'text-amber-400' },
] as const

export function StatsCards() {
  const { data, isLoading } = useStats()

  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      {cards.map(({ key, label, icon: Icon, color }) => (
        <div
          key={key}
          className="rounded-lg border border-zinc-800 bg-zinc-850 p-5"
        >
          <div className="flex items-center gap-3">
            <div className={`rounded-md bg-zinc-800 p-2 ${color}`}>
              <Icon size={18} />
            </div>
            <span className="text-sm text-zinc-400">{label}</span>
          </div>
          <p className="mt-3 text-2xl font-semibold tabular-nums text-zinc-100">
            {isLoading ? (
              <span className="inline-block h-7 w-16 animate-pulse rounded bg-zinc-800" />
            ) : (
              (data?.[key] ?? 0).toLocaleString()
            )}
          </p>
        </div>
      ))}
    </div>
  )
}
