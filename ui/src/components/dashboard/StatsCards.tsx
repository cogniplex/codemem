import { Brain, GitFork, Network, Timer } from 'lucide-react'
import { useStats } from '../../api/hooks'

const cards = [
  { key: 'memory_count', label: 'Memories', icon: Brain, accent: 'bg-violet-500/10 text-violet-400' },
  { key: 'node_count', label: 'Graph Nodes', icon: Network, accent: 'bg-cyan-500/10 text-cyan-400' },
  { key: 'edge_count', label: 'Edges', icon: GitFork, accent: 'bg-emerald-500/10 text-emerald-400' },
  { key: 'session_count', label: 'Sessions', icon: Timer, accent: 'bg-amber-500/10 text-amber-400' },
] as const

export function StatsCards() {
  const { data, isLoading } = useStats()

  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      {cards.map(({ key, label, icon: Icon, accent }) => (
        <div
          key={key}
          className="rounded-xl border border-zinc-800/60 bg-zinc-900 p-5"
        >
          <div className="flex items-center gap-3">
            <div className={`rounded-lg p-2 ${accent}`}>
              <Icon size={16} strokeWidth={1.5} />
            </div>
            <span className="text-[13px] font-medium text-zinc-400">{label}</span>
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
