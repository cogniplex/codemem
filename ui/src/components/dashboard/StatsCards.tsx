import { Brain, GitFork, Network, Timer } from 'lucide-react'
import { useStats } from '../../api/hooks'

const cards = [
  {
    key: 'memory_count',
    label: 'Memories',
    icon: Brain,
    gradient: 'from-violet-500/20 via-violet-500/5 to-transparent',
    iconBg: 'bg-violet-500/10',
    iconColor: 'text-violet-400',
    glow: 'shadow-violet-500/5',
  },
  {
    key: 'node_count',
    label: 'Graph Nodes',
    icon: Network,
    gradient: 'from-cyan-500/20 via-cyan-500/5 to-transparent',
    iconBg: 'bg-cyan-500/10',
    iconColor: 'text-cyan-400',
    glow: 'shadow-cyan-500/5',
  },
  {
    key: 'edge_count',
    label: 'Edges',
    icon: GitFork,
    gradient: 'from-emerald-500/20 via-emerald-500/5 to-transparent',
    iconBg: 'bg-emerald-500/10',
    iconColor: 'text-emerald-400',
    glow: 'shadow-emerald-500/5',
  },
  {
    key: 'session_count',
    label: 'Sessions',
    icon: Timer,
    gradient: 'from-amber-500/20 via-amber-500/5 to-transparent',
    iconBg: 'bg-amber-500/10',
    iconColor: 'text-amber-400',
    glow: 'shadow-amber-500/5',
  },
] as const

export function StatsCards() {
  const { data, isLoading } = useStats()

  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      {cards.map(({ key, label, icon: Icon, gradient, iconBg, iconColor, glow }) => (
        <div
          key={key}
          className={`group relative overflow-hidden rounded-xl border border-zinc-800/50 bg-zinc-900 shadow-lg ${glow} transition-all duration-300 hover:border-zinc-700/60 hover:shadow-xl`}
        >
          {/* Subtle gradient overlay */}
          <div className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${gradient} opacity-60`} />

          <div className="relative p-5">
            <div className="flex items-center justify-between">
              <span className="text-[13px] font-medium text-zinc-400">{label}</span>
              <div className={`rounded-lg p-2 ${iconBg} ${iconColor}`}>
                <Icon size={15} strokeWidth={1.5} />
              </div>
            </div>
            <p className="mt-3 text-3xl font-bold tabular-nums tracking-tight text-zinc-50">
              {isLoading ? (
                <span className="inline-block h-8 w-20 animate-pulse rounded-lg bg-zinc-800/80" />
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
