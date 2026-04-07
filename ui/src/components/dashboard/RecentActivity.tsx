import { Activity, Clock } from 'lucide-react'
import { useMemories } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { getTypeColors } from '../../utils/colors'
import { Card } from '../shared/Card'

function formatRelative(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 60) return `${mins}m`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h`
  const days = Math.floor(hours / 24)
  return `${days}d`
}

export function RecentActivity() {
  const namespace = useNamespaceStore((s) => s.active)
  const { data, isLoading } = useMemories({ namespace: namespace ?? undefined, limit: 12 })

  return (
    <Card
      title="Recent Activity"
      icon={<Activity size={14} className="text-zinc-500" />}
      actions={<span className="text-[11px] text-zinc-600">{data?.total ?? 0} total</span>}
      padded={false}
    >
      <div>
        {isLoading ? (
          <div className="space-y-0 divide-y divide-zinc-800/30">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="flex items-start gap-3 px-5 py-3">
                <span className="mt-1 h-5 w-14 animate-pulse rounded-md bg-zinc-700/30" />
                <span className="h-4 w-full animate-pulse rounded-md bg-zinc-700/20" />
              </div>
            ))}
          </div>
        ) : !data?.memories?.length ? (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="rounded-full bg-zinc-800/30 p-3">
              <Activity size={20} className="text-zinc-600" />
            </div>
            <p className="mt-3 text-[13px] text-zinc-500">No memories captured yet</p>
          </div>
        ) : (
          <div className="divide-y divide-zinc-800/20">
            {data.memories.map((m) => {
              const cfg = getTypeColors(m.memory_type)
              return (
                <div key={m.id} className="group flex items-start gap-3 px-5 py-2.5 transition-colors hover:bg-zinc-700/10">
                  <div className="mt-1 flex shrink-0 items-center gap-2">
                    <span className={`h-1.5 w-1.5 rounded-full ${cfg.dot}`} />
                    <span className={`rounded-md px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${cfg.bg} ${cfg.text}`}>
                      {m.memory_type.slice(0, 3)}
                    </span>
                  </div>
                  <p className="min-w-0 flex-1 line-clamp-2 text-[13px] leading-relaxed text-zinc-300 group-hover:text-zinc-200">
                    {m.content}
                  </p>
                  <span className="mt-0.5 flex shrink-0 items-center gap-1 text-[11px] text-zinc-600">
                    <Clock size={10} />
                    {formatRelative(m.created_at)}
                  </span>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </Card>
  )
}
