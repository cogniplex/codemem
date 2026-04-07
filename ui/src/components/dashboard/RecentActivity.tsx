import { Activity } from 'lucide-react'
import { useMemories } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'

const typeColors: Record<string, string> = {
  decision: 'bg-violet-500/15 text-violet-400',
  pattern: 'bg-cyan-500/15 text-cyan-400',
  preference: 'bg-amber-500/15 text-amber-400',
  style: 'bg-pink-500/15 text-pink-400',
  habit: 'bg-emerald-500/15 text-emerald-400',
  insight: 'bg-blue-500/15 text-blue-400',
  context: 'bg-zinc-500/15 text-zinc-400',
}

export function RecentActivity() {
  const namespace = useNamespaceStore((s) => s.active)
  const { data, isLoading } = useMemories({
    namespace: namespace ?? undefined,
    limit: 15,
  })

  return (
    <div className="rounded-xl border border-zinc-800/60 bg-zinc-900">
      <div className="flex items-center gap-2 border-b border-zinc-800/60 px-5 py-3">
        <Activity size={14} className="text-zinc-500" />
        <h3 className="text-[13px] font-medium text-zinc-100">Recent Activity</h3>
      </div>
      <div className="divide-y divide-zinc-800/40">
        {isLoading ? (
          Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="flex items-start gap-3 px-5 py-3">
              <span className="mt-1 h-5 w-14 animate-pulse rounded bg-zinc-800" />
              <span className="h-4 w-full animate-pulse rounded bg-zinc-800" />
            </div>
          ))
        ) : !data?.memories?.length ? (
          <p className="px-5 py-8 text-center text-[13px] text-zinc-600">No memories yet</p>
        ) : (
          data.memories.map((m) => (
            <div key={m.id} className="flex items-start gap-3 px-5 py-2.5">
              <span
                className={`mt-0.5 shrink-0 rounded-md px-1.5 py-0.5 text-[11px] font-medium ${
                  typeColors[m.memory_type] ?? typeColors.context
                }`}
              >
                {m.memory_type}
              </span>
              <p className="min-w-0 line-clamp-2 break-words text-[13px] text-zinc-300">{m.content}</p>
              <time className="ml-auto shrink-0 text-[11px] text-zinc-600">
                {new Date(m.created_at).toLocaleDateString()}
              </time>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
